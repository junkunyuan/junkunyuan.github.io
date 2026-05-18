"""Lightweight YAML validation. Raises `SchemaError` listing every problem
found across all files in one pass so the user fixes them together."""

from __future__ import annotations

import re
from typing import Iterable

import yaml


class SchemaError(Exception):
    pass


_DATE_RE = re.compile(r"^\d{8}$")

# Required fields differ between the publications list (the main page) and the
# reading-list domain files. Optional fields are validated only when present.
PUB_REQUIRED = {"title", "author", "date", "venue", "name", "pdf_url"}
PUB_OPTIONAL = {"code_url", "comment", "highlight", "selected"}

DOMAIN_REQUIRED = {"title", "date", "venue", "name", "category"}
DOMAIN_OPTIONAL = {
    "author", "organization", "pdf_url", "code_url", "comment",
    "jupyter_notes", "highlight", "draft", "summary", "details",
}


def _check_paper(p: dict, idx: int, required: set, optional: set,
                  errors: list, *, source: str,
                  domain_categories: set | None = None) -> None:
    label = f"{source}[{idx}] '{p.get('name', '?')}'"

    missing = [k for k in required if not p.get(k)]
    if missing:
        errors.append(f"{label}: missing required field(s): {', '.join(sorted(missing))}")

    allowed = required | optional
    unknown = set(p) - allowed
    if unknown:
        errors.append(f"{label}: unknown field(s): {', '.join(sorted(unknown))}")

    date = p.get("date", "")
    if date and not _DATE_RE.match(str(date)):
        errors.append(f"{label}: invalid date {date!r}, expected YYYYMMDD")

    if domain_categories is not None:
        cat = p.get("category", "")
        cats = cat if isinstance(cat, list) else [cat] if cat else []
        for c in cats:
            if c not in domain_categories:
                errors.append(f"{label}: category {c!r} not declared in domain.categories")

    if p.get("highlight") not in (None, True):
        errors.append(f"{label}: highlight must be `true` or omitted")
    if p.get("selected") not in (None, True):
        errors.append(f"{label}: selected must be `true` or omitted")
    if p.get("draft") not in (None, True, False):
        errors.append(f"{label}: draft must be a bool")


def _check_unique_names(papers: Iterable[dict], errors: list, source: str) -> None:
    seen: dict[str, int] = {}
    for p in papers:
        # Anchors include venue, so the same name across different venues is fine;
        # but the (name, venue) pair must be unique within a file.
        key = f"{p.get('name', '?')}@{p.get('venue', '')}"
        seen[key] = seen.get(key, 0) + 1
    for key, n in seen.items():
        if n > 1:
            errors.append(f"{source}: duplicate (name, venue) {key!r} appears {n} times")


def validate_pub_yaml(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    errors: list = []
    if not isinstance(data, dict) or "papers" not in data:
        return [f"{path}: top-level must have a `papers:` list"]
    papers = data["papers"]
    for i, p in enumerate(papers):
        _check_paper(p, i, PUB_REQUIRED, PUB_OPTIONAL, errors, source=path)
    _check_unique_names(papers, errors, path)
    return errors


def validate_domain_yaml(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    errors: list = []
    if not isinstance(data, dict) or "papers" not in data or "categories" not in data:
        return [f"{path}: top-level must have `categories:` and `papers:`"]

    declared = set(data["categories"])
    papers = data["papers"]
    for i, p in enumerate(papers):
        _check_paper(p, i, DOMAIN_REQUIRED, DOMAIN_OPTIONAL, errors,
                     source=path, domain_categories=declared)
    _check_unique_names(papers, errors, path)
    return errors


def validate_all(pub_path: str, domain_paths: list[str]) -> None:
    errors: list = []
    errors.extend(validate_pub_yaml(pub_path))
    for dp in domain_paths:
        errors.extend(validate_domain_yaml(dp))
    if errors:
        raise SchemaError("YAML validation failed:\n  - " + "\n  - ".join(errors))
