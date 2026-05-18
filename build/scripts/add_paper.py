"""Interactive CLI to append a paper entry to a YAML file.

Usage:
    python -m build.scripts.add_paper                 # append to a reading-list domain
    python -m build.scripts.add_paper --pub           # append to data/publications.yaml

Fields are prompted with defaults. The resulting block is appended at the end
of the relevant `papers:` list. After appending, validation is run."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAIN_DIR = os.path.join(PROJECT_ROOT, "data", "reading_list")
PUB_PATH = os.path.join(PROJECT_ROOT, "data", "publications.yaml")


def _ask(prompt: str, default: str = "", required: bool = False) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        v = input(f"{prompt}{hint}: ").strip() or default
        if v or not required:
            return v
        print("  (required)")


def _ask_bool(prompt: str) -> bool:
    return _ask(f"{prompt} (y/N)").lower() == "y"


def _append_paper(path: str, entry: dict) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.setdefault("papers", []).append(entry)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=10**6)
    print(f"\nAppended to {path}")


def _add_domain_paper() -> None:
    domains = [f for f in sorted(os.listdir(DOMAIN_DIR)) if f.endswith(".yaml")]
    print("Available domains:")
    for i, f in enumerate(domains):
        print(f"  {i}: {f}")
    idx = int(_ask("Pick a domain (number)", required=True))
    path = os.path.join(DOMAIN_DIR, domains[idx])

    with open(path, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f)
    cats = domain.get("categories", [])
    print("\nCategories:")
    for i, c in enumerate(cats):
        print(f"  {i}: {c}")
    cat = cats[int(_ask("Pick a category (number)", required=True))]

    entry: dict = {
        "title": _ask("Title", required=True),
        "author": _ask("Author list (use ** after a name for co-first, ## for corresponding)"),
        "organization": _ask("Organization"),
        "date": _ask("Date (YYYYMMDD)", default=datetime.now().strftime("%Y%m%d"), required=True),
        "venue": _ask("Venue (e.g. ICLR 2026, arXiv 2025)", required=True),
        "pdf_url": _ask("PDF url"),
        "code_url": _ask("Code url"),
        "name": _ask("Short name (used for anchor/filename)", required=True),
        "category": cat,
    }
    if _ask_bool("Highlight as impactful?"):
        entry["highlight"] = True
    if _ask_bool("Save as draft (hidden from build)?"):
        entry["draft"] = True

    summary = _ask("Summary (single line; leave empty to skip)")
    if summary:
        entry["summary"] = summary

    # Drop empty optional fields.
    for k in list(entry):
        if entry[k] in ("", None):
            del entry[k]

    _append_paper(path, entry)


def _add_pub_paper() -> None:
    entry: dict = {
        "title": _ask("Title", required=True),
        "author": _ask("Author list", required=True),
        "date": _ask("Date (YYYYMMDD)", default=datetime.now().strftime("%Y%m%d"), required=True),
        "venue": _ask("Venue", required=True),
        "name": _ask("Short name", required=True),
        "pdf_url": _ask("PDF url", required=True),
        "code_url": _ask("Code url"),
    }
    if _ask_bool("Highlight as impactful?"):
        entry["highlight"] = True
    if _ask_bool("Add a comment shown under the entry?"):
        entry["comment"] = _ask("Comment")
    for k in list(entry):
        if entry[k] in ("", None):
            del entry[k]
    _append_paper(PUB_PATH, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a paper entry to a YAML file.")
    parser.add_argument("--pub", action="store_true", help="Append to data/publications.yaml")
    args = parser.parse_args()

    if args.pub:
        _add_pub_paper()
    else:
        _add_domain_paper()

    print("Running validation…")
    from build.scripts.validate import main as validate_main
    return validate_main()


if __name__ == "__main__":
    sys.exit(main())
