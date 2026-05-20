"""Generate the paper-reading-list pages: one HTML per domain plus the
top-level index. Domain data lives in `data/reading_list/<name>.yaml`."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List

import yaml

from build.lib.main_content import MAIN_CONTENT
from build.lib.templates import (
    BACK_TO_TOP,
    PAGE_FOOT,
    PAPER_TOGGLE_SCRIPT,
    READING_LIST_HEAD_EXTRA,
    SEARCH_SCRIPT,
    THEME_TOGGLE,
    format_now,
    page_head,
)
from build.lib.utils import (
    accent_cycle,
    build_index_item,
    convert_fig_cap_to_figure,
    get_venue_full,
    paper_anchor,
    slugify,
    split_venue,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "reading_list")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper_reading_list")
DOMAIN_FILES = ["research", "coding", "fundamental", "rl", "cv", "nlp", "mm"]

# Domains where order in YAML is meaningful (resource pages, not papers).
PRESERVE_ORDER_TITLES = {"Research", "Coding and Engineering"}


def _load_domain(name: str) -> Dict[str, Any]:
    with open(os.path.join(DATA_DIR, f"{name}.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _drop_drafts(papers: List[dict]) -> List[dict]:
    return [p for p in papers if not p.get("draft")]


def _paper_categories(paper: dict) -> List[str]:
    cat = paper.get("category", "")
    if isinstance(cat, list):
        return list(cat)
    return [cat] if cat else []


def _filter_and_sort(papers: List[dict], category: str, domain_title: str) -> List[dict]:
    subset = [p for p in papers if category in _paper_categories(p)]
    if domain_title not in PRESERVE_ORDER_TITLES:
        subset.sort(key=lambda p: p["date"], reverse=True)
    return subset


def _format_venue(venue: str) -> tuple[str, str]:
    """Return (abbr_html, year_html). year_html is empty when no year is present."""
    abbr, year = split_venue(venue)
    abbr_html = f'<b>{abbr}</b>'
    year_html = year if year else ""
    return abbr_html, year_html


def _format_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%b %d, %Y")
    except ValueError:
        return ""


def _search_blob(paper: dict) -> str:
    """Pre-computed lowercase text used by the client-side filter."""
    return " ".join(
        str(paper.get(k, "")) for k in ("title", "author", "organization", "venue", "name", "category")
    ).lower()


def _paper_card(paper: dict, accent_var: str, category: str) -> str:
    raw_title = paper["title"]
    debug_open = raw_title.startswith("aa")
    display_title = raw_title[2:].lstrip() if debug_open else raw_title
    name_cls = "paper__name paper__name--highlight" if paper.get("highlight") else "paper__name"
    venue_full = get_venue_full(paper["venue"])
    abbr_html, year_html = _format_venue(paper["venue"])
    if venue_full:
        venue_label = f'{venue_full} ({abbr_html})'
        if year_html:
            venue_label += f', {year_html}'
    else:
        venue_label = abbr_html + (f', {year_html}' if year_html else "")

    name_html = f'<span class="{name_cls}">{paper["name"]}</span>'
    pdf_link = (
        f'<a href="{paper["pdf_url"]}">{name_html}</a>'
        if paper.get("pdf_url") else name_html
    )
    code_link = ""
    if paper.get("code_url"):
        code_link = (
            f' &nbsp;&nbsp;<span class="divider">|</span>&nbsp;&nbsp; '
            f'<a href="{paper["code_url"]}">code</a>'
        )

    anchor = paper_anchor(paper, slugify(category))
    details_id = f"{anchor}-details"

    author = (
        f'<p class="paper__detail">{paper["author"]}</p>'
        if paper.get("author") else ""
    )
    organization = (
        f'<p class="paper__detail">{paper["organization"]}</p>'
        if paper.get("organization") else ""
    )
    comment = (
        f'<p class="paper__detail paper__comment">{paper["comment"]}</p>'
        if paper.get("comment") else ""
    )

    jupyter = ""
    if paper.get("jupyter_notes"):
        jupyter = (
            f'<p><a class="note" href="https://github.com/junkunyuan/junkunyuan.github.io/'
            f'blob/main/paper_reading_list/resource/jupyters/{paper["jupyter_notes"]}">'
            f'(see notes in jupyter)</a></p>'
        )

    details = paper.get("details", "") or ""
    if details:
        details = details.replace(
            "<img src='", f"<img loading='lazy' src='resource/figs/{paper['name']}/{paper['name']}-"
        )
        if "fig:" in details:
            details = convert_fig_cap_to_figure(details, paper["name"])

    summary = paper.get("summary", "") or ""
    summary_block = f'<p>{summary}</p>' if summary else ""

    date_str = _format_date(paper["date"])

    return (
        f'<div id="{anchor}" class="paper paper--toggle" '
        f'data-search="{_search_blob(paper)}" '
        f'style="--paper-accent: var({accent_var});">'
        f'<p class="paper__title" onclick="toggleTable(\'{details_id}\')">{display_title}</p>'
        f'{author}{organization}'
        f'<p class="paper__detail">{venue_label}</p>'
        f'<p class="paper__detail">{date_str} &nbsp;&nbsp;<span class="divider">|</span>&nbsp;&nbsp; {pdf_link}{code_link}</p>'
        f'{comment}'
        f'<div id="{details_id}" class="info_detail{" is-open" if debug_open else ""}">'
        f'<hr class="dashed">{summary_block}{jupyter}<p>{details}</p>'
        f'</div>'
        f'</div>'
    )


def _category_block(papers: List[dict], category: str, domain_title: str,
                     accent_var: str) -> str:
    subset = _filter_and_sort(papers, category, domain_title)
    if not subset:
        return ""
    cat_slug = slugify(category)
    out = [
        f'<h2 id="{cat_slug}-table"><a class="no_dec" href="#cat-{cat_slug}">{category}</a></h2>'
    ]
    out.extend(_paper_card(p, accent_var, category) for p in subset)
    return "".join(out)


def _domain_toc(domain: dict, papers: List[dict]) -> str:
    out = ['<hr><p id="table" class="huger"><b>Table of contents:</b></p>']
    if domain["title"] not in PRESERVE_ORDER_TITLES:
        out.append(
            '<p>Papers are displayed in reverse chronological order. '
            'High-impact or inspiring works are highlighted in '
            '<span class="highlight">red</span>.</p>'
        )
    out.append("<ul>")
    for category in domain["categories"]:
        cat_slug = slugify(category)
        in_cat = _filter_and_sort(papers, category, domain["title"])
        index_html = (
            f'<p class="paper-index">'
            + " &nbsp;&nbsp;&nbsp; ".join(
                build_index_item(paper_anchor(p, cat_slug), p) for p in in_cat
            )
            + "</p>"
        )
        out.append(
            f'<li><a class="larger no_dec" id="cat-{cat_slug}" '
            f'href="#{cat_slug}-table"><b>{category} ({len(in_cat)})</b></a></li>'
            f"{index_html}<br>"
        )
    out.append("</ul>")
    return "".join(out)


def _domain_html(domain: dict, time_now: str) -> str:
    title = domain["title"]
    papers = _drop_drafts(list(domain["papers"]))

    paper_count = "" if title in PRESERVE_ORDER_TITLES else (
        f'<b><span class="highlight">{len(papers)}</span> papers</b>'
    )

    intro = f"""
<h1 id="top">{title}</h1>
<p class="larger"><b>{domain["description"]}</b></p>
<p class="larger">{paper_count}</p>
<p>Written by <a href="../index.html">Junkun Yuan</a>.</p>
<p>Click <a href="paper_reading_list.html">here</a> to go back to main contents.</p>
"""

    footer = (
        f'\n<p class="faint" style="margin-top: 2em; text-align: center;">'
        f'Last updated on {time_now}.</p>\n'
    )

    toc = _domain_toc(domain, papers)

    body_parts = [SEARCH_SCRIPT]
    colors = accent_cycle()
    for category in domain["categories"]:
        body_parts.append(_category_block(papers, category, title, next(colors)))

    head = page_head(
        title=title,
        css_href="../resource/style.css",
        favicon_href="../resource/my_photo.png",
        extra=READING_LIST_HEAD_EXTRA,
    )
    return (
        head + intro + toc + "".join(body_parts) + footer
        + PAPER_TOGGLE_SCRIPT + BACK_TO_TOP + THEME_TOGGLE + PAGE_FOOT
    )


def _main_index_html(domains: List[dict], counts: List[int], time_now: str) -> str:
    parts = [
        f"""
<h1 id="top">{MAIN_CONTENT["title"]}</h1>
<p class="larger"><b>Build AI systems.</b></p>
<p class="larger"><b><span class="highlight">{sum(counts)}</span> papers</b></p>
<p>Written by <a href="../index.html">Junkun Yuan</a>.</p>
""",
        '<hr><p id="table" class="huger"><b>Table of contents:</b></p><ul>',
    ]
    for domain, n in zip(domains, counts):
        n_display = "" if n == 0 else f" ({n} papers)"
        parts.append(
            f'<li class="larger"><a class="no_dec" href="{domain["file"]}">'
            f'<b>{domain["title"]}</b></a>{n_display}</li>'
        )
    parts.append("</ul>")

    head = page_head(
        title=MAIN_CONTENT["title"],
        css_href="../resource/style.css",
        favicon_href="../resource/my_photo.png",
    )
    footer = (
        f'\n<p class="faint" style="margin-top: 2em; text-align: center;">'
        f'Last updated on {time_now}.</p>\n'
    )
    return head + "".join(parts) + footer + BACK_TO_TOP + THEME_TOGGLE + PAGE_FOOT


_PRE_CODE_RE = re.compile(
    r'(<pre)([^>]*)>(\s*)<code\s+class="language-(\w+)"([^>]*)>',
    re.IGNORECASE,
)
_INLINE_FONTSIZE_RE = re.compile(r'\s*style="font-size:\s*14px;?"', re.IGNORECASE)
_ORPHAN_COPY_BTN_RE = re.compile(
    r'<p[^>]*>\s*<button[^>]*onclick="copyCodeBlock\([^)]*\)"[^>]*>[^<]*</button>\s*</p>',
    re.IGNORECASE,
)


def _normalize_code_blocks(html: str) -> str:
    """Promote `language-*` class to <pre>, drop inline font-size, strip
    the orphan custom Copy button (Prism toolbar provides one)."""
    html = _ORPHAN_COPY_BTN_RE.sub("", html)

    def repl(m: re.Match) -> str:
        pre_open, pre_attrs, ws, lang, code_extra = m.groups()
        code_extra = _INLINE_FONTSIZE_RE.sub("", code_extra)
        cls = f"language-{lang}"
        if 'class="' in pre_attrs:
            pre_attrs = re.sub(
                r'class="([^"]*)"',
                lambda mm: f'class="{mm.group(1).strip()} {cls}"',
                pre_attrs,
                count=1,
            )
        else:
            pre_attrs = f' class="{cls}"' + pre_attrs
        return f'{pre_open}{pre_attrs}>{ws}<code class="{cls}"{code_extra}>'

    return _PRE_CODE_RE.sub(repl, html)


def _write(filename: str, content: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    content = _normalize_code_blocks(content)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {path}")


def main() -> None:
    time_now = format_now()
    domains = [_load_domain(n) for n in DOMAIN_FILES]

    counts: List[int] = []
    for domain in domains:
        published = _drop_drafts(list(domain["papers"]))
        counts.append(0 if domain["title"] in PRESERVE_ORDER_TITLES else len(published))
        _write(domain["file"], _domain_html(domain, time_now))

    _write(MAIN_CONTENT["file"], _main_index_html(domains, counts, time_now))


if __name__ == "__main__":
    main()
