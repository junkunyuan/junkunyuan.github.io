"""Shared helpers for venue lookup, slugification, author formatting, and
figure markup conversion. All HTML helpers emit class-based markup; no
inline `<font>` tags. Styling lives in `resource/style.css`."""

from __future__ import annotations

import os
import re
from typing import Dict, Generator

VENUE_NAME_DICT: Dict[str, str] = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "CVPR": "Conference on Computer Vision and Pattern Recognition",
    "ECAI": "European Conference on Artificial Intelligence",
    "ECCV": "European Conference on Computer Vision",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "ICCV": "International Conference on Computer Vision",
    "ICASSP": "International Conference on Acoustics, Speech and Signal Processing",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "IJCV": "International Journal of Computer Vision",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "MM": "International Conference on Multimedia",
    "NeurIPS": "Advances in Neural Information Processing Systems",
    "SIGGRAPH-Asia": "ACM SIGGRAPH Annual Conference in Asia",
    "TKDD": "ACM Transactions on Knowledge Discovery from Data",
    "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
    "TMLR": "Transactions on Machine Learning Research",
    "TMM": "IEEE Transactions on Multimedia",
    "WACV": "Winter Conference on Applications of Computer Vision",
}

_ACCENT_PALETTE = ("--accent-1", "--accent-2", "--accent-3", "--accent-4")


def get_venue_full(venue: str) -> str:
    """Return the full venue name for "<abbr> <year>", or empty if unknown."""
    abbr = venue.split(" ", 1)[0]
    return VENUE_NAME_DICT.get(abbr, "")


def split_venue(venue: str) -> tuple[str, str]:
    """Split "ICLR 2026" into ("ICLR", "2026"); ("arXiv", "") if no year."""
    parts = venue.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return venue, ""


def slugify(text: str) -> str:
    """Lowercase, hyphenate spaces, drop everything but alnum/hyphen."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "paper"


def paper_anchor(paper: dict, scope: str) -> str:
    """Stable anchor id. Combines slugified name + venue + scope so two
    papers sharing a name (e.g. two "MPL" entries) still get unique anchors."""
    return f"{slugify(paper['name'])}-{slugify(paper['venue'])}-{scope}"


def accent_cycle() -> Generator[str, None, None]:
    """Yield CSS variable names for the year/category accent palette, cyclic."""
    i = 0
    while True:
        yield _ACCENT_PALETTE[i % len(_ACCENT_PALETTE)]
        i += 1


# ----------------------------------------------------------------------------
# Author formatting
# ----------------------------------------------------------------------------

_CO_FIRST = "<sup class=\"author-mark\">&#10035;</sup>"     # ✱
_CORRESP  = "<sup class=\"author-mark author-mark--mail\">&#9993;</sup>"      # ✉
_JUNKUN   = "Junkun Yuan"


def _format_one_author(name: str) -> str:
    """Apply Junkun highlight and ** / ## marker conversion to a single author."""
    out = name.replace(_JUNKUN, f'<span class="author--junkun">{_JUNKUN}</span>')
    out = out.replace("**", _CO_FIRST)
    out = out.replace("##", _CORRESP)
    return out


def format_authors(author_string: str, paper_uid: str, max_visible: int = 4) -> str:
    """Format an author list. When the list has more than `max_visible` names
    AND Junkun is past position `max_visible-1`, show authors up through Junkun;
    otherwise show the first `max_visible`. The remainder collapses behind a
    clickable "et al." toggle. `paper_uid` must be unique across the page."""
    if not author_string:
        return ""

    authors = [a.strip() for a in author_string.split(",")]
    if len(authors) <= max_visible:
        return ", ".join(_format_one_author(a) for a in authors)

    junkun_idx = next(
        (i for i, a in enumerate(authors) if _JUNKUN in a),
        -1,
    )
    cut = max(max_visible, junkun_idx + 1) if junkun_idx >= max_visible else max_visible

    visible = ", ".join(_format_one_author(a) for a in authors[:cut])
    hidden = ", ".join(_format_one_author(a) for a in authors[cut:])

    return (
        f'{visible}, '
        f'<span class="et-al" data-uid="{paper_uid}" '
        f'role="button" tabindex="0">et al.</span>'
        f'<span class="et-al-hidden" data-uid="{paper_uid}" hidden>{hidden}</span>'
    )


def junkun_author_mark(author: str) -> str:
    """Return superscript marker(s) to display next to "Junkun" in the
    compact index (✱ for co-first, ✉ for corresponding)."""
    if _JUNKUN not in author:
        return ""
    marks = ""
    # Find the marker characters attached to Junkun's name.
    m = re.search(rf"{re.escape(_JUNKUN)}([*#]+)", author)
    if m:
        token = m.group(1)
        if "*" in token:
            marks += _CO_FIRST
        if "#" in token:
            marks += _CORRESP
    elif author.strip().startswith(_JUNKUN):
        marks += _CO_FIRST
    return marks


# ----------------------------------------------------------------------------
# Compact index entry: "[Name] (Venue, Year)"
# ----------------------------------------------------------------------------

def build_index_item(anchor_id: str, paper: dict) -> str:
    abbr, year = split_venue(paper["venue"])
    year_part = f" {year}" if year else ""
    cls = "is-highlight" if paper.get("highlight") else ""
    return (
        f'<a href="#{anchor_id}" class="no_dec {cls}">'
        f'{paper["name"]}{junkun_author_mark(paper.get("author", ""))}'
        f'<span class="venue">({abbr}{year_part})</span>'
        f"</a>"
    )


# ----------------------------------------------------------------------------
# Figure markup ("fig: file.png 300\ncap: caption") → <figure>
# ----------------------------------------------------------------------------

def convert_fig_cap_to_figure(text: str, name: str) -> str:
    """Convert custom fig:/cap: markup into <figure>. Code blocks (<pre>) are
    left untouched. Image src uses native lazy loading."""
    parts = text.split("<pre>", 1)
    head, tail = parts[0], ("<pre>" + parts[1] if len(parts) == 2 else "")

    out_lines: list[str] = []
    lines = head.strip().splitlines()
    fig_n = 0
    i = 0
    prefix = os.path.join("resource", "figs", name)

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("fig:"):
            pairs = re.findall(r"(\S+)\s+(\d+)", line[len("fig:"):])
            caption = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("cap:"):
                caption = lines[i + 1].strip()[len("cap:"):].strip()
                i += 1
            fig_n += 1
            out_lines.append("<figure>")
            for fname, width in pairs:
                src = os.path.join(prefix, f"{name}-{fname}")
                out_lines.append(
                    f'<img src="{src}" width="{width}" loading="lazy" alt="">'
                )
            out_lines.append(f"<figcaption><b>Figure {fig_n}.</b> {caption}</figcaption>")
            out_lines.append("</figure>")
        else:
            out_lines.append(line)
        i += 1

    return "\n".join(out_lines) + tail
