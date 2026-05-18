"""Generate the personal home page (`index.html`): biography, publications,
professional service. All paper data comes from `data/publications.yaml`."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import yaml

from build.lib.templates import (
    BACK_TO_TOP,
    ET_AL_SCRIPT,
    PAGE_FOOT,
    THEME_TOGGLE,
    format_now,
    page_head,
)
from build.lib.utils import (
    accent_cycle,
    build_index_item,
    format_authors,
    get_venue_full,
    paper_anchor,
    split_venue,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB_PATH = os.path.join(PROJECT_ROOT, "data", "publications.yaml")
SCOPE = "publications"


def _load_papers() -> List[Dict[str, Any]]:
    with open(PUB_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["papers"]


def _footer_updated(time_now: str) -> str:
    return (
        f'\n<p class="profile__updated" style="margin-top: 2em; text-align: center;">'
        f'Last updated on {time_now} &nbsp;'
        f'<a href="paper_reading_list/paper_reading_list.html" title="Paper reading list" '
        f'style="text-decoration: none;">&#128214;</a></p>\n'
    )


def _profile_block(time_now: str) -> str:
    return f"""
<section class="profile">
    <div class="profile__text">
        <h1>Junkun Yuan &nbsp; 袁俊坤</h1>
        <p>Research Scientist, ByteDance (US)</p>
        <p>Work and live in San Jose (US) &nbsp;&amp;&nbsp; Shenzhen (China)</p>
        <p>yuanjk0921@outlook.com</p>
    </div>
    <img class="profile__photo" src="resource/my_photo.png" alt="Junkun Yuan">
</section>

<p>
<a href="#biography">Biography</a> &nbsp;&nbsp;
<a href="#publications">Selected Publications</a> &nbsp;&nbsp;
<a href="#professional-service">Professional Service</a>
</p>
"""


BIOGRAPHY = """
<h2 id="biography">Biography</h2>
<p>
    I am a Research Scientist at ByteDance (US), working on visual generative foundation models, such as Seedance 2.0, and their applications and products.<br><br>

    During 2023–2025, I worked as a Research Scientist in the Hunyuan Multimodal Generation Team at Tencent with <a href="https://scholar.google.com/citations?user=AjxoEpIAAAAJ">Wei Liu</a>, <a href="https://scholar.google.com.hk/citations?user=igtXP_kAAAAJ&hl=zh-CN&oi=ao">Zhao Zhong</a>, and <a href="https://scholar.google.com.hk/citations?user=FJwtMf0AAAAJ&hl=zh-CN&oi=ao">Liefeng Bo</a>, where my research focused on multimodal generative foundation models and downstream generation tasks.

    During 2022–2023, I was a research intern in the Computer Vision Group at Baidu with <a href="https://scholar.google.com/citations?user=PSzJxD8AAAAJ">Xinyu Zhang</a> and <a href="https://scholar.google.com/citations?user=z5SPCmgAAAAJ">Jingdong Wang</a>, where my research focused on visual self-supervised pre-training.<br><br>

    I received my Ph.D. degree in Computer Science from Zhejiang University (2019–2024), co-supervised by Professors <a href="https://scholar.google.com/citations?user=FOsNiMQAAAAJ">Kun Kuang</a>, <a href="https://person.zju.edu.cn/0096005">Lanfen Lin</a>, and <a href="https://scholar.google.com/citations?user=XJLn4MYAAAAJ">Fei Wu</a>. I received my B.E. degree in Automation from Zhejiang University of Technology (2015–2019), supervised by Professor <a href="https://scholar.google.com.hk/citations?user=smi7bpoAAAAJ&hl=zh-CN&oi=ao">Qi Xuan</a>.<br><br>

    I have been fortunate to work closely with friends including <a href="https://scholar.google.com/citations?user=F5P_8NkAAAAJ&hl=zh-CN&oi=ao">Defang Chen</a> and <a href="https://scholar.google.com/citations?user=kwBR1ygAAAAJ&hl=zh-CN&oi=ao">Yue Ma</a>; their insights have profoundly shaped my approach to research.
</p>
"""


SERVICE = """
<h2 id="professional-service">Professional Service</h2>
<ul>
<li><b>Conference Reviewer:</b>&nbsp;&nbsp;
    ICLR 2026 &nbsp;<span class="divider">|</span>&nbsp;
    ICML 2026 &nbsp;<span class="divider">|</span>&nbsp;
    ICCV 2023 &nbsp;<span class="divider">|</span>&nbsp;
    AAAI 2023, 2026 &nbsp;<span class="divider">|</span>&nbsp;
    MM 2023</li>
<li><b>Journal Reviewer:</b>&nbsp;&nbsp;
    TPAMI 2023 &nbsp;<span class="divider">|</span>&nbsp;
    TNNLS 2022 &nbsp;<span class="divider">|</span>&nbsp;
    TCSVT 2022, 2025 &nbsp;<span class="divider">|</span>&nbsp;
    PR 2025 &nbsp;<span class="divider">|</span>&nbsp;
    NN 2023</li>
</ul>
"""


def _index_by_year(papers: List[dict]) -> str:
    """Compact, year-grouped paper index with colored year labels."""
    by_year: Dict[str, List[dict]] = defaultdict(list)
    for p in papers:
        by_year[p["date"][:4]].append(p)

    colors = accent_cycle()
    year_color = {y: next(colors) for y in sorted(by_year, reverse=True)}

    sections: List[str] = []
    for year in sorted(by_year, reverse=True):
        items = " ".join(
            build_index_item(paper_anchor(p, SCOPE), p) for p in by_year[year]
        )
        sections.append(
            f'<p class="paper-index" style="--paper-accent: var({year_color[year]});">'
            f'<span class="paper-index__year">{year}:</span> {items}</p>'
        )
    return "\n".join(sections)


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
        return date_str


def _paper_card(paper: dict, accent_var: str, *, show_year_badge: str = "") -> str:
    venue_full = get_venue_full(paper["venue"])
    abbr_html, year_html = _format_venue(paper["venue"])
    if venue_full:
        venue_label = f'{venue_full} ({abbr_html})'
        if year_html:
            venue_label += f', {year_html}'
    else:
        venue_label = abbr_html + (f', {year_html}' if year_html else "")

    name_cls = "paper__name paper__name--highlight" if paper.get("highlight") else "paper__name"
    name_html = f'<span class="{name_cls}">{paper["name"]}</span>'
    pdf_link = f'<a href="{paper["pdf_url"]}">{name_html}</a>'

    code_link = ""
    if paper.get("code_url"):
        code_link = (
            f' &nbsp;&nbsp;<span class="divider">|</span>&nbsp;&nbsp; '
            f'<a href="{paper["code_url"]}">code</a>'
        )

    comment_html = (
        f'<p class="paper__detail paper__comment">{paper["comment"]}</p>'
        if paper.get("comment")
        else ""
    )

    year_badge = (
        f'<div class="paper__year-badge">{show_year_badge}</div>'
        if show_year_badge
        else ""
    )

    anchor = paper_anchor(paper, SCOPE)
    authors = format_authors(paper["author"], anchor)
    date_str = _format_date(paper["date"])

    return (
        f'<div id="{anchor}" class="paper" style="--paper-accent: var({accent_var});">'
        f'{year_badge}'
        f'<p class="paper__title">{paper["title"]}</p>'
        f'<p class="paper__detail">{authors}</p>'
        f'<p class="paper__detail">{venue_label}</p>'
        f'<p class="paper__detail">{date_str} &nbsp;&nbsp;<span class="divider">|</span>&nbsp;&nbsp; {pdf_link}{code_link}</p>'
        f'{comment_html}'
        f'</div>'
    )


def _publications_section(
    papers: List[dict],
    *,
    title: str = "Publications",
    full_link_url: str = "",
    show_year_index: bool = True,
    show_year_badges: bool = True,
) -> str:
    full_link_html = (
        f'<a href="{full_link_url}"><b>Full Publication List &rarr;</b></a> &nbsp;&nbsp; '
        if full_link_url
        else ""
    )
    header = f"""
<h2 id="publications">{title}</h2>
<p>
    {full_link_html}
    <a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a>
    &nbsp;&nbsp;
    <a href="https://www.semanticscholar.org/author/Junkun-Yuan/2304610230">Semantic Scholar Profile</a>
</p>
<p>(co-)first author<sup class="author-mark">&#10035;</sup> &nbsp;&nbsp; corresponding author<sup class="author-mark author-mark--mail">&#9993;</sup></p>
"""

    index_html = _index_by_year(papers) if show_year_index else ""

    colors = accent_cycle()
    uniform_accent = next(accent_cycle()) if not show_year_badges else ""
    year_color: Dict[str, str] = {}
    cards: List[str] = []
    current_year = ""
    for paper in papers:
        year = paper["date"][:4]
        first_of_year = year != current_year
        if first_of_year:
            year_color[year] = next(colors)
            current_year = year
        cards.append(
            _paper_card(
                paper,
                accent_var=uniform_accent or year_color[year],
                show_year_badge=year if (show_year_badges and first_of_year) else "",
            )
        )

    return header + index_html + "".join(cards)


def main() -> None:
    papers = _load_papers()
    selected = [p for p in papers if p.get("selected")]
    time_now = format_now()

    head = page_head(
        title="Junkun Yuan",
        css_href="resource/style.css",
        favicon_href="resource/my_photo.png",
    )

    html = (
        head
        + _profile_block(time_now)
        + BIOGRAPHY
        + _publications_section(
            selected,
            title="Selected Publications",
            full_link_url="publications.html",
            show_year_index=False,
            show_year_badges=False,
        )
        + SERVICE
        + _footer_updated(time_now)
        + ET_AL_SCRIPT
        + BACK_TO_TOP
        + THEME_TOGGLE
        + PAGE_FOOT
    )

    out_path = os.path.join(PROJECT_ROOT, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {out_path} ({len(selected)} selected / {len(papers)} total)")


if __name__ == "__main__":
    main()
