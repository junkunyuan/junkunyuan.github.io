"""Generate the full publications page (`publications.html`). The home page
lists only papers tagged `selected: true`; this page lists every paper from
`data/publications.yaml`."""

from __future__ import annotations

import os

from build.index import _load_papers, _publications_section
from build.lib.templates import (
    BACK_TO_TOP,
    ET_AL_SCRIPT,
    PAGE_FOOT,
    THEME_TOGGLE,
    page_head,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BACK_LINK = """
<p class="larger"><a href="index.html">&larr; Back to home</a></p>
<h1>Junkun Yuan &nbsp; 袁俊坤</h1>
"""


def main() -> None:
    papers = _load_papers()

    head = page_head(
        title="Junkun Yuan — Publications",
        css_href="resource/style.css",
        favicon_href="resource/my_photo.png",
    )

    html = (
        head
        + BACK_LINK
        + _publications_section(papers)
        + ET_AL_SCRIPT
        + BACK_TO_TOP
        + THEME_TOGGLE
        + PAGE_FOOT
    )

    out_path = os.path.join(PROJECT_ROOT, "publications.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {out_path} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
