"""
Paper Reading List HTML Generator

This module generates HTML files for the paper reading list website.
It processes domain-specific paper data and creates formatted HTML pages
with proper styling, navigation, and interactive features.
"""

from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

from resource.utils import (
    get_venue_all,
    border_color_generator,
    convert_fig_cap_to_figure,
    TOP_BUTTON,
)
from resource.main_content import MAIN_CONTENT
from resource.research import LIST as RESEARCH
from resource.coding import LIST as CODING
from resource.fundamental import LIST as FUNDAMENTAL_COMPONENT
from resource.visual_understanding import LIST as VISUAL_UNDERSTANDING
from resource.language_generation import LIST as LANGUAGE_GENERATION
from resource.reinforcement_learning import LIST as REINFORCEMENT_LEARNING
from resource.visual_generation import LIST as VISUAL_GENERATION
from resource.multimodal_understanding import LIST as MULTIMODAL_UNDERSTANDING
from resource.native_multimodal_generation import LIST as NATIVE_MULTIMODAL_GENERATION

# Configuration constants
DOMAINS: List[Dict[str, Any]] = [
    RESEARCH,
    CODING,
    FUNDAMENTAL_COMPONENT,
    # VISUAL_UNDERSTANDING,
    LANGUAGE_GENERATION,
    REINFORCEMENT_LEARNING,
    VISUAL_GENERATION,
    MULTIMODAL_UNDERSTANDING,
    NATIVE_MULTIMODAL_GENERATION
]

EXCLUDE_TITLE: List[str] = ["Research", "Coding and Engineering"]

# Generate current timestamp
time_now: str = datetime.now().strftime('%B %d, %Y at %H:%M')

# HTML template constants
PREFIX: str = f"""<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="resource/html.css" type="text/css">
    <link rel="shortcut icon" href="resource/my_photo.jpg">
    <title>Paper Reading List</title>
    <meta name="description" content="Paper Reading List">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <div id="layout-content" style="margin-top:25px">
</head>
<body>
"""
INTRODUCTION_TEMPLATE: str = f"""
<h1 id="top">title</h1>
<p class="larger"><b>description</b></p>
<p class="larger"><b>total_paper_num</b></p>
<p>Written by <a href="https://junkunyuan.github.io/">Junkun Yuan</a>.</p>
<p>Click <a href="paper_reading_list.html">here</a> to go back to main contents.</p>
<p><font color=#B0B0B0>Last updated on time_now (UTC+8).</font></p>
"""

SUFFIX: str = """</body>
</html>"""


def build_paper_index(papers: List[Dict[str, Any]], category: str) -> str:
    """
    Build a compact, beautiful index of all papers with anchor links.
    Each entry: [name] (venue_abbr, year)
    """
    # Assign anchor ids for each paper
    index_items = []
    for idx, paper in enumerate(papers):
        anchor_id = f"{paper['name']}{category.lower()}"
        # venue_abbr and year
        try:
            venue_abbr, venue_year = paper["venue"].rsplit(" ", 1)
        except Exception:
            venue_abbr, venue_year = paper["venue"], ""
        name = paper["name"]
        color = "#D04040" if "**" in paper.get("info", "") else "#505050"
        # Use a short, tight style
        index_items.append(
            f'<a href="#{anchor_id}" class="no_dec"><font color={color}><b>{name}</b> <font style="color:#888;font-size:12px;">({venue_abbr} {venue_year})</font></font></a>'
        )
    # Arrange in a compact multi-row flexbox
    html = """
    <div style="margin: 0.5em 0 1.2em 0;">
      <div style="display: flex; flex-wrap: wrap; gap: 0.7em 1.5em; align-items: center; font-size: 14px;">
        {}
      </div>
    </div>
    """.format("\n        ".join(index_items))
    return html


def _build_paper_html(paper: pd.Series, category: str, color_bar: str, domain_title: str) -> str:
    """Build HTML content for a single paper."""
    color = "#D04040" if "**" in paper["info"] else "#404040"

    # Build links
    items = []
    if paper["pdf_url"].strip():
        items.append(f"""<a href="{paper["pdf_url"]}">paper</a>""")
    if paper["code_url"].strip():
        items.append(f"""<a href="{paper['code_url']}">code</a>""")
    links_html = " &nbsp;&nbsp;<font color=#BBBBBB>|</font>&nbsp;&nbsp; ".join(items)

    # Format date and venue
    venue = paper["venue"]
    venue_all = get_venue_all(paper["venue"])
    date = ""
    if domain_title not in EXCLUDE_TITLE:
        try:
            date = datetime.strptime(paper["date"], "%Y%m%d").strftime("%b %d, %Y")
        except ValueError:
            date = ""

    # Build optional elements
    comment = f"""<p class="paper_detail"><font color=#D04040>{paper["comment"]}</font></p>""" if paper["comment"] else ""

    jupyter_note = ""
    if paper.get("jupyter_notes", ""):
        jupyter_note = f"""
        <p><a href="https://github.com/junkunyuan/junkunyuan.github.io/blob/main/paper_reading_list/resource/jupyters/{paper["jupyter_notes"]}" class="note">(see notes in jupyter)</a></p>
        """

    # Process details
    details = paper["details"].replace("<img src='", f"<img src='resource/figs/{paper["name"]}/{paper["name"]}-")
    if "fig:" in details:
        details = convert_fig_cap_to_figure(details, paper["name"])

    # Build optional author and organization
    author = f"""<p class="paper_detail">{paper["author"]}</p>""" if paper["author"] else ""
    organization = f"""<p class="paper_detail">{paper["organization"]}</p>""" if paper["organization"] else ""

    # Debug content for new papers
    debug = ""
    if "new paper" in paper["title"].lower():
        debug = f"""<p>{paper["summary"]}</p><p>{details}</p>"""
        debug = debug.replace("data-src", "src")

    # Build final HTML
    paper_html = f"""
    <p class="little_split" id='{paper["name"] + category.lower()}'></p>
    <div style="border-left: 16px solid {color_bar}; padding-left: 10px">
    <div style="height: 0.3em;"></div>
    <p class="paper_title" onclick="toggleTable('{paper["name"]}-{category}-details')"><i>{paper["title"]}</i></p>
    {author}
    {organization}
    <p class="paper_detail">{venue} &nbsp; <font color=#D0D0D0>{venue_all}</font></p>
    <p class="paper_detail"><b>{date}</b> &nbsp;&nbsp;<font color=#BBBBBB>|</font>&nbsp;&nbsp; <b><font color={color}>{paper["name"]}</font></b></p>
    <p class="paper_detail">{links_html}</p>
    {comment}
    {debug}
    <div id='{paper["name"]}-{category}-details' class="info_detail">
        <hr class="dashed">
        <p><font color=#202020>{paper["summary"]}</font></p>
        {jupyter_note}
        <p>{details}</p>
    </div>
    <div style="height: 0.05em;"></div>
    </div>
    <p class="little_split"></p>
    """

    return paper_html


def _build_domain_content(domain: Dict[str, Any], papers: pd.DataFrame) -> str:
    """Build the content for all categories in a domain."""
    content_all_domain = ""
    color_bar_generator = border_color_generator()

    for category in domain["categories"]:
        color_bar = next(color_bar_generator)

        # Filter papers for this category
        category_mask = papers["category"].str.contains(category)
        if category_mask.sum() == 0:
            continue

        paper_subset = papers[category_mask]
        if domain["title"] not in EXCLUDE_TITLE:
            paper_subset = paper_subset.sort_values(by="date", ascending=False)

        # Build category header
        content_cate = f"""<h2 id="{category}-table"><a class="no_dec" href="#{category}">{category}</a></h2>"""

        # Build HTML for each paper
        for _, paper in paper_subset.iterrows():
            content_cate += _build_paper_html(paper, category, color_bar, domain["title"])

        content_all_domain += content_cate

    return content_all_domain


def _build_table_of_contents(domain: Dict[str, Any], papers: pd.DataFrame) -> str:
    """Build the table of contents for a domain."""
    catalog = """<hr><p id='table' class="huger"><b>Table of contents:</b></p>"""

    if domain["title"] not in EXCLUDE_TITLE:
        catalog += """<p>Papers are displayed in reverse chronological order. Some highly-impact or inspiring works are highlighted in <font color="#D04040">red</font>.</p><ul>"""
    else:
        catalog += "<ul>"

    for category in domain["categories"]:
        paper_subset = papers[papers["category"].str.contains(category)]
        if domain["title"] not in EXCLUDE_TITLE:
            paper_subset = paper_subset.sort_values(by="date", ascending=False)

        # Convert DataFrame to list of dictionaries for build_paper_index
        papers_list = paper_subset.to_dict('records')
        paper_links_table = build_paper_index(papers_list, category)
        catalog += f"""<li><a class="larger low_margin" id="{category}" href="#{category}-table"><b>{category}</b></a></li>{paper_links_table}<br>"""

    catalog += "</ul>"
    return catalog


def build_main_content_all_domains(domains: List[Dict[str, Any]], num_papers: List[int]) -> str:
    """
    Build the main table of contents for all domains.

    Args:
        domains: List of domain dictionaries containing title and file info
        num_papers: List of paper counts for each domain

    Returns:
        HTML string containing the table of contents
    """
    content = """<hr><p id="table" class="huger"><b>Table of contents:</b></p><ul>"""
    for domain, num_paper in zip(domains, num_papers):
        paper_num_display = "" if num_paper == 0 else f" ({num_paper} papers)"
        content += f"""<li class="larger"><a class="no_dec" href={domain["file"]}><b>{domain["title"]}</b></a>{paper_num_display}</li>"""
    content += "</ul>"
    return content


def _load_papers_data(domain: Dict[str, Any]) -> pd.DataFrame:
    """
    Load paper data from domain dictionary into a pandas DataFrame.

    Args:
        domain: Domain dictionary containing papers list

    Returns:
        DataFrame with paper data
    """
    paper_fields = [
        "title", "author", "organization", "date", "venue", "pdf_url",
        "code_url", "name", "comment", "category", "jupyter_notes",
        "info", "summary", "details"
    ]

    papers_data = {field: [] for field in paper_fields}

    for paper in tqdm(domain["papers"], desc=f"Reading {domain['title']} papers"):
        for field in paper_fields:
            papers_data[field].append(paper.get(field, ""))

    return pd.DataFrame(papers_data)


def build_main_content_of_each_domain(domain: Dict[str, Any]) -> str:
    """
    Build the main content for a specific domain including table of contents and paper details.

    Args:
        domain: Domain dictionary containing papers, categories, title, etc.

    Returns:
        HTML string containing the domain content
    """
    # Load papers data
    papers = _load_papers_data(domain)

    # Build table of contents
    catalog = _build_table_of_contents(domain, papers)
    
    # Build contents for each category
    content_all_domain = _build_domain_content(domain, papers)
    # Add JavaScript and navigation
    toggle_script = """
    <script>
        function toggleTable(tableId) {
            const container = document.getElementById(tableId);
            const isVisible = window.getComputedStyle(container).display !== 'none';
            if (!isVisible) {
                const images = container.querySelectorAll('img');
                images.forEach(img => {
                    if (img.dataset.src !== '') {img.src = img.dataset.src;}
                });
                container.style.display = 'block';
            } else {container.style.display = 'none';}
        }
    </script>
    """

    content_all_domain += toggle_script
    content_all_domain += TOP_BUTTON

    return catalog + content_all_domain

def _build_enhanced_prefix() -> str:
    """Build enhanced HTML prefix with syntax highlighting and math support."""
    return PREFIX.replace("<body>",
        """<script src="https://cdn.jsdelivr.net/npm/prismjs@1.24.1/prism.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/prismjs@1.24.1/themes/prism.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/prismjs@1.24.1/components/prism-bash.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/prismjs@1.24.1/components/prism-python.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <body>""")


def _write_html_file(filepath: str, content: str) -> None:
    """Write HTML content to file with proper encoding."""
    try:
        with open(filepath, "w", encoding="utf-8-sig") as f:
            f.write(content)
        print(f"Successfully generated: {filepath}")
    except IOError as e:
        print(f"Error writing file {filepath}: {e}")
        raise


def main() -> None:
    """Main function to build all HTML files."""
    print("Starting HTML generation process...")

    # Prepare introduction template
    intro_temp = INTRODUCTION_TEMPLATE.replace("time_now", time_now)

    # Build HTML for all domains
    num_papers: List[int] = []
    enhanced_prefix = _build_enhanced_prefix()

    for domain in DOMAINS:
        try:
            num_paper = 0 if domain["title"] in EXCLUDE_TITLE else len(domain["papers"])
            num_papers.append(num_paper)

            paper_num_display = "" if domain["title"] in EXCLUDE_TITLE else f"<font color='#D93053'>{num_paper}</font> papers"

            intro = intro_temp.replace("total_paper_num", paper_num_display)
            intro = intro.replace("title", domain["title"])
            intro = intro.replace("description", domain["description"])

            papers_content = build_main_content_of_each_domain(domain)
            content_domain = enhanced_prefix + intro + papers_content + SUFFIX

            _write_html_file(domain["file"], content_domain)

        except Exception as e:
            print(f"Error processing domain '{domain['title']}': {e}")
            raise

    # Build main content page
    try:
        intro = intro_temp.replace("total_paper_num", f"<font color='#D93053'>{sum(num_papers)}</font> papers")
        intro = intro.replace("title", MAIN_CONTENT["title"])
        intro = intro.replace("description", "Build AI systems.")
        intro = intro.replace("""<p>Click <a href="paper_reading_list.html">here</a> to go back to main contents.</p>""", "")

        content = build_main_content_all_domains(DOMAINS, num_papers)
        main_content = PREFIX + intro + content + SUFFIX

        _write_html_file(MAIN_CONTENT["file"], main_content)

    except Exception as e:
        print(f"Error building main content: {e}")
        raise

    print("HTML generation completed successfully!")


if __name__ == "__main__":
    main()
    