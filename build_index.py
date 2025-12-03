"""
Main index page generator for the personal website.

This module generates the main index.html page with biography,
publications, and professional service information.
"""

from datetime import datetime
from typing import List, Dict, Any

from resource.pub_list import PAPERS
from paper_reading_list.resource.utils import get_venue_all, border_color_generator, TOP_BUTTON

# Generate current timestamp
TIME_NOW: str = datetime.now().strftime('%B %d, %Y at %H:%M')

# HTML template constants
PREFIX: str = f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="resource/index.css" type="text/css">
    <link rel="shortcut icon" href="resource/my_photo.jpg">
    <title>Junkun Yuan</title>
    <meta name="description" content="Junkun Yuan">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script>
        function toggleAuthors(paperId) {{
            var etAlElement = document.getElementById('et_al_' + paperId);
            var hiddenElement = document.getElementById('hidden_authors_' + paperId);
            
            if (hiddenElement.style.display === 'none') {{
                // Show hidden authors and hide et al.
                hiddenElement.style.display = 'inline';
                etAlElement.style.display = 'none';
            }} else {{
                // Hide hidden authors and show et al.
                hiddenElement.style.display = 'none';
                etAlElement.style.display = 'inline';
            }}
        }}
    </script>
    <div id="layout-content" style="margin-top:25px">
    <table>
        <tbody>
            <tr>
                <td width="870">
                    <h1>Junkun Yuan &nbsp; 袁俊坤</h1>
                    <p>Research Scientist</p>
                    <p>yuanjk0921@outlook.com</p>
                    <p>work and live in Shenzhen, China</p>
                    <p><font color=#D0D0D0>Last updated on {TIME_NOW} (UTC+8)</font></p>
                </td>
                <td style="padding-right: 100px; padding-top: 20px;">
                    <img src="resource/my_photo.jpg" width="160" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
                </td>
            </tr>
        </tbody>
    </table>
</head>
<body>
"""
TABLE_OF_CONTENTS = \
"""
<p class="larger">
<a href="#biography" color: #404040; font-weight: 500;">Biography</a> &nbsp;&nbsp; <a href="#publications" color: #404040; font-weight: 500;">Publications</a> &nbsp;&nbsp; <a href="#professional-service" color: #404040; font-weight: 500;">Professional Service</a>
</p>
"""

BIOGRAPHY = \
"""
<h2 id="biography">Biography</h2>
<p>
    During Sep 2023 — Nov 2025, I interned and then worked in Multimodal Generation Foundation Model Team at <b>Tencent Hunyuan (Shenzhen, China)</b>, focusing on multimodal generative foundation models and applications, with <a href="https://scholar.google.com/citations?user=AjxoEpIAAAAJ">Wei Liu</a>, <a href="https://scholar.google.com.hk/citations?user=FJwtMf0AAAAJ&hl=zh-CN&oi=ao">Liefeng Bo</a>, and <a href="https://scholar.google.com.hk/citations?user=igtXP_kAAAAJ&hl=zh-CN&oi=ao">Zhao Zhong</a>.
    
    During Jul 2022 — Aug 2023, I interned in the Computer Vision Group at <b>Baidu VIS (Beijing, China)</b>, focusing on visual self-supervised pre-training, with <a href="https://scholar.google.com/citations?user=PSzJxD8AAAAJ">Xinyu Zhang</a> and <a href="https://scholar.google.com/citations?user=z5SPCmgAAAAJ">Jingdong Wang</a>.<br><br>

    I received my Ph.D. degree in Computer Science from Zhejiang University (2019 — 2024), co-supervised by professors of <a href="https://scholar.google.com/citations?user=FOsNiMQAAAAJ">Kun Kuang</a>, <a href="https://person.zju.edu.cn/0096005">Lanfen Lin</a>, and <a href="https://scholar.google.com/citations?user=XJLn4MYAAAAJ">Fei Wu</a>. I received my B.E. degree in Automation from Zhejiang University of Technology (2015 — 2019), co-supervised by professors of <a href="https://scholar.google.com.hk/citations?user=smi7bpoAAAAJ&hl=zh-CN&oi=ao">Qi Xuan</a> and <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=CnBn6FwAAAAJ">Li Yu</a>.<br><br>

    I have been fortunate to work closely with some friends such as <a href="https://scholar.google.com/citations?user=F5P_8NkAAAAJ&hl=zh-CN&oi=ao">Defang Chen</a> and <a href="https://scholar.google.com/citations?user=kwBR1ygAAAAJ&hl=zh-CN&oi=ao">Yue Ma</a>, their insights also profoundly shape my approach to research.
</p>
"""
SERVICE = \
"""
<h2 id="professional-service">Professional Service</h2>
<ul>
<li> <b>Conference Reviewer:</b>&nbsp;&nbsp; 
ICLR 2026 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
ICCV 2023 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
AAAI 2023, 2026 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
MM 2023

<li> <b>Journal Reviewer:</b>&nbsp;&nbsp;
TPAMI 2023 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
TNNLS 2022 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
TCSVT 2022, 2025 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
PR 2025 &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp;
NN 2023
</ul>
"""
SUFFIX = \
"""
</body>
</html>
"""

def build_paper_index(papers: List[Dict[str, Any]], category: str) -> str:
    """
    Build a compact, beautiful index of all papers with anchor links.
    Each entry: [name] (venue_abbr, year)
    Papers are grouped by year with colored year labels.
    """
    # Group papers by year
    papers_by_year = {}
    for paper in papers:
        paper_year = paper["date"][:4]
        if paper_year not in papers_by_year:
            papers_by_year[paper_year] = []
        papers_by_year[paper_year].append(paper)
    
    # Sort years in descending order
    sorted_years = sorted(papers_by_year.keys(), reverse=True)
    
    # Generate color generator
    color_gen = border_color_generator()
    year_colors = {}
    
    # Assign colors to years
    for year in sorted_years:
        year_colors[year] = next(color_gen)
    
    # Build HTML for each year
    year_sections = []
    for year in sorted_years:
        year_color = year_colors[year]
        papers_in_year = papers_by_year[year]
        
        # Build paper items for this year
        index_items = []
        for paper in papers_in_year:
            anchor_id = f"{paper['name']}{category.lower()}"
            # venue_abbr and year
            venue_abbr, venue_year = paper["venue"], ""
            if any(char.isdigit() for char in paper["venue"]):
                try:
                    venue_abbr, venue_year = paper["venue"].rsplit(" ", 1)
                    venue_year = " " + venue_year
                except Exception:
                    ...
            name = paper["name"]
            color = "#C55253" if "**" in paper.get("info", "") else "#777777"
            mark = ""
            author = paper["author"]
            if "Junkun Yuan" in author:
                if "Junkun Yuan##**" in author or "Junkun Yuan**##" in author:
                    mark = "<sup>&#10035</sup><sup>&#9993</sup>"
                elif "Junkun Yuan**" in author or author.strip().startswith("Junkun Yuan"):
                    mark = "<sup>&#10035</sup>"
                elif "Junkun Yuan##" in author:
                    mark = "<sup>&#9993</sup>"

            index_items.append(
                f'<a href="#{anchor_id}" class="no_dec"><font color={color}><b>{name}{mark}</b> <font style="color:#AAAAAA;font-size:11px;">({venue_abbr}{venue_year})</font></font></a>'
            )
        
        # Create year section
        year_section = f"""
    <p style="display: flex; flex-wrap: wrap; font-size: 13px; margin-bottom: 8px;">
        <span style="font-weight: bold; color: {year_color}; margin-right: 8px;">{year}:</span>
        {" &nbsp; &nbsp; &nbsp; ".join(index_items)}
    </p>"""
        year_sections.append(year_section)
    
    return "".join(year_sections)


def format_authors_with_et_al(author_string: str, paper_name: str, paper_title: str = "") -> str:
    """
    Format authors with et al. logic for papers with more than 5 authors.
    
    Args:
        author_string: Original author string
        paper_name: Paper name for unique ID generation
    
    Returns:
        Formatted author string with et al. functionality
    """
    # Split authors by comma and clean up
    authors = [author.strip() for author in author_string.split(',')]
    
    # If 4 or fewer authors, format normally
    if len(authors) <= 4:
        formatted_authors = []
        for author in authors:
            # Apply highlighting and formatting
            formatted_author = author.replace("Junkun Yuan", "<b><font color=#404040>Junkun Yuan</font></b>")
            formatted_author = formatted_author.replace("**", "<sup>&#10035</sup>")
            formatted_author = formatted_author.replace("##", "<sup>&#9993</sup>")
            formatted_authors.append(formatted_author)
        return ', '.join(formatted_authors)
    
    # For more than 4 authors, implement et al. logic
    # Find Junkun Yuan's position
    junkun_position = -1
    for i, author in enumerate(authors):
        if "Junkun Yuan" in author:
            junkun_position = i
            break
    
    # Generate unique ID for this paper's author toggle using title
    if paper_title:
        # Use first few words of title to create unique ID
        title_words = paper_title.split()[:4]  # Use first 4 words of title
        paper_id = '_'.join(title_words).replace(':', '').replace(',', '').replace('.', '').lower()
        # Add hash to ensure uniqueness
        import hashlib
        title_hash = hashlib.md5(paper_title.encode()).hexdigest()[:6]
        paper_id = f"{paper_id}_{title_hash}"
    else:
        # Fallback to original method if no title provided
        paper_id = paper_name.replace(' ', '_').replace('-', '_').lower()
    
    # If Junkun Yuan is in first 4 authors, show first 4 + et al.
    if junkun_position < 4:
        visible_authors = authors[:4]
        hidden_authors = authors[4:]
    else:
        # If Junkun Yuan is after position 3, show authors up to Junkun Yuan + et al.
        visible_authors = authors[:junkun_position + 1]
        hidden_authors = authors[junkun_position + 1:]
    
    # Format visible authors
    formatted_visible = []
    for author in visible_authors:
        formatted_author = author.replace("Junkun Yuan", "<b><font color=#404040>Junkun Yuan</font></b>")
        formatted_author = formatted_author.replace("**", "<sup>&#10035</sup>")
        formatted_author = formatted_author.replace("##", "<sup>&#9993</sup>")
        formatted_visible.append(formatted_author)
    
    # Format hidden authors
    formatted_hidden = []
    for author in hidden_authors:
        formatted_author = author.replace("Junkun Yuan", "<b><font color=#404040>Junkun Yuan</font></b>")
        formatted_author = formatted_author.replace("**", "<sup>&#10035</sup>")
        formatted_author = formatted_author.replace("##", "<sup>&#9993</sup>")
        formatted_hidden.append(formatted_author)
    
    # Create the HTML with toggle functionality
    visible_part = ', '.join(formatted_visible)
    hidden_part = ', '.join(formatted_hidden)
    # import pdb; pdb.set_trace()
    
    et_al_html = f"""
    {visible_part}, <span id="et_al_{paper_id}" onclick="toggleAuthors('{paper_id}')" style="cursor: pointer; color: #404040; text-decoration: underline;">et al.</span>
    <span id="hidden_authors_{paper_id}" onclick="toggleAuthors('{paper_id}')" style="display: none; cursor: pointer; color: #404040; text-decoration: underline;">{hidden_part}</span>
    """
    
    return et_al_html


def build_paper(papers: List[Dict[str, Any]]) -> str:
    """
    Build HTML content for publications section.

    Args:
        papers: List of paper dictionaries containing publication details

    Returns:
        HTML string containing the publications section
    """
    content = """
    <h2 id="publications">Publications</h2>
    <p class="larger"><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a> &nbsp;&nbsp; <a href="https://www.semanticscholar.org/author/Junkun-Yuan/2304610230">Semantic Scholar Profile</a></p>
    <p>&#10035: (co-)first author &nbsp;&nbsp; &#9993: corresponding author</p>
    """
    # Add compact index below Google Scholar Profile
    content += build_paper_index(papers, "Publications")

    item_content = ""
    current_year = ""
    color_bar_gen = border_color_generator()
    year_colors = {}  # Store color for each year

    for _, paper in enumerate(papers):
        venue_full = get_venue_all(paper["venue"])
        [venue_name, venue_date] = paper["venue"].rsplit(" ", 1)
        if venue_full.strip():
            venue = f"""(<b><font color=#404040>{venue_name}</font></b>), <b>{venue_date}</b>"""
        else:
            venue = f"""<b><font color=#404040>{venue_name}</font></b> <b>{venue_date}</b>"""
        color = "#C55253" if "**" in paper.get("info", "") else "#404040"
        name = f"""<font color={color}>{paper["name"]}</font>"""
        paper_link = f"""<a href="{paper["pdf_url"]}">{name}</a>"""

        code_link = ""
        if paper.get('code_url', '').strip():
            code_link = f""" &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp; <a href="{paper['code_url']}">code</a>"""

        comment_html = ""
        if "comment" in paper and paper["comment"]:
            comment_html = f"""<p class="paper_detail"><font color=#C55253>{paper["comment"]}</font></p>"""

        # Parse and format date
        try:
            date_obj = datetime.strptime(paper["date"], "%Y%m%d")
            date_formatted = date_obj.strftime("%b %d, %Y")
        except ValueError:
            date_formatted = paper["date"]

        # Format author with et al. logic
        author = format_authors_with_et_al(paper["author"], paper['name'], paper['title'])

        # Update color bar for new year
        paper_year = paper["date"][:4]
        is_first_paper_of_year = False
        if paper_year != current_year:
            color_bar = next(color_bar_gen)
            year_colors[paper_year] = color_bar  # Store color for this year
            current_year = paper_year
            is_first_paper_of_year = True
        
        # Build paper HTML, add anchor for index
        anchor_id = f"{paper['name']}publications"
        
        # Add year display for first paper of each year
        year_display = ""
        if is_first_paper_of_year:
            year_color = year_colors[paper_year]
            year_display = f"""
            <div style="position: absolute; right: 0; top: 0; font-size: 28px; font-weight: bold; color: {year_color}; opacity: 0.8;">
                {paper_year}
            </div>"""
        
        item_content += f"""
        <p class="little_split"></p>
        <div id="{anchor_id}" style="border-left: 16px solid {color_bar}; padding-left: 10px; position: relative;">
        {year_display}
        <div style="height: 0.3em;"></div>
        <p class="paper_title"><i>{paper["title"]}</i></p>
        <p class="paper_detail">{author}</p>
        <p class="paper_detail">{venue_full} {venue}</p>
        <p class="paper_detail"><font color=#404040>{date_formatted}</font> &nbsp;&nbsp;<font color=#CCCCCC>|</font>&nbsp;&nbsp; {paper_link}{code_link}</p>
        {comment_html}
        <div style="height: 0.05em;"></div>
        </div>
        <p class="little_split"></p>
        """

    return content + item_content


def main() -> None:
    """Main function to generate the index.html page."""
    print("Generating main index page...")

    try:
        # Build paper contents
        PUB_LIST = build_paper(PAPERS)

        # Combine all content sections
        html_content = PREFIX + TABLE_OF_CONTENTS + BIOGRAPHY + PUB_LIST + SERVICE + TOP_BUTTON + SUFFIX

        # Write to file
        html_file = "index.html"
        with open(html_file, "w", encoding="utf-8-sig") as f:
            f.write(html_content)

        print(f"Successfully generated {html_file}")

    except Exception as e:
        print(f"Error generating index page: {e}")
        raise


if __name__ == "__main__":
    main()