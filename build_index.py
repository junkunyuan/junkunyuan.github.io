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
    <div id="layout-content" style="margin-top:25px">
    <table>
        <tbody>
            <tr>
                <td width="870">
                    <h1>Junkun Yuan &nbsp; 袁俊坤</h1>
                    <p>Research Scientist, &nbsp;Hunyuan Multimodal Model Group&nbsp;&nbsp;@&nbsp;&nbsp;Tencent</p>
                    <p>yuanjk0921@outlook.com</p>
                    <p>work and live in Shenzhen, China</p>
                    <p><font color=#D0D0D0>Last updated on {TIME_NOW} (UTC+8)</font></p>
                    <p><font color="D04040">I am currently on the job market and welcome potential opportunities. Please feel free to reach out to me.</p>
                </td>
                <td style="padding-right: 120px; padding-top: 10px;">
                    <img src="resource/my_photo.jpg" width="160">
                </td>
            </tr>
        </tbody>
    </table>
</head>
<body>
"""
BIOGRAPHY = \
"""
<h2>Biography</h2>
<p>
    I have been working as a research scientist in the Foundation Model Team of the Hunyuan Multimodal Model Group at Tencent since Jul 2024, working with <a href="https://scholar.google.com.hk/citations?user=igtXP_kAAAAJ&hl=zh-CN&oi=ao">Zhao Zhong</a> and <a href="https://scholar.google.com.hk/citations?user=FJwtMf0AAAAJ&hl=zh-CN&oi=ao">Liefeng Bo</a>. I am focusing on multimodal generative foundation models and their various downstream applications.
    <br><br>

    During Sep 2023 — Jul 2024, I interned in the Hunyuan Multimodal Model Group at Tencent, working with <a href="https://scholar.google.com/citations?user=AjxoEpIAAAAJ">Wei Liu</a>.
    
    During Jul 2022 — Aug 2023, I interned in the Computer Vision Group at Baidu, working with <a href="https://scholar.google.com/citations?user=PSzJxD8AAAAJ">Xinyu Zhang</a> and <a href="https://scholar.google.com/citations?user=z5SPCmgAAAAJ">Jingdong Wang</a>.<br><br>

    I received my Ph.D. degree in Computer Science from Zhejiang University (2019 — 2024), co-supervised by professors of <a href="https://scholar.google.com/citations?user=FOsNiMQAAAAJ">Kun Kuang</a>, <a href="https://person.zju.edu.cn/0096005">Lanfen Lin</a>, and 
  <a href="https://scholar.google.com/citations?user=XJLn4MYAAAAJ">Fei Wu</a>. I received my B.S. degree in Automation from Zhejiang University of Technology (2015 — 2019), co-supervised by professors of <a href="https://scholar.google.com.hk/citations?user=smi7bpoAAAAJ&hl=zh-CN&oi=ao">Qi Xuan</a> and <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=CnBn6FwAAAAJ">Li Yu</a>.<br><br>

  I have been fortunate to work closely with some friends such as <a href="https://scholar.google.com.hk/citations?user=F5P_8NkAAAAJ&hl=zh-CN&oi=ao">Defang Chen</a>, <a href="https://scholar.google.com.hk/citations?user=kwBR1ygAAAAJ&hl=zh-CN&oi=ao">Yue Ma</a>, their insights also profoundly shape my approach to research.
</p>
"""
SERVICE = \
"""
<br>
<h2>Professional Service</h2>
<ul>
<li> <b>Conference Reviewer.</b>&nbsp;&nbsp; 
CVPR 2021 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
ICCV 2023 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
AAAI 2023, 2026 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
MM 2023

<li> <b>Journal Reviewer.</b>&nbsp;&nbsp; 
TNNLS 2022 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
TCSVT 2023 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
PR 2026 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
NN 2023 &nbsp;&nbsp;<font color=#A0A0A0>|</font>&nbsp;&nbsp;
TKDD 2023
</ul>
"""
SUFFIX = \
"""
</body>
</html>
"""


def build_paper(papers: List[Dict[str, Any]]) -> str:
    """
    Build HTML content for publications section.

    Args:
        papers: List of paper dictionaries containing publication details

    Returns:
        HTML string containing the publications section
    """
    content = """
    <h2>Publications</h2>
    <p class="larger"><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a></p>
    """

    item_content = ""
    current_year = ""
    color_bar_gen = border_color_generator()

    for paper in papers:
        # Format venue and links
        venue = f"""<b><font color=#404040>{paper["venue"]}</font></b>"""
        paper_link = f"""<a href="{paper["pdf_url"]}">paper</a>"""
        venue_full = get_venue_all(paper["venue"])

        code_link = ""
        if paper.get('code_url', '').strip():
            code_link = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{paper['code_url']}">code</a>"""

        comment_html = ""
        if "comment" in paper and paper["comment"]:
            comment_html = f"""<p class="paper_detail"><font color=#D04040>{paper["comment"]}</font></p>"""

        # Parse and format date
        try:
            date_obj = datetime.strptime(paper["date"], "%Y%m%d")
            date_formatted = date_obj.strftime("%b %d, %Y")
        except ValueError:
            date_formatted = paper["date"]

        # Format author with highlights
        author = paper["author"].replace("Junkun Yuan", "<b><font color=#404040>Junkun Yuan</font></b>")
        author = author.replace("**", "<sup>&#10035</sup>")
        author = author.replace("##", "<sup>&#9993</sup>")

        # Update color bar for new year
        paper_year = paper["date"][:4]
        if paper_year != current_year:
            color_bar = next(color_bar_gen)
            current_year = paper_year

        # Build paper HTML
        item_content += f"""
        <p class="little_split"></p>
        <div style="border-left: 14px solid {color_bar}; padding-left: 10px">
        <div style="height: 0.3em;"></div>
        <p class="paper_title"><i>{paper["title"]}</i></p>
        <p class="paper_detail">{author}</p>
        <p class="paper_detail">{date_formatted} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <font color=#D0D0D0>{venue_full}</font></p>
        <p class="paper_detail">{paper_link}{code_link}</p>
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
        paper_content = build_paper(PAPERS)

        # Combine all content sections
        html_content = PREFIX + BIOGRAPHY + paper_content + SERVICE + TOP_BUTTON + SUFFIX

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
    