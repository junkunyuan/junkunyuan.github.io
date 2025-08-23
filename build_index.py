from datetime import datetime
from resource.pub_list import PAPERS
from paper_reading_list.resource.utils import get_venue_all, border_color_generator
from paper_reading_list.resource.utils import TOP_BUTTON

time_now = datetime.now().strftime('%B %d, %Y at %H:%M')

PREFIX = \
f"""
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
                    <p>Research Scientist, &nbsp;<a href="https://hunyuan.tencent.com/">Hunyuan Multimodal Generation Group</a>&nbsp;&nbsp;@&nbsp;&nbsp;<a href="https://www.tencent.com/">Tencent</a></p>
                    <p>yuanjk0921@outlook.com</p>
                    <p>work and live in Shenzhen, China</p>
                    <p><font color=#D0D0D0>Last updated on {time_now} (UTC+8)</font></p>
                    <p><font color="D04040">I am currently on the job market and welcome potential opportunities. Please feel free to reach out to me.</p>
                </td>
                <td style="padding-right: 120px; padding-top: 10px;">
                    <img src="resource/my_photo.jpg" width="160">
                </td>
            </tr>
        </tbody>
    </table>
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


def build_paper(papers):
    content = \
    """
    <h2>Publications</h2>
    <p class="larger"><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a></p>
    """
    item_content = ""
    color_year = ""
    color_bar_generator = border_color_generator()
    for paper in papers:
        venue = f"""<b><font color=#404040>{paper["venue"]}</font></b>"""
        paper_ = """<a href="{paper["pdf_url"]}">paper</a>"""
        venue_all = get_venue_all(paper["venue"])
        code = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{paper['code_url']}">code</a>""" if len(paper['code_url']) > 0 else ""
        comment = f"""<p class="paper_detail"><font color=#D04040>{paper["comment"]}</font></p>""" if "comment" in paper else ""
        date = datetime.strptime(paper["date"], "%Y%m%d").strftime("%b %d, %Y")

        author = paper["author"].replace("Junkun Yuan", "<b><font color=#404040>Junkun Yuan</font></b>")
        author = author.replace("**", "<sup>&#10035</sup>")
        author = author.replace("##", "<sup>&#9993</sup>")

        if paper["date"][:4] != color_year:
            color_bar = next(color_bar_generator)
            color_year = paper["date"][:4]
        
        item_content += \
        f"""
        <p class="little_split"></p>
        <div style="border-left: 14px solid {color_bar}; padding-left: 10px">
        <div style="height: 0.3em;"></div>
        <p class="paper_title"><i>{paper["title"]}</i></p>
        <p class="paper_detail">{author}</p>
        <p class="paper_detail">{date} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <font color=#D0D0D0>{venue_all}</font></p>
        <p class="paper_detail">{paper_}{code}</p>
        {comment}
        <div style="height: 0.05em;"></div>
        </div>
        <p class="little_split"></p>
        """
    content += item_content
    return content


if __name__ == "__main__":
    ## Build paper contents
    paper_content = build_paper(PAPERS)

    ## Build html contents
    html_content = PREFIX + BIOGRAPHY + paper_content + SERVICE + TOP_BUTTON + SUFFIX

    ## Write contents to html
    html_file = "index.html"
    with open(html_file, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    