from datetime import datetime
from resource.pub_list import PAPERS
from paper_reading_list.resource.utils import get_venue_all, border_color_generator

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
                    <p>Research Scientist, &nbsp;<a href="https://hunyuan.tencent.com/">Hunyuan Multimodal Generation Team</a>&nbsp;&nbsp;@&nbsp;&nbsp;<a href="https://www.tencent.com/">Tencent</a></p>
                    <p>yuanjk0921@outlook.com</p>
                    <p>work and live in Shenzhen, China</p>
                    <p><font color=#B0B0B0>Last updated on {time_now} (UTC+8)</font></p>
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
  I am a research scientist in <a href="https://hunyuan.tencent.com/">Hunyuan Multimodal Generation Team</a> at <a href="https://www.tencent.com/">Tencent</a>, working on multimodal generative foundation models.
  <br><br>

  I previously worked/interned in <a href="https://hunyuan.tencent.com/">Hunyuan Multimodal Generation Team</a> at <a href="https://www.tencent.com/">Tencent</a> from 2023 to 2025 (working with <a href="https://scholar.google.com/citations?user=AjxoEpIAAAAJ">Wei Liu</a>), and in <a href="http://vis.baidu.com/">Computer Vision Group</a> at <a href="https://home.baidu.com/">Baidu</a> from 2022 to 2023 (working with <a href="https://scholar.google.com/citations?user=PSzJxD8AAAAJ">Xinyu Zhang</a> and <a href="https://scholar.google.com/citations?user=z5SPCmgAAAAJ">Jingdong Wang</a>).<br><br>

  I received my PhD degree from <a href="http://www.zju.edu.cn/">Zhejiang University</a> in 2024, co-supervised by professors of <a href="https://scholar.google.com/citations?user=FOsNiMQAAAAJ">Kun Kuang</a>, <a href="https://person.zju.edu.cn/0096005">Lanfen Lin</a>, and 
  <a href="https://scholar.google.com/citations?user=XJLn4MYAAAAJ">Fei Wu</a>.<br><br>
</p>
"""
SUFFIX  = \
"""
</body>
</html>
"""


def build_paper(papers):
    content = \
    """
    <h2>Publications</h2>
    <p><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a></p>
    """
    item_content = ""
    color_year = ""
    color_bar_generator = border_color_generator()
    for paper in papers:
        venue = f"""<b><a href="{paper["pdf_url"]}"><font color=#202020>{paper["venue"]}</font></a></b>"""
        venue_all = get_venue_all(paper["venue"])
        code = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{paper['code_url']}">code</a>""" if len(paper['code_url']) > 0 else ""
        comment = f"""<p class="paper_detail"><font color=#FF000>{paper["comment"]}</font></p>""" if "comment" in paper else ""
        date = datetime.strptime(paper["date"], "%Y%m%d").strftime("%b %d, %Y")

        author = paper["author"].replace("Junkun Yuan", "<b><font color=#202020>Junkun Yuan</font></b>")
        author = author.replace("**", "<sup>&#10035</sup>")
        author = author.replace("##", "<sup>&#9993</sup>")

        if paper["date"][:4] != color_year:
            color_bar = next(color_bar_generator)
            color_year = paper["date"][:4]
        
        item_content += \
        f"""
        <p class="little_split"></p>
        <div style="border-left: 8px solid {color_bar}; padding-left: 10px">
        <div style="height: 0.3em;"></div>
        <p class="paper_title"><i>{paper["title"]}</i></p>
        <p class="paper_detail">{author}</p>
        <p class="paper_detail">{date} {code} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <font color=#B0B0B0>{venue_all}</font></p>
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
    html_content = PREFIX + BIOGRAPHY + paper_content + SUFFIX

    ## Write contents to html
    html_file = "index.html"
    with open(html_file, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    