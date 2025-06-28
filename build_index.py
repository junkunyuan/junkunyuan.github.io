from datetime import datetime
from resource.pub_list import PUB
from paper_reading_list.resource.venue_name import get_venue_all

time_now = datetime.now().strftime('%B %d, %Y at %H:%M')

BORDER_COLOR_MAPPING = {
    "2024": "#ADDEFF",
    "2023": "#FFD6AD",
    "2022": "#B2EEC8",
    "2021": "#FFBBCC",
}

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
                <td style="padding-right: 120px;">
                    <img src="resource/my_photo.jpg" width="150">
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


def build_pub(pub_list):
    content = \
    """
    <h2>Publications</h2>
    <p><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a></p>
    """
    item_content = ""
    for pub in pub_list:
        venue = f"""<b><a href="{pub["pdf_url"]}"><font color=#202020>{pub["venue"]}</font></a></b>"""
        venue_all = get_venue_all(pub["venue"])
        code = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{pub['code_url']}">code</a>""" if len(pub['code_url']) > 0 else ""
        comment = f"""<p class="pub_detail">{pub["comment"]}</p>""" if "comment" in pub else ""

        author = pub["author"].replace("Junkun Yuan", "<b><font color=#202020>Junkun Yuan</font></b>")
        author = author.replace("**", "<sup>&#10035</sup>")
        author = author.replace("##", "<sup>&#9993</sup>")

        for key, value in BORDER_COLOR_MAPPING.items():
            if key in pub["date"]:
                border_color = value
        
        item_content += \
        f"""
        <p class="little_split"></p>
        <div style="border-left: 8px solid {border_color}; padding-left: 10px">
        <div style="height: 0.3em;"></div>
        <p class="pub_title"><i>{pub["title"]}</i></p>
        <p class="pub_detail">{author}</p>
        <p class="pub_detail">{pub["date"]} {code} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <font color=#B0B0B0>{venue_all}</font></p>
        {comment}
        <div style="height: 0.05em;"></div>
        </div>
        <p class="little_split"></p>
        """
    content += item_content
    return content


if __name__ == "__main__":
    ## Build publication contents
    pub_content = build_pub(PUB)

    ## Build html contents
    html_content = PREFIX + BIOGRAPHY + pub_content + SUFFIX

    ## Write contents to html
    html_file = "index.html"
    with open(html_file, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    