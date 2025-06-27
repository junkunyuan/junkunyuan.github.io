from datetime import datetime

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
                    <p><FONT color=#B0B0B0>Last updated on {time_now} (UTC+8)</p>
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
  <a href="https://scholar.google.com/citations?user=XJLn4MYAAAAJ">Fei Wu</a>.
</p>
"""
SUFFIX  = \
"""
</body>
</html>
"""
PUB = [
    ## contribute equally: <sup>&#10035</sup>; corresponding: <sup>&#9993</sup>
    {
    "title": "HunyuanVideo: A Systematic Framework For Large Video Generative Models",
    "author": "Hunyuan Multimodal Generation Team at Tencent",
    "date": "Dec. 2024",
    "venue": "arXiv 2024",
    "venue_all": "",
    "pdf_url": "https://arxiv.org/pdf/2412.03603",
    "code_url": "https://github.com/Tencent-Hunyuan/HunyuanVideo",
    "comment": "<font color=#FF000>It is the first open-sourced large-scale video generation model with 13B parameters. It has 200+ citations and 10K+ GitHub stars (as of June 2025)."
    },
    {
    "title": "Follow-Your-Canvas: Higher-Resolution Video Outpainting with Extensive Content Generation",
    "author": "Qihua Chen<sup>&#10035</sup>, Yue Ma<sup>&#10035</sup>, Hongfa Wang<sup>&#10035</sup>, Junkun Yuan<sup>&#10035</sup><sup>&#9993</sup>, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen<sup>&#9993</sup>, and Wei Liu",
    "date": "Sep. 2024",
    "venue": "AAAI 2025",
    "venue_all": "AAAI Conference on Artificial Intelligence",
    "pdf_url": "https://arxiv.org/pdf/2409.01055",
    "code_url": "https://github.com/mayuelala/FollowYourCanvas"
    },
    {
    "title": "Mutual Prompt Leaning for Vision Language Models",
    "author": "Sifan Long<sup>&#10035</sup>, Zhen Zhao<sup>&#10035</sup>, Junkun Yuan<sup>&#10035</sup>, Zichang Tan<sup>&#10035</sup>, Jiangjiang Liu, Jingyuan Feng, Shengsheng Wang<sup>&#9993</sup>, and Jingdong Wang",
    "date": "Sep. 2024",
    "venue": "IJCV 2024",
    "venue_all": "International Journal of Computer Vision",
    "pdf_url": "https://arxiv.org/pdf/2303.17169",
    "code_url": ""
    },
    {
    "title": "Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation",
    "author": "Yue Ma<sup>&#10035</sup>, Hongyu Liu<sup>&#10035</sup>, Hongfa Wang<sup>&#10035</sup>, Heng Pan<sup>&#10035</sup>, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu<sup>&#9993</sup>, and Qifeng Chen<sup>&#9993</sup>",
    "date": "June 2024",
    "venue": "SIGGRAPH-Asia 2024",
    "venue_all": "ACM SIGGRAPH Annual Conference in Asia",
    "pdf_url": "https://arxiv.org/pdf/2406.01900",
    "code_url": "https://github.com/mayuelala/FollowYourEmoji"
    },
    {
    "title": "Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding",
    "author": "Hunyuan Multimodal Generation Team at Tencent (as an intern)",
    "date": "May 2024",
    "venue": "arXiv 2024",
    "venue_all": "",
    "pdf_url": "https://arxiv.org/pdf/2405.08748",
    "code_url": "https://github.com/Tencent/HunyuanDiT"
    },
    {
    "title": "HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception",
    "author": "Junkun Yuan**, Xinyu Zhang**##, Hao Zhou, Jian Wang, Zhongwei Qiu, Zhiyin Shao, Shaofeng Zhang, Sifan Long, Kun Kuang##, Kun Yao, Junyu Han, Errui Ding, Lanfen Lin, Fei Wu, and Jingdong Wang##",
    "date": "Oct. 2023",
    "venue": "NeurIPS 2023",
    "venue_all": "Advances in Neural Information Processing Systems",
    "pdf_url": "https://arxiv.org/pdf/2310.20695",
    "code_url": "https://github.com/junkunyuan/HAP"
    },
    {
    "title": "MAP: Towards Balanced Generalization of IID and OOD through Model-Agnostic Adapters",
    "author": "Min Zhang, Junkun Yuan, Yue He, Wenbin Li, Zhengyu Chen, and Kun Kuang##",
    "date": "Oct. 2023",
    "venue": "ICCV 2023",
    "venue_all": "International Conference on Computer Vision",
    "pdf_url": "https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_MAP_Towards_Balanced_Generalization_of_IID_and_OOD_through_Model-Agnostic_ICCV_2023_paper.pdf",
    "code_url": ""
    },
    {
    "title": "Neural Collapse Anchored Prompt Tuning for Generalizable Vision-Language Models",
    "author": "Didi Zhu, Zexi Li, Min Zhang, Junkun Yuan, Yunfeng Shao, Jiashuo Liu, Kun Kuang<sup>&#9993</sup>, Yinchuan Li, and Chao Wu",
    "date": "June. 2023",
    "venue": "KDD 2024",
    "venue_all": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "pdf_url": "https://arxiv.org/pdf/2306.15955",
    "code_url": ""
    },
    {
    "title": "Quantitatively Measuring and Contrastively Exploring Heterogeneity for Domain Generalization",
    "author": "Yunze Tong, Junkun Yuan, Min Zhang, Didi Zhu, Keli Zhang, Fei Wu, and Kun Kuang##",
    "date": "May 2023",
    "venue": "KDD 2023",
    "venue_all": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "pdf_url": "https://arxiv.org/pdf/2305.15889",
    "code_url": "https://github.com/YunzeTong/HTCL"
    },
    {
    "title": "Universal Domain Adaptation via Compressive Attention Matching",
    "author": "Didi Zhu**, Yincuan Li**, Junkun Yuan, Zexi Li, Kun Kuang, and Chao Wu##",
    "date": "Apr. 2023",
    "venue": "ICCV 2023",
    "venue_all": "International Conference on Computer Vision",
    "pdf_url": "https://arxiv.org/pdf/2304.11862",
    "code_url": ""
    },
    {
    "title": "Task-Oriented Multi-Modal Mutual Leaning for Vision-Language Models",
    "author": "Sifan Long**, Zhen Zhao**, Junkun Yuan**, Zichang Tan, Jiangjiang Liu, Luping Zhou, Shengsheng Wang##, and Jingdong Wang##",
    "date": "Mar. 2023",
    "venue": "ICCV 2023",
    "venue_all": "International Conference on Computer Vision",
    "pdf_url": "https://arxiv.org/pdf/2303.17169",
    "code_url": ""
    },
    {
    "title": "CAE v2: Context Autoencoder with CLIP Target",
    "author": "Xinyu Zhang**, Jiahui Chen**, Junkun Yuan, Qiang Chen, Jian Wang, Xiaodi Wang, Shumin Han, Xiaokang Chen, Jimin Pi, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang##",
    "date": "Nov. 2022",
    "venue": "TMLR 2023",
    "venue_all": "Transactions on Machine Learning Research",
    "pdf_url": "https://arxiv.org/pdf/2211.09799",
    "code_url": "https://github.com/Atten4Vis/CAE"
    },
    {
    "title": "Label-Efficient Domain Generalization via Collaborative Exploration and Generalization",
    "author": "Junkun Yuan**, Xu Ma**, Defang Chen, Kun Kuang##, Fei Wu, and Lanfen Lin",
    "date": "Aug. 2022",
    "venue": "MM 2022",
    "venue_all": "International Conference on Multimedia",
    "pdf_url": "https://arxiv.org/pdf/2208.03644",
    "code_url": "https://github.com/junkunyuan/CEG"
    },
    {
    "title": "Domain-Specific Bias Filtering for Single Labeled Domain Generalization",
    "author": "Junkun Yuan**, Xu Ma**, Defang Chen, Kun Kuang##, Fei Wu, and Lanfen Lin",
    "date": "Oc. 2021",
    "venue": "IJCV 2022",
    "venue_all": "International Journal of Computer Vision",
    "pdf_url": "https://arxiv.org/pdf/2110.00726",
    "code_url": "https://github.com/junkunyuan/DSBF"
    },
    {
    "title": "Collaborative Semantic Aggregation and Calibration for Federated Domain Generalization",
    "author": "Junkun Yuan**, Xu Ma**, Defang Chen, Fei Wu, Lanfen Lin, and Kun Kuang##",
    "date": "Oct. 2021",
    "venue": "TKDE 2023",
    "venue_all": "IEEE Transactions on Knowledge and Data Engineering",
    "pdf_url": "https://arxiv.org/pdf/2110.06736",
    "code_url": "https://github.com/junkunyuan/CSAC"
    },

]

def build_pub(pub_list):
    content = \
    """
    <p class="little_split"></p>
    <h2>Publications</h2>
    <p><a href="https://scholar.google.com/citations?user=j3iFVPsAAAAJ">Google Scholar Profile</a></p>
    """
    item_content = ""
    for pub in pub_list:
        venue_all = f"""{pub["venue_all"]}""" if len(pub["venue_all"]) > 0 else ""

        code = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{pub['code_url']}">code</a>""" if len(pub['code_url']) > 0 else ""

        venue = f"""<b><a href="{pub["pdf_url"]}"><FONT color=#202020>{pub["venue"]}</FONT></a></b>"""

        comment = f"""<p class="pub_detail">{pub["comment"]}</p>""" if "comment" in pub else ""

        author = pub["author"].replace("Junkun Yuan", "<b><FONT color=#202020>Junkun Yuan</FONT></b>")
        author = author.replace("**", "<sup>&#10035</sup>")
        author = author.replace("##", "<sup>&#9993</sup>")
        
        item_content += \
        f"""
        <p class="little_split"></p>
        <div style="border-left: 8px solid #ADDEFF; padding-left: 10px">
        <div style="height: 0.3em;"></div>
        <p class="pub_title"><i>{pub["title"]}</i></p>
        <p class="pub_detail">{author}</p>
        <p class="pub_detail">{pub["date"]} {code} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <FONT color=#B0B0B0>{venue_all}</FONT></p>
        {comment}
        <div style="height: 0.05em;"></div>
        </div>
        <p class="little_split"></p>
        """
    content += item_content
    return content


if __name__ == "__main__":
    pub_content = build_pub(PUB)

    html_content = PREFIX + BIOGRAPHY + pub_content + SUFFIX

    ## Write contents to html
    html_file = "index.html"
    with open(html_file, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    