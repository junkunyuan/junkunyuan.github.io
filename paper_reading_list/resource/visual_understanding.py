LIST = dict()
LIST["file"] = "visual_understanding.html"
LIST["title"] = "Visual Perception & Understanding"
LIST["description"] = "Generate visual signals (e.g., images, video, and 3D)."
LIST["categories"] = ["Foundation Algorithms & Models"]
LIST["papers"] = [
# {
# "title": "",
# "author": "",
# "organization": "",
# "date": "",
# "venue": "",
# "pdf_url": "",
# "code_url": "",
# "name": "",
# "comment": "",
# "category": "",
# "jupyter_notes": "",
# "summary": """""",
# "details": 
# """
# <ul>
#     <li>
# </ul>
# """,
# },
{
"title": "Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective",
"author": "Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang",
"organization": "DAMO Academy, Alibaba Group, Hupan Lab, Zhejiang University, Tsinghua University",
"date": "20250711",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2507.08801",
"code_url": "https://github.com/alibaba-damo-academy/Lumos",
"name": "Lumos-1",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It employs LLM architecture to achieve auto-regressive video generation with some improvement on RoPE and masking strategy.""",
"details": 
"""
<ul>
    <li><b>Structure:</b> Llama with a new RoPE strategy to model multimodal spatiotemporal dependency.
    <li><b>Tokenizer:</b> Cosmos's visual tokenizer with spatiotemporal compression rates of 8x8x4; Chameleon's text encoder.
    <li> <b>Model size:</b> 0.5B, 1B, and 3B.
</ul>
fig: fig1.png 550
cap: <b>Text-to-image</b> generation performance on <b>GenEval</b>.
fig: fig2.png 650
cap: <b>Image-to-video</b> generation performance on <b>VBench-I2V</b>.
fig: fig3.png 650
cap: <b>Text-to-video</b> generation performance on <b>VBench-T2V</b>.
""",
},
]