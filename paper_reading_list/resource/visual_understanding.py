LIST = dict()
LIST["file"] = "visual_understanding.html"
LIST["title"] = "Visual Understanding"
LIST["description"] = "Percept and understand visual signals by supervised or unsupervised learning."
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
# "info": "",
# "summary": 
# """
# """,
# "details": 
# """
# """,
# },
{
"title": "An Empirical Study of Training Self-Supervised Vision Transformers",
"author": "Xinlei Chen, Saining Xie, Kaiming He",
"organization": "Facebook AI Research (FAIR)",
"date": "20210405",
"venue": "ICCV 2021",
"pdf_url": "https://arxiv.org/pdf/2104.02057",
"code_url": "https://github.com/facebookresearch/moco-v3",
"name": "MoCo v3",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces a random patch projection trick that <b>freezes the first ViT layer</b> to stabilize contrastive self-supervised training.
""",
"details": 
"""
""",
},
{
"title": "Masked Autoencoders Are Scalable Vision Learners",
"author": "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick",
"organization": "Facebook AI Research (FAIR)",
"date": "20211111",
"venue": "CVPR 2022",
"pdf_url": "https://arxiv.org/pdf/2111.06377",
"code_url": "https://github.com/facebookresearch/mae",
"name": "MAE",
"comment": "It introduces an efficient self-supervised learning paradigm that reconstructs missing image patches, enabling scalable pretraining with reduced computational cost, and significantly improving performance and transferability across vision benchmarks. It has over 11,000 citations (as of Sep 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces a <b>masked autoencoder</b> that reconstructs 75% masked patches, enabling scalable self-supervised pre-training of Vision Transformers.
""",
"details": 
"""
""",
},
]