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
"title": "Exploring Simple Siamese Representation Learning",
"author": "Xinlei Chen, Kaiming He",
"organization": "Facebook AI Research (FAIR)",
"date": "20201120",
"venue": "CVPR 2021",
"pdf_url": "https://arxiv.org/pdf/2011.10566",
"code_url": "https://github.com/facebookresearch/simsiam",
"name": "SimSiam",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces a simple yet effective <b>Siamese architecture</b> that learns visual representations by contrasting positive and negative pairs.
""",
"details": 
"""
""",
},
{
"title": "Momentum Contrast for Unsupervised Visual Representation Learning",
"author": "Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick",
"organization": "Facebook AI Research (FAIR)",
"date": "20191113",
"venue": "CVPR 2020",
"pdf_url": "https://arxiv.org/pdf/1911.05722",
"code_url": "https://github.com/facebookresearch/moco",
"name": "MoCo",
"comment": "It advances unsupervised visual representation learning by introducing a momentum-updated encoder with a dynamic queue of negatives, enabling scalable contrastive training that rivaled supervised pretraining and shaped subsequent self-supervised learning research. It has over 17,000 citations (as of Sep 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces <b>momentum contrast</b> to train Vision Transformers in a self-supervised manner.
""",
"details": 
"""
""",
},
{
"title": "BEiT: BERT Pre-Training of Image Transformers",
"author": "Hangbo Bao, Li Dong, Songhao Piao, Furu Wei",
"organization": "Harbin Institute of Technology, Microsoft Research",
"date": "20210615",
"venue": "ICLR 2022",
"pdf_url": "https://arxiv.org/pdf/2106.08254",
"code_url": "https://github.com/microsoft/unilm",
"name": "BEiT",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces <b>masked image modeling</b> with discrete visual tokens to pre-train Vision Transformers in a self-supervised BERT-like fashion.
""",
"details": 
"""
""",
},
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