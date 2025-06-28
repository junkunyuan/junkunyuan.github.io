VISUAL_GENERATIVE_MODELS = dict()
VISUAL_GENERATIVE_MODELS["file"] = "visual_generative_models.html"
VISUAL_GENERATIVE_MODELS["title"] = "Visual Generative Models"
VISUAL_GENERATIVE_MODELS["description"] = "Models that learn to generate visual signals, e.g., images, videos, 3D, etc."
VISUAL_GENERATIVE_MODELS["categories"] = ["Foundation Algorithms & Models", "Datasets & Evaluation"]
VISUAL_GENERATIVE_MODELS["papers"] = [
{
"title": "Denoising Diffusion Probabilistic Models",
"author": "Jonathan Ho, Ajay Jain, Pieter Abbeel",
"organization": "UC Berkeley",
"date": "20200619",
"venue": "NeurIPS 2020",
"pdf_url": "https://arxiv.org/pdf/2006.11239",
"code_url": "https://github.com/hojonathanho/diffusion/",
"name": "DDPM",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "visual_generative_models-ddpm.ipynb",
"summary": "It proposes <b>denoising diffusion probabilistic models</b> that iteratively denoises data from random noise.",
"details": 
"""
<figure>
    <img src='resource/figs/2020-06-19-ddpm-fig1.png' width=500>
    <figcaption><b>Figure 1.</b> Diffusion (forward) and denoising (reverse) processes of DDPM.</figcaption>
</figure>
<br>
<figure>
    <img src='resource/figs/2020-06-19-ddpm-fig2.png' width=700>
    <figcaption><b>Figure 2.</b> Training and sampling algorithms of DDPM.</figcaption>
</figure>
""",
},
{
"title": "Unified Reward Model for Multimodal Understanding and Generation",
"author": "Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, Jiaqi Wang",
"organization": "Fudan University, Shanghai Innovation Institute, Shanghai AI Lab, Shanghai Academy of Artificial Intelligence for Science",
"date": "20250307",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.05236",
"code_url": "https://github.com/CodeGoat24/UnifiedReward/",
"name": "UnifiedReward",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"summary": "It fine-tunes LLaVA-OneVision 7B for both <b>multimodal understanding & generation evaluation</b> by pairwise ranking & pointwise scoring.",
"details": ""
}
]
