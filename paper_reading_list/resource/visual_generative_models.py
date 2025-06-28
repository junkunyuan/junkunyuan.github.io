VISUAL_GENERATIVE_MODELS = dict()
VISUAL_GENERATIVE_MODELS["file"] = "visual_generative_models.html"
VISUAL_GENERATIVE_MODELS["title"] = "Visual Generative Models"
VISUAL_GENERATIVE_MODELS["description"] = "Models that learn to generate visual signals, e.g., images, videos, 3D, etc."
VISUAL_GENERATIVE_MODELS["categories"] = ["Foundation Algorithms & Models", "Datasets & Evaluation"]
VISUAL_GENERATIVE_MODELS["papers"] = [
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
},
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
"title": "Neural Discrete Representation Learning",
"author": "Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu",
"organization": "DeepMind",
"date": "20171102",
"venue": "NeurIPS 2017",
"pdf_url": "https://arxiv.org/pdf/1711.00937",
"code_url": "",
"name": "VQ-VAE",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": "It proposes <b>vector quantised variational autoencoder</b> to generate discrete codes while the prior is also learned.",
"details": 
"""
<ul>
    <li><b>Posterior collapse problem:</b> a strong decoder and a strong KL constraint could make the learned posterior <i>q(z|x)</i> very close to prior <i>p(z)</i>, so that the conditional generation task collapses to an unconditional generation task.</li>
    <li><b>How VQ-VAE avoids the collapse problem by employing discrete codes/latents?</b> (1) It learns <i>q(z|x)</i> by choosing one from some candidates rather than directly generating a simple prior; (2) The learned <i>q(z|x)</i> is continuous but <i>p(z)</i> is discrete, so the encoder can not be "lazy".</li>
    <li><b>Optimization objectives:</b> (1) The decoder is optimized by a recontruction loss; (2) The encoder is optimized by a reconstruction loss and a matching loss; (3) The embedding is optimized by a matching loss.</li>
    <li><b>How to back-propagate gradient with quantization exists? Straight-Through Estimator:</b> directly let the graident of loss to the quantized embedding equal to the gradient of loss to the embedding that before being quantized.</li>
</ul>
<figure>
    <img src='resource/figs/2017-11-02-VQ-VAE-fig1.png' width=900>
</figure>
""",
},
]
