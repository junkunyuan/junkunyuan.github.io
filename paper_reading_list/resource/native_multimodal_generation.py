NATIVE_MULTIMODAL_GENERATION = dict()
NATIVE_MULTIMODAL_GENERATION["file"] = "native_multimodal_generation.html"
NATIVE_MULTIMODAL_GENERATION["title"] = "Native Multimodal Generation"
NATIVE_MULTIMODAL_GENERATION["description"] = "Process and generate multiple modalities (e.g., text, images, video, audio) within a unified architecture."
NATIVE_MULTIMODAL_GENERATION["categories"] = ["Foundation Algorithms & Models"]
NATIVE_MULTIMODAL_GENERATION["papers"] = [
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
# <figure>
#     <img src='' width=500>
#     <figcaption>
#     <b>Figure 1.</b> 
#     </figcaption>
# </figure>
# """,
# },
{
"title": "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model",
"author": "Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy",
"organization": "Meta, Waymo, University of Southern California",
"date": "20240820",
"venue": "ICLR 2025",
"pdf_url": "https://arxiv.org/pdf/2408.11039",
"code_url": "",
"name": "Transfusion",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It trains a unified model (7B) on 2T multi-modal tokens by predicting discrete text tokens and diffusing continuous image tokens.""",
"details": 
"""
<ul>
    <li> Data: use total 2T tokens from: (1) Llama 2 tokenizer and corpus (2T tokens), (2) 380M Shutterstock images and captions (resized to 256x256).
    <li> Training details: optimizer=AdamW, lr=3e-4, 250K steps, lambda=5, train_timesteps=1000, infer_timesteps=250, cfg=3.
    <li> In text-to-image generation task, Transfusion exceeds Chameleon at less than a third of the compute.
    <li> In image-to-text generation task, Transfusion exceeds Chameleon at 21.8% of the FLOPs.
    <li> In text-to-text generation task, Transfusion exceeds Chameleon at 50% of FLOPs.
</ul>
<figure>
    <img src='2024-08-20-Transfusion-fig1.png' width=500>
    <figcaption>
    <b>Figure 1.</b> <b>Transfusion structure.</b> It applies next-token prediction on discrete text tokens and diffusion loss on continuous image tokens: <i>L=L_LM+lambda*L_diffusion</i>. It uses <i>modality-specific</i> components with unshared parameters: embedding layer for text, and VAE (U-Net or linear structure, 8x8-8c) with linear or up/down blocks for images. It applies causal mask on text tokens and bidirectional mask on image tokens.
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig2.png' width=500>
    <figcaption>
    <b>Figure 2.</b> Transfusion outperforms Chameleon while scaling.
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig3.png' width=400>
    <figcaption>
    <b>Figure 3.</b> Transfusion outperforms Chameleon by using few FLOPs, both are 7B. 
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig4.png' width=500>
    <figcaption>
    <b>Figure 4.</b> Transfusion achieves competitive results compared with Llama2. 
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig5.png' width=400>
    <figcaption>
    <b>Figure 5.</b> <b>Encoder:</b> U-Net is better than linear (maybe bring more inductive bias). <b>Attention:</b> bidirectional is better than causal. 
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig6.png' width=500>
    <figcaption>
    <b>Figure 6.</b> Small patch size leads to better performance for providing more visual tokens.
    </figcaption>
</figure>
<figure>
    <img src='2024-08-20-Transfusion-fig7.png' width=500>
    <figcaption>
    <b>Figure 7.</b> Final performance.
    </figcaption>
</figure>
""",
},
{
"title": "GPT-4o System Card",
"author": "",
"organization": "OpenAI",
"date": "20241025",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2410.21276?",
"code_url": "",
"name": "GPT-4o",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It proposes a unified autoregressive model trained end-to-end across text, vision, and audio.""",
"details": 
"""
<figure>
    <img src='2025-04-08-GPT-4o-Empirical-Study-fig1.png' width=500>
    <img src='2025-04-08-GPT-4o-Empirical-Study-fig2.png' width=500>
    <figcaption>
    <b>Figure 1.</b> <b>Visual generation capability evaluation.</b> <i>Text rendering:</i> correct spelling, alignment, formatting in document-style. <i>Compositional generation and prompt following:</i> accrately assembling complex scene elements, styles, attributes. <i>Geometric consistency and viewpoint realism:</i> 3D view synthesis, camera control, depth-conditioned rendering. <i>Comprehensive image transformation:</i> from low-level to high-level tasks.
    </figcaption>
</figure>
""",
},
{
"title": "Unified Autoregressive Visual Generation and Understanding with Continuous Tokens",
"author": "Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, Michael Rubinstein, Michalis Raptis, Deqing Sun, Radu Soricut",
"organization": "Google DeepMind, MIT",
"date": "20250317",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.13436",
"code_url": "",
"name": "UniFluid",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It achieves visual generation and understanding by applying diffusion loss on continuous visual tokens and cross-entropy loss on discrete text tokens.""",
"details": 
"""
<figure>
    <img src='2025-03-17-UniFluid-fig1.png' width=500>
    <figcaption><b>Figure 1.</b> <b>Framework:</b> joint training of visual generation and understanding tasks through next-token prediction. <b>Tokenizer:</b> use VAE to provide tokens for visual generation, use SigLIP to provide tokens for visual understanding, use SentencePiece to provide text tokens. <b>Prediction head:</b> use <i>modality-specific prediction heads</i> to calculate losses and sampling for each modality. <b>Loss:</b> image understanding loss on text answer + image generation loss on image tokens. <b>Training details:</b> batchsize=2048, optimizer=AdamW, lr=1e-4, steps=1M, init_ckpt=Gemma-2.
    </figcaption>
</figure>
<figure>
    <img src='2025-03-17-UniFluid-fig3.png' width=800>
    <figcaption><b>Figure 2.</b> There is <b>trade-off</b> between generation & understanding.</figcaption>
</figure>
<figure>
    <img src='2025-03-17-UniFluid-fig2.png' width=250>
    <figcaption><b>Figure 3.</b> <b>Unified training improves generation.</b></figcaption>
</figure>
<figure>
    <img src='2025-03-17-UniFluid-fig4.png' width=500>
    <figcaption><b>Figure 4.</b> <b>Better pre-trained LLM backbone</b> leads to better visual generation and understanding performance.</figcaption>
</figure>
""",
},
]