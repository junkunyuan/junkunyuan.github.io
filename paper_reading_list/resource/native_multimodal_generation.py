NATIVE_MULTIMODAL_GENERATION = dict()
NATIVE_MULTIMODAL_GENERATION["file"] = "native_multimodal_generation.html"
NATIVE_MULTIMODAL_GENERATION["title"] = "Native Multimodal Generation"
NATIVE_MULTIMODAL_GENERATION["description"] = "Learn to jointly generate multimodal signals, e.g., text, images, audio, etc."
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
## <b>Figure 1.</b> 
# """,
# },
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