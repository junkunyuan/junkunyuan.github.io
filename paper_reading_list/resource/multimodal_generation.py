MULTIMODAL_GENERATION = dict()
MULTIMODAL_GENERATION["file"] = "multimodal_generation.html"
MULTIMODAL_GENERATION["title"] = "Multimodal Generation"
MULTIMODAL_GENERATION["description"] = "Understand and reason by integrating multiple modalities (e.g., text, images, video, audio)."
MULTIMODAL_GENERATION["categories"] = ["Foundation Algorithms & Models"]
MULTIMODAL_GENERATION["papers"] = [
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
"title": "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities",
"author": "Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou",
"organization": "Alibaba Group",
"date": "20230824",
"venue": "arXiv 2023",
"pdf_url": "https://arxiv.org/pdf/2308.12966",
"code_url": "https://github.com/QwenLM/Qwen-VL",
"name": "Qwen-VL",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """Built upon the language model Qwen-7B, it makes Qwen-VL to learn image description, QA, <b>grounding</b>, and <b>text-reading</b> by three-stage training.""",
"details": 
"""
<ul>
    <li> <b>Visual encoder (1.9B):</b> ViT (Openclip's ViT-bigG).
    <li> <b>Vision-language adapter (0.08B):</b> Q-Former with 2D absolute positional encodings to produce 256 visual tokens.
    <li> <b>LLM (7.7B):</b> Qwen-7B. 
    <li> <b>Special tokens:</b> `&lt;img&gt; &lt;/img&gt;`: images; `&lt;box&gt; &lt;/box&gt;`: normalized bounding box; `&lt;ref&gt; &lt;/ref&gt;`: the content referred by bounding box.
    <li> <b>Stage 1 (pre-training):</b> large-scale, weakly labeled, web-crawled image-text pairs. 5B data, 1.4B cleaned data (77% English and 23% Chinese). Freeze LLM and optimize the vision encoder and VL adapter. Train 50K steps with batchsize of 30720, consume 1.5B samples. Image: 224x224.
    <li> <b>Stage 2 (multi-task pre-training).</b> Captioning, VQA, grounding, ref grounding, grounded captioning, OCR, pure-text autoregression. Image: 448x448. Train the whole model.
    <li> <b>Stage 3 (instruction tuning).</b> Use 350K instruction tuning data. Freeze visual encoder and optimize the LLM and adapter.
    <li> Qwen-VL can handle multi-lingual, multi-image, and multi-round conversation.
</ul>
<figure>
    <img src='Qwen-VL-fig3.png' width=500>
    <figcaption>
    <b>Figure 1.</b> Three-stage Training.
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig4.png' width=400>
    <figcaption>
    <b>Figure 2.</b> Data for training stage 1.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig5.png' width=500>
    <figcaption>
    <b>Figure 3.</b> Data for training stage 2.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig6.png' width=500>
    <figcaption>
    <b>Figure 4.</b> Performance on image captioning and VQA.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig7.png' width=500>
    <figcaption>
    <b>Figure 5.</b> Performance on text-oriented VQA.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig8.png' width=500>
    <figcaption>
    <b>Figure 6.</b> Performance on referring expression comprehension.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig9.png' width=500>
    <figcaption>
    <b>Figure 7.</b> Performance on instruction-following benchmarks.  
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig1.png' width=400>
    <figcaption>
    <b>Figure 8.</b> Qwen-VL performance.
    </figcaption>
</figure>
<figure>
    <img src='Qwen-VL-fig2.png' width=900>
    <figcaption>
    <b>Figure 9.</b> Qwen-VL capability.
    </figcaption>
</figure>
""",
},
]
