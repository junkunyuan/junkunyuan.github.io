LIST = dict()
LIST["file"] = "multimodal_understanding.html"
LIST["title"] = "Multimodal Understanding"
LIST["description"] = "Understand and reason by integrating multiple modalities (e.g., text, images, and videos)."
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
# "summary": """""",
# "details": 
# """
# <ul>
#     <li>
# </ul>
# """,
# },
{
"title": "xGen-MM (BLIP-3): A Family of Open Large Multimodal Models",
"author": "Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, Shrikant Kendre, Jieyu Zhang, Shaoyen Tseng, Gustavo A Lujan-Moreno, Matthew L Olson, Musashi Hinck, David Cobbley, Vasudev Lal, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, Ran Xu",
"organization": "Salesforce AI Research, Intel Labs, University of Washington",
"date": "20240816",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2408.08872",
"code_url": "https://github.com/salesforce/LAVIS/tree/xgen-mm",
"name": "BLIP-3",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """It improves BLIP-2 by introducing interleaved multimodal data, unified training objective, and visual resampler.""",
"details": 
"""
<ul>
    <li> <b>Training.</b> (1) Stage 1: base resolution pre-training on 100B tokens with 384x384 visual resolution; (2) Stage 2: high resolution pre-training on high-quality data; (3) Stage 3: SFT on single-image instruction-following data; (4) Stage 4: SFT on multi-image interleaved data.
</ul>
fig: fig1.png 700
cap: BLIP-3 improves BLIP-2 by introducing <b>interleaved data</b>, using <b>unified training objective</b>, and <b>fine-grained training stages</b>.
fig: fig2.png 400
cap: <b>Structure.</b> It replaces Q-Former in BLIP-2 by a <b>sampler</b> (inspired by Flamingo). Only the sampler and the LLM (Phi-3) are trained.
""",
},
{
"title": "Visual Instruction Tuning",
"author": "Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee",
"organization": "University of Wisconsin-Madison, Microsoft Research, Columbia University",
"date": "20230417",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2304.08485",
"code_url": "https://github.com/haotian-liu/LLaVA",
"name": "LLaVA",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It makes the first attempt to use GPT-4 to generate multimodal instruction-following data and performs multimodal <b>instruction fine-tuning</b>.""",
"details": 
"""
<ul>
    <li> <b>Structure.</b> (1) Vision encoder: pre-trained CLIP; (2) Connector: a linear layer; (3) Lanauge mode: Vicuna.
    <li> <b>Instruction-following data.</b> 158K: 25K conversations + 23K detailed description + 77K complex reasoning.
    <li> <b>Training.</b> (1) Stage 1: train connector on CC3M instruction-following data; (2) Stage 2: train connector & LLM on 158K instruction-following data.
</ul>
fig: fig2.png 350
cap: <b>Structure.</b>
fig: fig1.png 500
cap: Use the context to build <b>instruction-following data</b> by promting GPT.
""",
},
{
"title": "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models",
"author": "Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi",
"organization": "Salesforce Research",
"date": "20230130",
"venue": "ICML 2023",
"pdf_url": "https://arxiv.org/pdf/2301.12597",
"code_url": "https://github.com/salesforce/LAVIS/tree/main/projects/blip2",
"name": "BLIP-2",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """""",
"details": 
"""
<ul>
    <li> <b>Vision encoder.</b> ViT-L/14 from CLIP or ViT-G/14 from EVA-CLIP.
    <li> <b>Language model.</b> OPT model or FlanT5.
    <li> <b>Querying Transformer (Q-Former).</b> An image transformer & a text transformer, they are initialized from BERT and share the self-attention layer.
    <li> <b>Training.</b> (1) Stage 1 (250K steps): learn vision-langauge representations from a frozen image encoder by optimizing the three losses used in BLIP; (2) Stage 2 (80K steps): learn vision-to-language generation from a frozen LLM.
    <li> <b>Data.</b> Basically same as BLIP. Only the Q-Former is trained.
</ul>
fig: fig1.png 400
cap: <b>Overall structure</b>. Visual query embeddings are projected and prepended to the input text embeddings as Q-Former output & LLM input. 
fig: fig2.png 700
cap: <b>Q-Former</b> (left) with 32 query tokens, and <b>self-attention masking strategy</b> (right) for different training tasks.
""",
},
{
"title": "Flamingo: a Visual Language Model for Few-Shot Learning",
"author": "Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan",
"organization": "DeepMind",
"date": "20220429",
"venue": "NeurIPS 2022",
"pdf_url": "https://arxiv.org/pdf/2204.14198",
"code_url": "",
"name": "Flamingo",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It achieves <b>few-shot in-context learning</b> ability by brideging vision and language models and trains on inter-leaved visual and textual data.""",
"details": 
"""
<ul>
    <li> <b>Visual encoder.</b> Use pre-trained and frozen Normalizer-Free ResNet, and pre-train it using contrastive loss. Images and videos (sample_fps=1) are compressed to spatio-temporal grid of features.
    <li> <b>Perceiver resampler (Q-Former).</b> It processes a variable number of image or video tokens and produces a fixed number of visual tokens (64).
    <li> <b>Gated xattn-dense layers.</b> They are inserted to the pre-trained, frozen language model (Chinchilla) and are trained from scratch.
    <li> <b>Model size.</b> Flamingo-3B, Flamingo-9B, and Flamingo-80B.
</ul>
fig: fig2.png 500 fig3.png 600
cap: <b>Overall structure</b> (top) and <b>gated xattn-dense layers</b> (bottom).
""",
},
{
"title": "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation",
"author": "Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi",
"organization": "Salesforce Research",
"date": "20220128",
"venue": "ICML 2022",
"pdf_url": "https://arxiv.org/pdf/2201.12086",
"code_url": "https://github.com/salesforce/BLIP",
"name": "BLIP",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It enables both vision-language <b>understanding & generation</b> by multi-task learning with a unified framework, as well as a data bootstrapping strategy.""",
"details": 
"""
fig: fig1.png 650
cap: <b>Structure.</b> (1) Unimodal encoder is trained with an image-text contrastive (ITC) loss; (2) Image-grounded text encoder uses cross-attention layers, trained with an image-text matching (ITM) loss; (3) Image-grounded text decoder is trained with a language modeling (LM) loss.
""",
},
{
"title": "MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-Task Learning",
"author": "Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny",
"organization": "King Abdullah University of Science and Technology (KAUST), Meta AI Research",
"date": "20231014",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2310.09478v1",
"code_url": "https://github.com/Vision-CAIR/MiniGPT-4",
"name": "MiniGPT-v2",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """It makes the model learn to tackle 6 tasks with different <b>task identifiers</b> through three-stage training (maybe inspired by Qwen-VL).""",
"details": 
"""
<ul>
    <li> <b>Visual structure.</b> Use ViT-G/14 from EVA-CLIP with a Q-Former (same as MiniGPT-4). Image resolution is increased from 224x224 to 448x448, and every four neighboring visual tokens are concatenated into a single token to save compute by reducing tokens.
    <li> <b>Language structure.</b> Language model is upgraded from Vicuna to LLaMA2-chat (7B).
    <li> <b>Task identifiers</b> are used by the model to identify tasks. VQA: [vqa]; captioning: [caption]; grounded captioning: [grounding]; referring expression comprehension: [refer]; referring expression generation: [identify]; object parsing and grounding: [detection].
    <li> The <b>grounding task</b> is introduced to improve MiniGPT (maybe inspired by Qwen-VL).
</ul>
fig: fig1.png 500
cap: <b>Training data.</b>
""",
},
{
"title": "MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models",
"author": "Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny",
"organization": "King Abdullah University of Science and Technology",
"date": "20230420",
"venue": "ICLR 2024",
"pdf_url": "https://arxiv.org/pdf/2304.10592",
"code_url": "https://github.com/Vision-CAIR/MiniGPT-4",
"name": "MiniGPT-4",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It aligns a frozen visual encoder with a frozen LLM (Vicuna) using <b>one projection layer</b>.""",
"details": 
"""
<ul>
    <li> <b>Structure.</b> The same pretrained vision module as BLIP-2: ViT-G/14 from EVA-CLIP with a Q-Former. Language model: Vicuna. Connector: a single projection layer.
    <li> <b>Training.</b> Pre-training + instruction-tuning. It only fine-tunes the projection layer.
    <li> <b>Training data.</b> Pre-training: LAION, Conceptual Captions, SBU. Instruction-tuning: 3500 images from Conceptual Caption with captions generated by the pre-trained model (cleaned by ChatGPT).
</ul>
""",
},
{
"title": "Learning Transferable Visual Models From Natural Language Supervision",
"author": "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever",
"organization": "OpenAI",
"date": "20210226",
"venue": "ICML 2021",
"pdf_url": "https://arxiv.org/pdf/2103.00020",
"code_url": "https://github.com/OpenAI/CLIP",
"name": "CLIP",
"comment": "CLIP shifts computer vision research from high-quality, crowd-labeled data with pre-defined labels, e.g., ImageNet, to web-scale data with natural language supervision. CLIP generalizes well on visual benchmarks, and spurs research on multimodal foundation models. It has over 30,000 citations (as of Jul 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """By training on 400M internet text-image pairs through contrastive learning, it shows great generalization on visual benchmarks.""",
"details": 
"""
fig: fig1.png 650
cap: <b>Training and inference pipelines.</b>
fig: fig2.png 300
cap: <b>Pseudocode for training CLIP.</b>
fig: fig3.png 400
cap: <b>Zero-shot CLIP</b> outperforms few-shot probes of SoTA visual models.
fig: fig4.png 500
cap: <b>Linear probe CLIP</b> outperforms SoTA visual models.
fig: fig5.png 700
cap: CLIP is much more robust to <b>distribution shift</b>.
""",
},
{
"title": "Qwen2.5-VL Technical Report",
"author": "Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin",
"organization": "Alibaba Group",
"date": "20250219",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2502.13923",
"code_url": "https://github.com/QwenLM/Qwen2.5-VL",
"name": "Qwen2.5-VL",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It makes improvement to Qwen2-VL by employing window attention, native dynamic resolution, absoute time encoding, more and high-quality data.""",
"details": 
"""
<ul>
    <li> <b>Visual encoder.</b> The used ViT is trained from scratch. It employs self-attention + window attention to improve efficiency. It employs MRoPE as position embedding. Images and videos are sampled at native resolutions and dynamic frame rates.
    <li> <b>Vision-Language Merger.</b> Group adjacent four visual patches, concat them along feature dimensions, and project them using a two-layer MLP.
    <li> <b>Language model.</b> Qwen2.5 LLM.
    <li> <b>Pre-training stages.</b> (1) Stage 1: ViT is trained to learn visual knowledge; (2) Stage 2: all model parameters are optimized to learn diverse knowledge and tasks; (3) Stage 3: all model parameters are optimized to learn long sequences by incorpoating video and agent-based data.
    <li> <b>Post-training stages.</b> SFT and DPO are employed to optimize the language model.
    <li> <b>Sparkling capabilities.</b> Omni-document parsing, precise object grounding (based on real resolution), ultra-long video understanding and grounding, and enhanced agent functionality.
</ul>
fig: fig1.png 700
cap: <b>Model structure.</b>
fig: fig2.png 450
cap: <b>Model structure details.</b>
fig: fig3.png 500
cap: <b>Pre-training data.</b>
""",
},
{
"title": "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution",
"author": "Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, Junyang Lin",
"organization": "Alibaba Group",
"date": "20240918",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2409.12191",
"code_url": "https://github.com/QwenLM/Qwen2.5-VL",
"name": "Qwen2-VL",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It improves Qwen-VL by using a <b>naive dynamic resolution</b> mechanism with <b>multimodal RoPE</b>, and a <b>unified image-video processing</b> paradigm.""",
"details": 
"""
<ul>
    <li> <b>Visual encoder (675M).</b> Use self-developed ViT. Employ Naive Dynamic Resolution with 2D-RoPE to provide a variable number of visual tokens for images or videos with different resolution and frame number. Compress visual tokens by 2x2 using MLP.
    <li> <b>Language model (1.5B, 7.6B, 72B).</b> Qwen series.
    <li> <b>Unified image and video processing:</b> (1) Sample each video at two frames per second; (2) Compress video inputs by 4x using 3D convs; (3) Each image is treated as two identical frames for consistency. (4) Limit tokens per video are set to 16384 by adjusting the resolution.
    <li> <b>Three-stage training (same as Qwen-VL).</b> (1) Pre-training on 600B tokens by optimizing ViT; (2) Multi-task pre-raining on 600B + 800B tokens by optimizing all model parameters; (3) Instruction tuning on instructuion-following data (ChatML format) by optimizing LLM. 
    <li> <b>Three model sizes:</b> Qwen2-VL-2B (on-device), Qwen2-VL-7B (performance-optimized), Qwen2-VL-72B (most capable).
    <li> <b>Capabilities:</b> general chat, multilingual image text understanding, formula recognition, function calling, UI interation, long document understanding, code/math reasoning, video understanding, grounding, live chat, and agent potential.
</ul>
fig: fig1.png 600
cap: <b>Qwen2-VL Structure.</b> It discards the multimodal connector module used in Qwen-VL.
fig: fig2.png 600
cap: <b>Unified Multimodal Rotary Position Embedding (M-RoPE)</b> for text, images, and videos.
fig: fig3.png 500 fig4.png 500 fig5.png 500 
cap: <b>Dataset format</b> example.
""",
},
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
"info": "**",
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
    <li> <b>Capabilities:</b> multi-lingual, multi-image, and multi-round conversation.
</ul>
fig: fig3.png 500
cap: <b>Three-stage Training.</b>
fig: fig4.png 350
cap: Data for training <b>stage 1</b>.
fig: fig5.png 500
cap: Data for training <b>stage 2</b>.
""",
},
]
