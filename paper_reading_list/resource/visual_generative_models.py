VISUAL_GENERATIVE_MODELS = dict()
VISUAL_GENERATIVE_MODELS["file"] = "visual_generative_models.html"
VISUAL_GENERATIVE_MODELS["title"] = "Visual Generative Models"
VISUAL_GENERATIVE_MODELS["description"] = "Models that learn to generate visual signals, e.g., images, videos, 3D, etc."
VISUAL_GENERATIVE_MODELS["categories"] = ["Foundation Algorithms & Models", "Datasets & Evaluation"]
VISUAL_GENERATIVE_MODELS["papers"] = [
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
# """,
# },
{
"title": "Step-Video-TI2V Technical Report: A State-of-the-Art Text-Driven Image-to-Video Generation Model",
"author": "Step-Video Team",
"organization": "StepFun",
"date": "20250314",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.11251",
"code_url": "https://github.com/stepfun-ai/Step-Video-TI2V/",
"name": "Step-Video-TI2V",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """<b>StepFun</b>'s image-to-video generation model (30B), trained upon Step-Video-T2V, by incorporating conditions of motion and channel-concat image.""",
"details": 
"""
<figure>
    <img src='2025-03-14-Step-Video-TI2V-fig1.png' width=600>
    <figcaption>
        <b>Figure 1.</b> <b>Image condition:</b> channel-concat of <i>noise-augmented</i> image condition.
        <b>Motion condition:</b> optical flow-based motion + timestep. </figcaption>
</figure>
""",
},
{
"title": "Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model",
"author": "ByteDance's Seed Vision Team",
"organization": "ByteDance",
"date": "20250310",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.07703",
"code_url": "",
"name": "Seedream2.0",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """<b>ByteDance Sead Vision Team</b> 's image generation model that employs MMDiT structure and has Chinese-English bilingual capability.""",
"details": 
"""
<ul>
    <li> It uses a <i>self-developed bilingual LLM</i> and Glyph-Aligned ByT5 as text encoders.
    <li> It uses a <i>self-developed VAE</i>.
    <li> It uses learned positional embeddings on text tokens and scaled 2D RoPE on image tokens.
    <li> Training stages: pre-training => continue training => supervised fine-tuning => human feedback alignment.
    <li> Inference stages: user prompt => prompt engineering => text encoding => generation => refinement => output.
    <li> User experience platform: Doubao (豆包) and Dreamina (即梦).
</ul>
<figure>
    <img src='2025-03-10-Seedream2.0-fig1.png' width=700>
    <figcaption><b>Figure 1.</b> Performance with English and Chinese prompts.</figcaption>
</figure>
<figure>
    <img src='2025-03-10-Seedream2.0-fig2.png' width=400>
    <figcaption><b>Figure 2.</b> Pre-training data system.</figcaption>
</figure>
<figure>
    <img src='2025-03-10-Seedream2.0-fig6.png' width=600>
    <figcaption><b>Figure 3.</b> Model structure is similar to MMDiT (SD3).</figcaption>
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
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"summary": """It fine-tunes LLaVA-OneVision 7B for both <b>multimodal understanding & generation evaluation</b> by pairwise ranking & pointwise scoring.""",
"details": """"""
},
{
"title": "Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model",
"author": "Step-Video Team",
"organization": "StepFun",
"date": "20250214",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2502.10248",
"code_url": "https://github.com/stepfun-ai/Step-Video-T2V/",
"name": "Step-Video-T2V",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """<b>StepFun</b>'s open-sourced model (30B) with DiT structure for text-to-video generation.""",
"details": 
"""
<figure>
    <img src='2025-02-14-Step-Video-T2V-fig1.png' width=500>
    <figcaption><b>Figure 1.</b> <b>Main structure.</b> a VAE with a 8x8x4 compression ratio and 16 channels, bilingual text encoders (HunyuanCLIP and Step-LLM), DiT with RoPE-3D and QK-Norm, and a DPO pipeline. Text prompt conditions are incorporated into DiT by cross-attention modules.</figcaption>
</figure>
<figure>
    <img src='2025-02-14-Step-Video-T2V-fig2.png' width=400>
    <figcaption><b>Figure 2.</b> <b>VAE</b> compresses videos by 16x16x8 with 16 feature channels.</figcaption>
</figure>
<figure>
    <img src='2025-02-14-Step-Video-T2V-fig4.png' width=500>
    <figcaption><b>Figure 3.</b> <b>DPO framework.</b> use training data prompts and handcrafted prompts to generate samples, which are scored through human annotation or reward models. Diffusion-DPO method is adapted here by reducing beta and increasing learning rate for achieving faster convergence.</figcaption>
</figure>
<figure>
    <img src='2025-02-14-Step-Video-T2V-fig5.png' width=900>
    <figcaption><b>Figure 4.</b> Using <b>2B video-text pairs</b>, <b>3.8B image-text pairs</b>. <i>Filters:</i> video segmentation, video quality assessment, aesthetic score, NSFW score, watermark detection, subtitle detection, saturation score, blur score, black border detection, video motion assessment, K-means-based concept balancing, and CLIP score alignment. <i>Video captioning:</i> short caption, dense caption, and original title.</figcaption>
</figure>
<figure>
    <img src='2025-02-14-Step-Video-T2V-fig7.png' width=600>
    <figcaption><b>Figure 5.</b> Pre-training stages. </figcaption>
</figure>
""",
},
{
"title": "Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis",
"author": "Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu",
"organization": "ByteDance",
"date": "20241205",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2412.04431",
"code_url": "https://github.com/FoundationVision/Infinity/",
"name": "Infinity",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It improves VAR by applying <b>bitwise modeling</b> that makes vocabulary "infinity" to open up new posibilities of discrete text-to-image generation.""",
"details": 
"""
<figure>
    <img src='2024-12-05-Infinity-fig1.png' width=500>
    <figcaption><b>Figure 1.</b> <b>Viusal tokenization and quantization:</b> instead of predicting <i>2**d</i> indices, infinite-vocabulary classifier predicts <i>d</i> bits instead.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig3.png' width=400>
    <figcaption><b>Figure 2.</b> <b>Infinity:</b> it is fast and better.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig4.png' width=300>
    <figcaption><b>Figure 3.</b> <b>Tokenizer:</b> it outperforms continuous SD VAE.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig5.png' width=500>
    <figcaption><b>Figure 4.</b> <b>Inifinite-Vocabulary Classifier:</b> it needs low memory but performs better.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig8.png' width=400>
    <figcaption><b>Figure 5.</b> <b>Self-correction</b> mitigates the train-test discrepancy.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig6.png' width=700>
    <figcaption><b>Figure 6.</b> <b>Scaling up vocabulary:</b> vocabulary size and model size scale well.</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig7.png' width=750>
    <figcaption><b>Figure 7.</b> <b>Scaling up model size:</b> there is strong correlation between validation loss and evaluation metrics (as observed by Fluid).</figcaption>
</figure>
<figure>
    <img src='2024-12-05-Infinity-fig9.png' width=900>
    <figcaption><b>Figure 8.</b> Using <b>2D RoPE</b> outperforms using APE.</figcaption>
</figure>
""",
},
{
"title": "HunyuanVideo: A Systematic Framework For Large Video Generative Models",
"author": "Hunyuan Multimodal Generation Team",
"organization": "Tencent",
"date": "20241203",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2412.03603",
"code_url": "https://github.com/Tencent/HunyuanVideo/",
"name": "HunyuanVideo",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """<b>Tencent Hunyuan Team</b>'s open-sourced text-to-video and image-to-video generation model (13B) with diffusion transformer (FLUX structure).""",
"details": 
"""
""",
},
{
"title": "Movie Gen: A Cast of Media Foundation Models",
"author": "Movie Gen Team",
"organization": "Meta",
"date": "20241017",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2410.13720",
"code_url": "",
"name": "Movie Gen",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """<b>Meta Movie Gen Team</b>'s diffusion transformer-based model (30B) for 16s / 1080p / 16fps video and synchronized audio generation.""",
"details": 
"""
""",
},
{
"title": "Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens",
"author": "Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, Yonglong Tian",
"organization": "Google DeepMind, MIT",
"date": "20241017",
"venue": "ICLR 2025",
"pdf_url": "https://arxiv.org/pdf/2410.13863",
"code_url": "",
"name": "Fluid",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It shows auto-regressive models with <b>continuous tokens beat discrete tokens counterpart</b>, and finds some empirical observations during scaling.""",
"details": 
"""
<figure>
    <img src='2024-10-17-Fluid-fig1.png' width=500>
    <figcaption>
        <b>Figure 1.</b> <b>Image tokenizer:</b> discrete (VQGAN) or continuous (VAE). <b>Text tokenizer:</b> discrete (T5-XXL).
        <b>Model structure:</b> transformer with cross-attention modules attending to text embeddings.
        <b>Loss:</b> cross-entropy loss on text tokens and diffusion loss on image tokens.
    </figcaption>
</figure>
<figure>
    <img src='2024-10-17-Fluid-fig2.png' width=700>
    <figcaption><b>Figure 2.</b> Scaling behavior of validation loss on <b>model size</b>.</figcaption>
</figure>
<figure>
    <img src='2024-10-17-Fluid-fig3.png' width=700>
    <figcaption><b>Figure 3.</b> <b>Random-order masks</b> on <b>continuous image tokens</b> perform the best. Continuous prefers random order, discrete prefers raster order. </figcaption>
</figure>
<figure>
    <img src='2024-10-17-Fluid-fig4.png' width=700>
    <figcaption><b>Figure 4.</b> Random-order masks on continuous tokens scale with <b>training computes</b>.</figcaption>
</figure>
<figure>
    <img src='2024-10-17-Fluid-fig5.png' width=700>
    <figcaption><b>Figure 5.</b> Strong correlation between <b>validation loss</b> and <b>evaluation metrics</b>.</figcaption>
</figure>
"""
},
{
"title": "Scaling Diffusion Transformers to 16 Billion Parameters",
"author": "Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Junshi Huang",
"organization": "Kunlun Inc.",
"date": "20240716",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2407.11633",
"code_url": "https://github.com/feizc/DiT-MoE/",
"name": "DiT-MoE",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It proposes diffusion transformer (16B) with <b>Mixture-of-Experts</b> by inserting experts into DiT blocks for image generation.""",
"details": 
"""
<ul>
    <li> Incorporating <i>shared expert routing</i> improves convergence and performance, but the improvement is little when using more than one.
    <li> Increasing experts reduces loss but introduces more loss spikes.
</ul>
<figure>
    <img src='2024-07-16-DiT-MoE-fig1.png' width=700>
    <figcaption><b>Figure 1.</b> It is built upon DiT and replaces MLP within Transformer blocks by sparsely activated mixture of MLPs as experts.</figcaption>
</figure>
""",
},
{
"title": "Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation",
"author": "Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, Zehuan Yuan",
"organization": "The University of Hong Kong, ByteDance",
"date": "20240610",
"venue": "arXiv 2024",
"pdf_url": "https://arxiv.org/pdf/2406.06525",
"code_url": "https://github.com/FoundationVision/LlamaGen/",
"name": "LlamaGen",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It shows that applying "next-token prediction" to <b>vanilla autoregressive language models</b> can achieve good  image generation performance.""",
"details": 
"""
<ul>
    <li> It trains a discrete visual tokenizer that is competitive to the continuous ones, e.g., SD VAE, SDXL VAE, and Consistency Decoder from OpenAI.
    <li> Vanilla autoregressive models, e.g., LlaMA, without inductive biases on visual signals can serve as the basis of image generation system.
    <li> It is trained on 50M subset of LAION-COCO and 10M internal high aesthetics quality images.
</ul>
""",
},
{
"title": "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction",
"author": "Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang",
"organization": "Peking University, Bytedance",
"date": "20240403",
"venue": "NeurIPS 2024",
"pdf_url": "https://arxiv.org/pdf/2404.02905",
"code_url": "https://github.com/FoundationVision/VAR/",
"name": "VAR",
"comment": "NeurIPS 2024 best paper award.",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It improves <u>auto-regressive</u> image generation on image quality, inference speed, data efficiency, and scalability, by proposing <b>next-scale prediction</b>.""",
"details": 
"""
<figure>
    <img src='2024-04-03-VAR-fig1.png' width=700>
    <figcaption> <b>Figure 1.</b> <b>Next-scale prediction:</b> start from 1x1 token map; at each step, it predicts the next higher-resolution token map given all previous ones.</figcaption>
</figure>
<figure>
    <img src='2024-04-03-VAR-fig3.png' width=800>
    <figcaption><b>Figure 2.</b> <b>Training pipeline of tokenzier and VAR.</b>  Tokenzier (similar to VQ-VAE): the same architecture and training data (OpenImages), using codebook of 4096 and spatial downsample ratio of 16. VAR: the standard transformer with AdaLN; not use RoPE, SwiGLU MLP, RMS Norm. </figcaption>
</figure>
<figure>
    <img src='2024-04-03-VAR-fig4.png' width=700>
    <figcaption><b>Figure 3.</b> <b>Algorithms of tokenizer:</b> encoding and reconstruction.</figcaption>
</figure>
<figure>
    <img src='2024-04-03-VAR-fig2.png' width=350>
    <figcaption><b>Figure 4.</b> VAR shows good <b>scaling behavior</b>, and significantly outperforms DiT.</figcaption>
</figure>
""",
},
{
"title": "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis",
"author": "Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach",
"organization": "Stability AI",
"date": "20230704",
"venue": "ICLR 2024",
"pdf_url": "https://arxiv.org/pdf/2307.01952",
"code_url": "https://github.com/Stability-AI/generative-models/",
"name": "SDXL",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It improves older SD by employing <b>larger UNet backbone</b>, <b>resolution conditions</b>, <b>two text encoders</b>, and a <b>refinement model</b>.""",
"details": 
"""
<p><b>Architecture of SDXL:</b>.<br>
(1) It has 2.6B parameters with different transformer blocks, SD 1.4/1.5/2.0/2.1 has about 860M parameters.<br>
(2) It uses two text encoders: OpenCLIP ViT-bigG & CLIP ViT-L.<br>
(3) The embeddings of height & width and cropping top & left and bucketing heigh & width are added to timestep embeddings as conditions.<br>
(4) It improves VAE by employing EMA and a larger batchsize of 256.<br>
(5) It employs a refinement model of SDEdit to refine visual details.</p>
<p><b>Training stages:</b> (1) reso=256x256, steps=600,000, batchsize=2048; (2) reso=512x512, steps=200,000; (3) mixed resolution and aspect ratio training.</p>
<figure>
    <img src='2023-07-04-SDXL-fig1.png' width=600>
</figure>
""",
},
{
"title": "Scalable Diffusion Models with Transformers",
"author": "William Peebles, Saining Xie",
"organization": "UC Berkeley, New York University",
"date": "20221219",
"venue": "ICCV 2023",
"pdf_url": "https://arxiv.org/pdf/2212.09748",
"code_url": "https://github.com/facebookresearch/DiT/",
"name": "DiT",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "visual_generative_models-dit.ipynb",
"summary": """It replaces the conventional U-Net structure with <b>transformer</b> for scalable image generation, the timestep and condition are injected by adaLN-Zero""",
"details": 
"""
<figure>
    <img src='2022-12-19-DiT-fig1.png' width=800>
    <figcaption><b>Figure 1.</b> Using adaLN-Zero structure to inject timestep and class condition performs better than using cross-attention or in-context.</figcaption>
</figure>
""",
},
{
"title": "CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers",
"author": "Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, Jie Tang",
"organization": "Tsinghua University, BAAI",
"date": "20220529",
"venue": "ICLR 2023",
"pdf_url": "https://arxiv.org/pdf/2205.15868",
"code_url": "https://github.com/THUDM/CogVideo/",
"name": "CogVideo",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It proposes a transformer-based video generation model (9B) that performs <b>auto-regressive</b> frame  generation and recursive frame interpolatation""",
"details": 
"""
<figure>
    <img src='2022-05-29-cogvideo-fig1.png' width=500>
    <figcaption><b>Figure 1.</b> CogVideo is trained upon CogView2. It generates frames auto-regressively and interpolates them recursively.</figcaption>
</figure>
""",
},
{
"title": "High-Resolution Image Synthesis with Latent Diffusion Models",
"author": "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer",
"organization": "Heidelberg University, Runway ML",
"date": "20211220",
"venue": "CVPR 2022",
"pdf_url": "https://arxiv.org/pdf/2112.10752",
"code_url": "https://github.com/CompVis/latent-diffusion/",
"name": "LDM",
"comment": "It makes high-resolution image synthesis efficiently by performing generation in a compressed VAE latent space. It has over 20,000 citations (as of Jun 29, 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It achieves efficient high-resolution image generation by applying diffusion and denoising processes in the <b>compressed VAE latent space</b>.""",
"details": 
"""
<figure>
    <img src='2021-12-20-ldm-fig1.png' width=400>
    <figcaption><b>Figure 1.</b> Diffusion and denoising processes are conducted in the compressed VAE latent space. The conditions are injected by cross-attention.</figcaption>
</figure>
""",
},
{
"title": "Classifier-Free Diffusion Guidance",
"author": "Jonathan Ho, Tim Salimans",
"organization": "Google Research, Brain team",
"date": "20211208",
"venue": "NeurIPS workshop 2021",
"pdf_url": "https://arxiv.org/pdf/2207.12598",
"code_url": "",
"name": "CFG",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "visual_generative_models-cfg.ipynb",
"summary": """It improves conditional image generation with <b>classifier-free condition guidance</b> by jointly training a conditional model and an unconditional model.""",
"details": "",
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
"comment": "It achieves high-quality image synthesis through iterative denoising diffusion processes. It has over 20,000 citations (as of Jun 29 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "visual_generative_models-ddpm.ipynb",
"summary": """It proposes <b>denoising diffusion probabilistic models</b> that iteratively denoises data from random noise.""",
"details": 
"""
<figure>
    <img src='2020-06-19-ddpm-fig1.png' width=400>
    <figcaption><b>Figure 1.</b> Diffusion (forward) and denoising (reverse) processes of DDPM.</figcaption>
</figure>
<figure>
    <img src='2020-06-19-ddpm-fig2.png' width=600>
    <figcaption><b>Figure 2.</b> Training and sampling algorithms of DDPM.</figcaption>
</figure>
""",
},
{
"title": "Generating Diverse High-Fidelity Images with VQ-VAE-2",
"author": "Ali Razavi, Aaron van den Oord, Oriol Vinyals",
"organization": "DeepMind",
"date": "20190602",
"venue": "NeurIPS 2019",
"pdf_url": "https://arxiv.org/pdf/1906.00446",
"code_url": "",
"name": "VQ-VAE-2",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """In order to generate large scale images efficiently, it improves VQ-VAE by employing a <b>hierarchical organization</b>.""",
"details": 
"""
<ul>
    <li><b>Structure:</b> (1) a top-level encoder to learn top-level priors from images; (2) a bottom-level encoder to learn bottom-level priors from images and top-level priors; (3) a decoder to generate images from both top-level and bottom-level priors.</li>
    <li><b>Training stage 1:</b> training the top-level encoder and the bottom-level encoder to encode images onto the two levels of discrete latent space.</li>
    <li><b>Training stage 2:</b> training PixelCNN to predict bottom-level priors from top-level priors, while fixing the two encoders.</li>
    <li><b>Sampling:</b> (1) sampling a top-level prior; (2) predicting bottom-level prior from the top-level prior using the trained PixelCNN; (3) generating images from both the top-level and the bottom-level priors by the trained decoder.</li>
</ul>
<figure>
    <img src='2019-06-02-VQ-VAE-2-fig1.png' width=900>
    <figcaption><b>Figure 1.</b> Training and sampling frameworks.</figcaption>
</figure>
<figure>
    <img src='2019-06-02-VQ-VAE-2-fig2.png' width=550>
    <figcaption><b>Figure 2.</b> Training and sampling algorithms.</figcaption>
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
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"summary": """It proposes <b>vector quantised variational autoencoder</b> to generate discrete codes while the prior is also learned.""",
"details": 
"""
<ul>
    <li><b>Posterior collapse problem:</b> a strong decoder and a strong KL constraint could make the learned posterior <i>q(z|x)</i> very close to prior <i>p(z)</i>, so that the conditional generation task collapses to an unconditional generation task.</li>
    <li><b>How VQ-VAE avoids the collapse problem by employing discrete codes/latents?</b> (1) It learns <i>q(z|x)</i> by choosing one from some candidates rather than directly generating a simple prior; (2) The learned <i>q(z|x)</i> is continuous but <i>p(z)</i> is discrete, so the encoder can not be "lazy".</li>
    <li><b>Optimization objectives:</b> (1) The decoder is optimized by a recontruction loss; (2) The encoder is optimized by a reconstruction loss and a matching loss; (3) The embedding is optimized by a matching loss.</li>
    <li><b>How to back-propagate gradient with quantization exists? Straight-Through Estimator:</b> directly let the graident of loss to the quantized embedding equal to the gradient of loss to the embedding that before being quantized.</li>
</ul>
<figure>
    <img src='2017-11-02-VQ-VAE-fig1.png' width=900>
</figure>
""",
},
]
