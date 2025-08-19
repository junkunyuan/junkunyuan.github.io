VISUAL_GENERATION = dict()
VISUAL_GENERATION["file"] = "visual_generation.html"
VISUAL_GENERATION["title"] = "Visual Generation"
VISUAL_GENERATION["description"] = "Generate visual signals (e.g., images and videos)."
VISUAL_GENERATION["categories"] = ["Foundation Algorithms & Models", "Reinforcement Learning", "Inference-Time Improvement", "Acceleration", "Datasets & Evaluation", "Editing & Inpainting & Outpainting", "Human", "Style"]
VISUAL_GENERATION["papers"] = [
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
# <ul>
#     <li>
# </ul>
# """,
# },
{
"title": "DuoLoRA : Cycle-consistent and Rank-disentangled Content-Style Personalization",
"author": "Aniket Roy, Shubhankar Borse, Shreya Kadambi, Debasmit Das, Shweta Mahajan, Risheek Garrepalli, Hyojin Park, Ankita Nayak, Rama Chellappa, Munawar Hayat, Fatih Porikli",
"organization": "Johns Hopkins University, Qualcomm AI Research",
"date": "20250415",
"venue": "ICCV 2025",
"pdf_url": "https://arxiv.org/pdf/2504.13206",
"code_url": "",
"name": "DuoLoRA",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It disentangles content and style in diffusion LoRA merging by rank-dimension masking, layer priors, and cycle-consistency loss.
""",
"details": 
"""
<ul>
    <li>
</ul>
""",
},
{
"title": "StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer",
"author": "Ruojun Xu, Weijie Xi, Xiaodi Wang, Yongbo Mao, Zach Cheng",
"organization": "Zhejiang University, Dcar",
"date": "20250120",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2501.11319",
"code_url": "https://github.com/bytedance/StyleSSP",
"name": "StyleSSP",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It employs DDIM reversion of content images with low frequency component removed to obtain a <b>better initial noise</b> for training-free style transfer.
""",
"details": 
"""
""",
},
{
"title": "SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer",
"author": "Hongda Liu, Longguang Wang, Ye Zhang, Ziru Yu, Yulan Guo",
"organization": "Sun Yat-Sen University, Aviation University of Air Force",
"date": "20250320",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2503.15934",
"code_url": "https://github.com/Chernobyllight/SaMam",
"name": "SaMam",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
Introduce <b>Mamba model</b> into style transfer to improve efficiency.
""",
"details": 
"""
""",
},
{
"title": "Style-Editor: Text-driven Object-centric Style Editing",
"author": "Jihun Park, Jongmin Gim, Kyoungmin Lee, Seunghun Lee, Sunghoon Im",
"organization": "Republic of Korea",
"date": "20240816",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2408.08461",
"code_url": "",
"name": "Style-Editor",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It achieves <b>object-level text-driven style editing</b> by identifying an object patches using CLIP and editing its style while preserving the background.
""",
"details": 
"""
""",
},
{
"title": "StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements",
"author": "Mingkun Lei, Xue Song, Beier Zhu, Hao Wang, Chi Zhang",
"organization": "Westlake University, Fudan University, Nanyang Technological University, The Hong Kong University of Science and Technology (Guangzhou)",
"date": "20241211",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2412.08503",
"code_url": "https://github.com/Westlake-AGI-Lab/StyleStudio",
"name": "StyleStudio",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It improves text-driven style transfer by <b>cross-modal AdaIN</b>, teacher model guidance, and <b>style CFG</b> with negative style images.
""",
"details": 
"""
""",
},
{
"title": "K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs",
"author": "Ziheng Ouyang, Zhen Li, Qibin Hou",
"organization": "Nankai University",
"date": "20250225",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2502.18461",
"code_url": "https://github.com/HVision-NKU/K-LoRA",
"name": "K-LoRA",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It introduces <b>training-free LoRA fusion</b> that compares Top-K elements in LoRAs to be fused and determines which LoRA to select for optimal fusion.
""",
"details": 
"""
""",
},
{
"title": "V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents",
"author": "Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, Yali Wang",
"organization": "Shanghai Jiao Tong University, Shanghai AI Laboratory, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, University of Chinese Academy of Sciences",
"date": "20250315",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2503.12077",
"code_url": "https://github.com/ZhengrongYue/V-Stylist",
"name": "V-Stylist",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It enables video stylization with <b>open style prompt</b> by searching a style tree and asigning the obtained weights to ControlNets.
""",
"details": 
"""
<ul>
    <li> <b>Video parser</b> splits the input video into shots and generates their text prompts.
    <li> <b>Style parser</b> search the matched style ControlNet model combination from a style tree.
    <li> <b>Style artist</b> renders the video shots by applying the matched style ControlNet models.
</ul>
""",
},
{
"title": "OmniStyle: Filtering High Quality Style Transfer Data at Scale",
"author": "Ye Wang, Ruiqi Liu, Jiang Lin, Fei Liu, Zili Yi, Yilin Wang, Rui Ma",
"organization": "Jilin University, Nanjing University, ByteDance, Adobe, Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China",
"date": "20250520",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2505.14028",
"code_url": "https://github.com/wangyePHD/OmniStyle",
"name": "OmniStyle",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It introduces a dataset consisting of <b>150K</b> content-style-stylized image triplets across <b>1,000 styles</b> with textual discriptions and instruction prompts.
""",
"details": 
"""
""",
},
{
"title": "StyleMaster: Stylize Your Video with Artistic Generation and Translation",
"author": "Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, Wenhan Luo",
"organization": "Hong Kong University of Science and Technology, KuaiShou Technology",
"date": "20241210",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2412.07744",
"code_url": "https://github.com/KwaiVGI/StyleMaster",
"name": "StyleMaster",
"comment": "",
"category": "Style",
"jupyter_notes": "",
"info": "",
"summary": 
"""
""",
"details": 
"""
fig: fig1.png 900
cap: <b>Structure.</b> (1) <b>CLIP</b> extracts patch features and embeddings. (2) <b>Patch features</b> with less similarity with text prompt are employed to remove content infomation. (3) <b>Global projection</b>, trained by contrastive data, extracts global style from <b>CLIP embeddings</b>. (4) <b>Local and global features</b> are injected into model by cross-attention. (5) <b>Gray tile ControlNet</b> injects gray information while removing style information.
""",
},
{
"title": "Yan: Foundational Interactive Video Generation",
"author": "Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, Wei Yang, Wenkai Lv, Yangbin Yu, Yewen Wang, Yonghang Guan, Zhihao Hu, Zhongbin Fang, Zhongqian Sun",
"organization": "Tencent",
"date": "20250812",
"venue": "arXiv 2025",
"pdf_url": "https://www.arxiv.org/pdf/2508.08601",
"code_url": "",
"name": "Yan",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": 
"""
""",
"details": 
"""
<ul>
    <li> <b>AAAI-level simulation (Yan-Sim).</b> Design a highly-compressed, low-latency 3D-VAE (32x32x2-16chan with a light decoder) coupled with a KV-cache-based shift-window denoising inference process, achieving real-time 1080P/60FPS interative simulation.
    <li> <b>Multi-modal generation (Yan-Gen).</b> Use autoregressive caption method to inject game-specific knowledge into open-domain, multimodal, interative video diffusion models.
    <li> <b>Multi-granularity editing (Yan-Edit).</b> Disentangle interative mechanics simulation from visual rendering, enabling mutli-granularity video content editing during interaction through text.
    <li><b>Data pipeline.</b> Use agent to collect and clean data (action & image pair) in the game environment of a renowned modern 3D game (Yuanmeng Star). Use VLM and depth model to obtain prompt and depth. Both labeled and unlabled data are used for training.
</ul>
fig: fig1.png 500
cap: <b>Dataset comparisons.</b>
fig: fig2.png 550
cap: <b>Yan-Sim structure.</b>
fig: fig3.png 550
cap: <b>Yan-Gen structure.</b>
fig: fig4.png 550
cap: <b>Yan-Edit structure.</b>
""",
},
{
"title": "Flow Matching for Generative Modeling",
"author": "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le",
"organization": "Meta AI (FAIR), Weizmann Institute of Science",
"date": "20221006",
"venue": "ICLR 2023",
"pdf_url": "https://arxiv.org/pdf/2210.02747",
"code_url": "",
"name": "Flow Matching",
"comment": "It proposes a simple yet powerful framework for training continuous-time generative models by directly learning the vector field of an optimal transport path between data and noise, enabling fast training, stable optimization, and high sample quality. It has over 1,800 citations (as of Aug 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "FlowMatching.ipynb",
"info": "**",
"summary": 
"""
It trains <b>continuous normalizing flows</b> using conditional probability paths, resulting in fast training, high sample quality, and efficient sampling.
""",
"details": 
"""
""",
},
{
"title": "Genie: Generative Interactive Environments",
"author": "Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, Tim Rocktäschel",
"organization": "Google DeepMind, University of British Columbia",
"date": "20240223",
"venue": "",
"pdf_url": "ICML 2024",
"code_url": "",
"name": "Genie",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It proposes a foundation world model (11B), comprising of a <b>video tokenizer</b>, an <b>autoregressive dynamics world</b>, and a <b>latent action model</b>. It is trained on 200K hours of Internet <b>gaming videos</b> without action or text labels, is controllable on frame-by-frame via a learned latent action space.
""",
"details": 
"""
<ul>
    <li> <b>Data.</b> It is trained on 200K hours of Internet gaming videos without action or text annotations.
    <li> <b>Training pipeline.</b> (1) Train the video tokenizer. (2) Co-train the latent action model and the dynamics model.
</ul>
fig: fig1.png 550
cap: <b>Structure.</b>
fig: fig2.png 350
cap: <b>Latent action model</b> infers the latent action between each pair of frames. It is a VQ-VAE with discrete set of codes equal the possible actions. The <b>encoder</b> takes an previous frames and the next frame and outputs latent actions. The <b>decoder</b> takes all previous frames and latent actions and predicts the next frame. <b>Inference:</b> the entire LAM is discarded apart from the VQ codebook, and is replaced with user actions.
fig: fig3.png 350
cap: <b>Video tokenizer</b> converts each frame of raw video into discrete tokens.
fig: fig4.png 200
cap: <b>Dynamics model</b> takes latent action and past frame tokens and predicts the next frame. It is a decoder-only MaskGIT transformer.
""",
},
{
"title": "Qwen-Image Technical Report",
"author": "Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, Zenan Liu",
"organization": "Qwen Team",
"date": "20250804",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2508.02324",
"code_url": "https://github.com/QwenLM/Qwen-Image",
"name": "Qwen-Image",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """<b>Alibaba Qwen Team</b>'s image generation foundation model, excelling in <b>complex text rendering</b> and <b>precise image editing</b>.""",
"details": 
"""
<ul>
    <li> <b>Structure.</b> It employs the standard <b>MMDiT</b> structure with <b>Qwen2.5 VL</b> as the text encoder.
    <li> <b>VAE.</b> It finetunes an <b>image decoder</b> and a <b>video decoder</b> upon <b>Wan-2.1-VAE</b> by optimizing only a reconstruction loss and a perceptual loss.
    <li> <b>Positional embedding.</b> It introduces Multimodal Scalable RoPE (MSRoPE), a <b>diagonal</b> position encoding.
    <li> <b>Data balance.</b> Nature: 55%. Design: 27%. People: 13%. Synthetic: 5%.
    <li> <b>Data filtering.</b> <b>Stage 1.</b> Initial pre-training. 256p. Broken files + resolution + deduplication + NSFW. <b>Stage 2.</b> Quality improvement. Rotation + brightness + saturation + entropy + texture. <b>Stage 3.</b> Alignment improvement. Chinese CLIP + SigLIP 2 + token length. <b>Stage 4.</b> Text-rendering enhancement. Intense filter + small character filter. <b>Stage 5.</b> High-resolution refinement. 640p. Image quality + resolution + aesthetic + abnormal element. <b>Stage 6.</b> Category balance and portrait augmentation. <b>Stage 7.</b> Balanced multi-scale training. 640p and 1328p.
    <li> <b>Data synthesis.</b> (1) Pure rendering in simple background. (2) Compositional rendering in contextual scenes. (3) Complex rendering in structured frames.
    <li> <b>RL.</b> DPO + GRPO.
    <li> <b>Editing.</b> Channel concatenation of the original image and the edited image. 
</ul>
fig: fig2.png 550 fig1.png 350
cap: <b>Structure and configuration.</b>
""",
},
{
"title": "D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples",
"author": "Zijing Hu, Fengda Zhang, Kun Kuang",
"organization": "Zhejiang University, Nanyang Technological University",
"date": "20250528",
"venue": "ICML 2025",
"pdf_url": "https://arxiv.org/pdf/2505.22002",
"code_url": "https://github.com/hu-zijing/D-Fusion",
"name": "D-Fusion",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It constructs new winning samples that are <b>visually consistent</b> with losing samples while aligning with winning samples in <b>prompt-following</b>.""",
"details": 
"""
<ul>
    <li> <b>Method details.</b> (1) It extracts a mask by <b>averaging cross-attention maps</b> across all heads and blocks in the first up-sampling layer. (2) It <b>binarizes the mask</b> by applying a fixed threshold, generating segmentation of attention, i.e., the importance map for the prompt. (3) It <b>fuses the self-attention key and value</b> by applying the segmentation mask to winning samples and losing samples.
</ul>
""",
},
{
"title": "Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation",
"author": "Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, Qifeng Chen",
"organization": "",
"date": "20240604",
"venue": "SIGGRAPH-Asia 2024",
"pdf_url": "https://arxiv.org/pdf/2406.01900",
"code_url": "https://github.com/mayuelala/FollowYourEmoji",
"name": "Follow-Your-Emoji",
"comment": "",
"category": "Human",
"jupyter_notes": "",
"info": "",
"summary": """""",
"details": 
"""
<ul>
    <li>
</ul>
""",
},
{
"title": "Follow-Your-Canvas: Higher-Resolution Video Outpainting with Extensive Content Generation",
"author": "Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen, Wei Liu",
"organization": "Tencent, HKUST, USTC, Tsinghua University",
"date": "20240902",
"venue": "AAAI 2025",
"pdf_url": "https://arxiv.org/pdf/2409.01055",
"code_url": "https://github.com/mayuelala/FollowYourCanvas",
"name": "Follow-Your-Canvas",
"comment": "",
"category": "Editing & Inpainting & Outpainting",
"jupyter_notes": "",
"info": "",
"summary": """It enables <b>higher-resolution</b> video outpainting with <b>extensive content</b> generation through sliding window and source video layout injection.""",
"details": 
"""
fig: fig1.png 600
cap: <b>Training framework.</b> An anchor window and a target window are randomly sampled, mimicking the "source video" and "region to perform outpaint" for inference. The anchor window is injected into the model through a layout encoder, as well as a relative region embedding calculated by the positional relation between the anchor window and the target window, aligning generated layout of the target window with the anchor window.
fig: fig2.png 600
cap: <b>Inference framework.</b> The given source video is covered by N spatial windows. During each denoising step t, outpainting is performed within each window in parallel on separate GPUs to accelerate inference. The windows are then merged through Gaussian weights to get outcome at step t-1. These windows may cover layer upon layer, allowing outpainting any videos to a higher resolution without being limited by GPU memory.
""",
},
{
"title": "DanceGRPO: Unleashing GRPO on Visual Generation",
"author": "Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, Ping Luo",
"organization": "ByteDance Seed, The University of Hong Kong",
"date": "20250512",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2505.07818",
"code_url": "https://github.com/XueZeyue/DanceGRPO",
"name": "DanceGRPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It reformulates ODE sampling to <b>SDE</b>, and adapts <b>GRPO</b> to visual generation, validating on different models, tasks, and reward models.""",
"details": 
"""
<ul>
    <li> It <b>omits the KL regularization</b> of GRPO because it brings little benefits.
    <li> It <b>assigns shared initialization noise</b> to samples from the same prompt to solve reward hacking and training instability.
</ul>
fig: fig1.png 550
cap: <b>Algorithm.</b>
fig: fig2.png 600
cap: Left: Sampling top k and bottom k with <b>best-of-N</b> improves results. Right: Training with some <b>timestep subsets</b> is potential.
""",
},
{
"title": "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion",
"author": "Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, Qiang Xu ",
"organization": "Tencent PCG ARC Lab, The Chinese University of Hong Kong",
"date": "20240311",
"venue": "ECCV 2024",
"pdf_url": "https://arxiv.org/pdf/2403.06976",
"code_url": "https://github.com/TencentARC/BrushNet",
"name": "BrushNet",
"comment": "",
"category": "Editing & Inpainting & Outpainting",
"jupyter_notes": "",
"info": "**",
"summary": """It introduces a <b>plug-and-play</b> dual-branch model and a segmentation-based inpainting training dataset <b>BrushData</b> and a benchmark <b>BrushBench</b>.""",
"details": 
"""
fig: fig1.png 550
cap: <b>Comparisons.</b> (a) lacks knowledge of mask boundaries. (b) struggles to obtain pure masked image features due to the text's influence. 
fig: fig2.png 550
cap: <b>Structure.</b> (1) Use VAE to process masked image to preserve original details. (2) Masked image, mask, noisy masked image are concatenated as the input. (3) Adopt hierarchical approach to incorporate UNet features. (4) Remove text cross-attention. (5) Use blurred blending.
""",
},
{
"title": "Training Diffusion Models with Reinforcement Learning",
"author": "Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine",
"organization": "University of California, Berkeley, Massachusetts Institute of Technology",
"date": "20230522",
"venue": "ICLR 2024",
"pdf_url": "https://arxiv.org/pdf/2305.13301",
"code_url": "https://github.com/jannerm/ddpo",
"name": "DDPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "**",
"summary": """It applies policy gradient to diffusion models, the reward is estimated from a VLM, to improve its aesthetics.""",
"details": 
"""
""",
},
{
"title": "Preference Alignment on Diffusion Model: A Comprehensive Survey for Image Generation and Editing",
"author": "Sihao Wu, Xiaonan Si, Chi Xing, Jianhong Wang, Gaojie Jin, Guangliang Cheng, Lijun Zhang, Xiaowei Huang",
"organization": "University of Liverpool, Institute of Software Chinese Academy of Sciences, University of Edinburgh, University of Bristol, University of Exeter",
"date": "20250210",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2502.07829",
"code_url": "",
"name": "Survey on Preference Alignment of Image Generation",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """A survey on preference alignment of image generation and editing.""",
"details": 
"""
fig: fig1.png 700
cap: <b>Preference Alignment methods for Image Generation.</b>
""",
},
{
"title": "PrefPaint: Aligning Image Inpainting Diffusion Model with Human Preference",
"author": "Kendong Liu, Zhiyu Zhu, Chuanhao Li, Hui Liu, Huanqiang Zeng, Junhui Hou",
"organization": "City University of Hong Kong, Yale University, Saint Francis University, Huaqiao University",
"date": "20241029",
"venue": "NeurIPS 2024",
"pdf_url": "https://arxiv.org/pdf/2410.21966",
"code_url": "https://github.com/Kenkenzaii/PrefPaint",
"name": "PrefPaint",
"comment": "",
"category": "Reinforcement Learning, Editing & Inpainting & Outpainting",
"jupyter_notes": "",
"info": "",
"summary": """It trains a reward model on 51K images with human preferences, and uses it to perform reinforcement learning of diffusion models.""",
"details": 
"""
""",
},
{
"title": "Calibrated Multi-Preference Optimization for Aligning Diffusion Models",
"author": "Kyungmin Lee, Xiahong Li, Qifei Wang, Junfeng He, Junjie Ke, Ming-Hsuan Yang, Irfan Essa, Jinwoo Shin, Feng Yang, Yinxiao Li1",
"organization": "Google DeepMind, KAIST, Google, Google Research, Georgia Institute of Technology",
"date": "20250204",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2502.02588",
"code_url": "",
"name": "CaPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It proposes calibrated preference optimization by calculating average win-rate of each sample to other samples as the reward.""",
"details": 
"""
""",
},
{
"title": "Curriculum Direct Preference Optimization for Diffusion and Consistency Models",
"author": "Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Nicu Sebe, Mubarak Shah",
"organization": "University of Bucharest, Romania, University of Trento, Italy, University of Central Florida, US",
"date": "20240522",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2405.13637",
"code_url": "https://github.com/CroitoruAlin/Curriculum-DPO",
"name": "Curriculum DPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It applies <b>curriculum learning</b> to DPO by learning from win samples and lose samples with their differences from small to large.""",
"details": 
"""
""",
},
{
"title": "Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization",
"author": "Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, Liang Zheng",
"organization": "The Australian National University, University of Liverpool, Southeast University, Microsoft, Microsoft Research Asia",
"date": "20240606",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2406.04314",
"code_url": "https://github.com/RockeyCoss/SPO",
"name": "SPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It trains a noise-aware reward model, and contructs DPO data from noisy samples.""",
"details": 
"""
<ul>
    <li> <b>DPO data construcion.</b> Start from a initial noise, denoise to some steps and build DPO samples from a noise-aware reward model.
</ul>
""",
},
{
"title": "InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment",
"author": "Yunhong Lu, Qichao Wang, Hengyuan Cao, Xierui Wang, Xiaoyin Xu, Min Zhang",
"organization": "Zhejiang University, Shanghai Institute for Advanced Study-Zhejiang University",
"date": "20250324",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2503.18454",
"code_url": "https://github.com/JaydenLyh/InPO",
"name": "InPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """Instead of maximizing accumulative rewards, it only maximizes the reward of the latent variable that have a strong correlation with the data.""",
"details": 
"""
fig: fig2.png 300
cap: It is more <b>trainig efficient</b>.
""",
},
{
"title": "Personalized Preference Fine-tuning of Diffusion Models",
"author": "Meihua Dang, Anikait Singh, Linqi Zhou, Stefano Ermon, Jiaming Song",
"organization": "Stanford University, Luma AI",
"date": "20250111",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2501.06655",
"code_url": "",
"name": "PPD",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It introduces <b>personalized preference alignment</b> by injecting VLM embeddings into diffusion models through cross-attention.""",
"details": 
"""
<ul>
    <li> <b>VLM.</b> Use LLaVA-OneVision to extract user preference embeddings from few-shot pairwise preference examples for each user.
    <li> <b>Cross-attention.</b> Similar to IP-Adapter, it injects VLM-based user embeddin via cross-attention, and adds the embeddings to text embeddings.
</ul>
fig: fig1.png 450
cap: <b>Structure.</b> Only the cross-attention module is optimized.
""",
},
{
"title": "VideoDPO: Omni-Preference Alignment for Video Diffusion Generation",
"author": "Runtao Liu, Haoyu Wu, Zheng Ziqiang, Chen Wei, Yingqing He, Renjie Pi, Qifeng Chen",
"organization": "HKUST, Renmin University of China, Johns Hopkins University",
"date": "20241218",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2412.14167",
"code_url": "https://github.com/CIntellifusion/VideoDPO",
"name": "VideoDPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It builds a metric for quality and semantic alignment evaluation, then uses the metric to build DPO data for preference alignment of video generation.""",
"details": 
"""
""",
},
{
"title": "RDPO: Real Data Preference Optimization for Physics Consistency Video Generation",
"author": "Wenxu Qian, Chaoyue Wang, Hou Peng, Zhiyu Tan, Hao Li, Anxiang Zeng",
"organization": "Fudan University, Shopee Inc",
"date": "20250623",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2506.18655",
"code_url": "",
"name": "RDPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It constructs positive DPO data by <b>reversing-then-denoising real data</b> to overcome the large domain gap between synthetic data and real data.""",
"details": 
"""
<ul>
    <li> <b>DPO data construction.</b> Positive samples: reversing-then-denoising real data. Negative samples: denoising from noise.
    <li> <b>Rejection sampling.</b> select the instance that is closest (L2 distance) to model's own sample at the same timestep.
    <li> <b>Progressive training.</b> Apply reversing-then-denoising sequentially from heavy to light. 8K preference pairs.
</ul>
fig: fig1.png 500
cap: <b>Construct DPO samples</b> (left) & <b>progressive training</b> (right).
fig: fig2.png 650
cap: Apply <b>SFT loss</b> to DPO loss improves performance. 
fig: fig3.png 700
cap: Mix <b>human-annotation data</b> improves performance.
fig: fig4.png 350
cap: <b>Progressive training.</b> Stage 1 requres larger reversing/sampling steps, e.g., 42. Stage 2 requries smaller, e.g., 40.
fig: fig5.png 700
cap: <b>Progressive training</b> is effective.
""",
},
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
"info": "",
"summary": """It employs LLM architecture to achieve auto-regressive video generation with some improvement on RoPE and masking strategy.""",
"details": 
"""
<ul>
    <li><b>Structure:</b> Llama with a new RoPE strategy to model multimodal spatiotemporal dependency.
    <li><b>Tokenizer:</b> Cosmos's visual tokenizer with spatiotemporal compression rates of 8x8x4; Chameleon's text encoder.
    <li> <b>Model size:</b> 0.5B, 1B, and 3B.
</ul>
""",
},
{
"title": "MAGI-1: Autoregressive Video Generation at Scale",
"author": "Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W.Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, Yuqiao Li",
"organization": "Sand AI",
"date": "20250519",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2505.13211",
"code_url": "https://github.com/SandAI-org/Magi-1",
"name": "Magi-1",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """It achieves <b>chunk-wise</b> auto-regressive video generation by employing transformer-based VAE, progressive-noise causal modeling with flow matching, advanced attention/distillation techniques to enable streaming-capable video generation with fixed peak inference costs regardless of video length.""",
"details": 
"""
<ul>
    <li> <b>VAE training.</b> (1) Stage 1: use training data of fixed-size videos with 256x256 resolution and 16 frames; (2) Stage 2: use mixed training data of images and 16-frame videos, and use variable resolution and aspect ratio. Training loss: <i>L = L_1 + L_KL + L_LPIPS + L_GAN</i>.
    <li> <b>VAE inference.</b> Use sliding window with size of 256x256 with a stride of 192 (25% overlap). Sliding windows are not applied to temporal frames.
    <li> <b>Model structure.</b> It is based on DiT with some modifications: (1) Use T5 as the text encoder; (2) Use learnable 3D RoPE to encode temporal positional information; (3) Use new kernel called Flexible-Flash-Attention; (4) Replace multi-head attention by grouped-query attention; (5) Apply LayerNorm before and after FFN and use SwiGLU to stablize training; (6) Constrain scaling value of AdaLN to [-1, 1] to stablize training.
    <li> <b>Guidance.</b> <i>output = (1 - w_prev) * output_current + (w_prev - w_text) * output_prev + w_text * output_prev</i> (see the paper for details).
    <li> <b>Prompt enhancement</b> for inference. Use distilled MLLM to enhance prompts. (1) Stage 1: analyze and describe the image content; (2) Stage 2: predict the temporal evolution of the scene or objects in the first frame, such as actions, motion trajectories, and transitions.
</ul>
fig: fig2.png 350
cap: <b>VAE structure.</b> Videos are compressed by 8x8 spatially and 4 temporally, generating 16-channel features.
fig: fig3.png 450
cap: <b>VAE performance.</b> Despite having the largest model size, its encoding and decoding is efficient. 
fig: fig1.png 800
cap: <b>Model design.</b> It generates videos chunk-by-chunk, where a chunk (usually 24 frames) is denoised to a certain extent and the next chunk begins generation (conditioned on all preceding chunks). The earlier chunks are cleaner than later ones. It allows multiple chunks (often 4) to be precessed concurrently. It unifies text-to-video, video continuation, and image-to-video generation.
fig: fig4.png 400 fig5.png 400
cap: <b>Model structure.</b>
fig: fig6.png 400
cap: <b>Data processing pipeline.</b> (1) Video quality: DOVER technical score; (2) Aesthetics: LAION aesthetic; (3) Overexposed & underexposed: average brightness on HSI color space; (4) Motion: RAFT optical flow model with saliency detection model; (5) Camera movement stability: evaluate consistency of optical flow between adjacent frames; (6) Slides movement: if divergence of optical flow remains consistently low; (7) Border: edge detection & Hough transform; (8) Text; (9) Logo: Florence-2; (10) Corner face: face detection model; (11) Transition: use CLIP to compute similarity between adjacent keyframes; (12) Deduplication: use pairwise similarity scores from CLIP & DINOv2; (13) Complex filtering: MLLM.
fig: fig7.png 400
cap: <b>Attributes for caption instruction.</b> Captioning by two stages: answer predefined attributes & final caption based on observed attributes. Based on the advantages of auto-regressive video generation, it applies fine-grained second-by-second descrptions for each video clip.
fig: fig8.png 300
cap: <b>Data configuration.</b>
""",
},
{
"title": "Step1X-Edit: A Practical Framework for General Image Editing",
"author": "Step1X-Image Team",
"organization": "StepFun",
"date": "20250424",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2504.17761",
"code_url": "https://github.com/stepfun-ai/Step1X-Edit/",
"name": "Step1X-Edit",
"comment": "",
"category": "Editing & Inpainting & Outpainting",
"jupyter_notes": "",
"info": "",
"summary": """It uses a <b>MLLM to generate condition embedding</b> of the reference image and instructions for image generation editing.""",
"details": 
"""
<ul>
    <li> <b>Training date:</b> 1M images & 20M instruction-image data.
    <li><b>Data construction.</b> (1) Subject addition and removal; (2) Subject replacement and background change; (3) Color Alteration and material modification; (4) Text modification; (5) Motion change; (6) Portrait editing; (7) Style transfer; (8) Tone transformation.
    <li><b>Caption strategy.</b> Redundancy-enhanced annotation: multi-round annotation strategy. Stylized annotation via contextual examples: use style-aligned examples as contextual references. Use GPT-4o to annotate data for training in-house annotators. Bilingual: Chinese and English.
</ul>
fig: fig1.png 600
cap: <b>Structure.</b> <b>Multimodal large language model (Qwen-VL)</b> is used to generate embeddings of instruction and reference images.
""",
},
{
"title": "Denoising Diffusion Implicit Models",
"author": "Jiaming Song, Chenlin Meng, Stefano Ermon",
"organization": "Stanford University",
"date": "20201006",
"venue": "ICLR 2021",
"pdf_url": "https://arxiv.org/pdf/2010.02502",
"code_url": "https://github.com/ermongroup/ddim/",
"name": "DDIM",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "DDIM.ipynb",
"info": "**",
"summary": """Accelerate sampling of diffusion models by introducing a <b>non-Markovian, deterministic process</b> that achieves high-quality results with fewer steps while preserving training consistency.""",
"details": 
"""
fig: fig1.png 500
cap: Comparisons between <b>Markovian DDPM</b> (left) and <b>non-Markovian DDIM</b> (right).
fig: fig2.png 250
cap: <b>Accelerate sampling</b> by skipping time steps.
""",
},
{
"title": "Improving Video Generation with Human Feedback",
"author": "Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, Wanli Ouyang",
"organization": "CUHK, Tsinghua University, Kuaishou Technology, Shanghai Jiao Tong University, Shanghai AI Lab",
"date": "20250123",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2501.13918",
"code_url": "https://github.com/KwaiVGI/VideoAlign/",
"name": "Flow-RWR, Flow-DPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It introduces a human preference video dataset, and adapts diffusion-based reinforcement learning to flow-based video generation models.""",
"details": 
"""
""",
},
{
"title": "Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step",
"author": "Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, Pheng-Ann Heng",
"organization": "CUHK, Peking University, Shanghai AI Lab",
"date": "20250123",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2501.13926",
"code_url": "https://github.com/ZiyuGuo99/Image-Generation-CoT/",
"name": "PARM",
"comment": "",
"category": "Inference-Time Improvement",
"jupyter_notes": "",
"info": "",
"summary": """It applies the idea of <b>Chain-of-Thought</b> into image generation and combines it with reinforcement learning to further improve performance.""",
"details": 
"""
fig: fig1.png 600
cap: <b>Strategy comparisons.</b> ORM is coarse, PRM does not know when to make decision, PARM combines them.
fig: fig2.png 250
cap: It is observed that <b>self-correction</b> also works in image generation by fine-tuning Show-o.
""",
},
{
"title": "Is Noise Conditioning Necessary for Denoising Generative Models?",
"author": "Qiao Sun, Zhicheng Jiang, Hanhong Zhao, Kaiming He",
"organization": "MiT",
"date": "20250218",
"venue": "ICML 2025",
"pdf_url": "https://arxiv.org/pdf/2502.13129",
"code_url": "",
"name": "noise-unconditional model",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """Theoretical and empirical analysis on denoising diffusion models <b>without a timestep input</b> for image generation.""",
"details": 
"""
<ul>
    <li> Many denoising generative models perform <b>robustly</b> even in the absence of noise conditioning.
    <li> <b>Flow</b>-based ones can even produce <b>improved results</b> without noise conditioning.
</ul>
""",
},
{
"title": "InstructVideo: Instructing Video Diffusion Models with Human Feedback",
"author": "Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, Dong Ni",
"organization": "Zhejiang University, Alibaba Group, Tsinghua University, Singapore University of Technology and Design, Nanyang Technological University, University of Cambridge",
"date": "20231219",
"venue": "CVPR 2024",
"pdf_url": "https://arxiv.org/pdf/2312.12490",
"code_url": "https://github.com/ali-vilab/VGen/",
"name": "InstructVideo",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It uses HPS v2 to provide reward feedback and train video generation models in an editing manner.""",
"details": 
"""
""",
},
{
"title": "Optimizing Prompts for Text-to-Image Generation",
"author": "Yaru Hao, Zewen Chi, Li Dong, Furu Wei",
"organization": "Microsoft Research",
"date": "20221219",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2212.09611",
"code_url": "https://github.com/microsoft/LMOps/",
"name": "promptist",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It uses <b>LLM to refine prompts</b> for preference-aligned image generation by taking relevance and aesthetics as rewards.""",
"details": 
"""
fig: fig1.png 500
cap: <b>Training pipeline.</b> (1) Fine-tune a language model (LM) to learn to optimize prompts; (2) Further fine-tune LM with PPO.
""",
},
{
"title": "Ideas in Inference-time Scaling can Benefit Generative Pre-training Algorithms",
"author": "Jiaming Song, Linqi Zhou",
"organization": "Luma AI",
"date": "20250310",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.07154",
"code_url": "",
"name": "Inference can Beat Pretraining",
"comment": "",
"category": "Inference-Time Improvement",
"jupyter_notes": "",
"info": "",
"summary": """Analyze generative pre-training from an <b>inference-first</b> idea, and scaling inference from a perspective of scaling sequence length & refinement steps.""",
"details": 
"""
<ul>
    <li> Pre-training algorithms should have <b>inference-scalability</b> in sequence length and refinement steps.
    <li> Algorithms should scale training efficiently by <b>reduing inference computation</b>.
    <li> One should verify whether the model has <b>enough capacity</b> to represent the target distribution during inference.
    <li> <b>Not scalable in either sequence length or refinement steps:</b> VAE, GAN, Normalizing Flows.
    <li> <b>Scalable in sequence length but not refinement steps:</b> GPT, PixelCNN, MaskGiT, VAR.
    <li> <b>Scalable in refinement steps but not in sequence length:</b> diffusion models, energy-based models, consistency models.
    <li> <b>Scalable in both, with sequence length in the outer loop:</b> AR-Diffusion, Rolling diffusion, MAR.
    <li> <b>Scalable in both, with refinement steps in the outer loop:</b> autoregression distribution smoothing.
</ul>
""",
},
{
"title": "Evaluating Text-to-Visual Generation with Image-to-Text Generation",
"author": "Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, Deva Ramanan",
"organization": "Crnegie Mellon University, Meta",
"date": "20240401",
"venue": "ECCV 2024",
"pdf_url": "https://arxiv.org/pdf/2404.01291",
"code_url": "https://github.com/linzhiqiu/t2v_metrics/",
"name": "VQAScore",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """VQAScore: alignment probability of "yes" answer from a VQA model (CLIP-FlanT5); GenAI-Bench: 1600 prompts for image generation evaluation.""",
"details": 
"""
""",
},
{
"title": "T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation",
"author": "Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, Xihui Liu",
"organization": "The University of Hong Kong, The Chinese University of Hong Kong, Huawei Noah's Ark Lab",
"date": "20240719",
"venue": "T2V-CompBench",
"pdf_url": "https://arxiv.org/pdf/2407.14505",
"code_url": "https://github.com/KaiyueSun98/T2V-CompBench/",
"name": "T2V-CompBench",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """Evaluate compositional video generation capability: consistent attribute, dynamic attribute, spatial relationships, motion, action, object interations, numeracy.""",
"details": 
"""
<ul>
    <li> Find nouns and verbs by identifying them using WordNet from <b>Pika Discord channels</b>, used to generate prompts by GPT-4.
    <li> <b>Consistent attribute binding:</b> two objects, two attributes, and at least one active verb from color, shape, texture, and human-related attributes.
    <li> <b>Dynamic attribute binding:</b> color and light change, shape and size change, texture change, combined change.
    <li> <b>Spatial relationships:</b> two objects with spatial relationships like "on the left of".
    <li> <b>Motion binding:</b> one or two objects with specified moving direction like "leftwards".
    <li> <b>Action binding:</b> bind actions to corresponding objects.
    <li> <b>Object interactions:</b> dynamic interactions like pysical interactions.
    <li> <b>Generative numeracy:</b> a specific number of objects.
    <li> <b>Video LLM-based metrics (Grid-LLaVa)</b> is used for evaluating consistent attribute binding, action binding, object interactions.
    <li> <b>Image LLM-based metrics (LLaVa)</b> is used for evaluating dynamic attribute binding.
    <li> <b>Grounding DINO</b> is used for evaluating spatial relationships and numeracy.
    <li> <b>Grounding SAM + DOT</b> is used for evaluating motion binding.
</ul>
fig: fig1.png 800
cap: <b>Categories (left), evaluation methods (middle), and benchmarking model performance (right).</b>
""",
},
{
"title": "Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection",
"author": "Lichen Bai, Shitong Shao, Zikai Zhou, Zipeng Qi, Zhiqiang Xu, Haoyi Xiong, Zeke Xie",
"organization": "The Hong Kong University of Science and Technology (Guangzhou), Mohamed bin Zayed University of Artificial Intelligence, Baidu Inc",
"date": "20241214",
"venue": "ICLR 2025",
"pdf_url": "https://arxiv.org/pdf/2412.10891",
"code_url": "https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/",
"name": "Z-Sampling",
"comment": "",
"category": "Inference-Time Improvement",
"jupyter_notes": "",
"info": "",
"summary": """It emploits <b>guidance gap between denosing and inversion</b> by iteratively performing them for improve image generation quality.""",
"details": 
"""
fig: fig2.png 300
cap: It capture more <b>semantics</b> by denoising more times.
fig: fig1.png 300
cap: It is more <b>efficient & effective</b> than common denoising.
""",
},
{
"title": "Understanding Diffusion Models: A Unified Perspective",
"author": "Calvin Luo",
"organization": "Google Brain",
"date": "20220825",
"venue": "arXiv 2022",
"pdf_url": "https://arxiv.org/pdf/2208.11970",
"code_url": "",
"name": "Unified Perspective on Diffusion Models",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """Introduction to VAE, DDPM, score-based generative model, guidance from a <b>unified generative perspective</b>.""",
"details": 
"""
""",
},
{
"title": "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps",
"author": "Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, Saining Xie",
"organization": "NYU, MIT, Google",
"date": "20250116",
"venue": "CVPR 2025",
"pdf_url": "https://arxiv.org/pdf/2501.09732",
"code_url": "",
"name": "Inference-Time Scaling Analysis",
"comment": "",
"category": "Inference-Time Improvement",
"jupyter_notes": "",
"info": "**",
"summary": """Analysis on <b>inference-time scaling</b> of diffusion models for image generation from the axes of <b>verifiers</b> and <b>algorithms</b>.""",
"details": 
"""
<ul>
    <li>Use some <b>verifiers</b> to provide feedback: FID, IS, CLIP, DINO; Aesthetic Score Predictor, CLIPScore, ImageReward, Ensemble.</li>
    <li>Use some <b>algorithms</b> to find better noise: Random Search, Zero-Order Search, Search Over Paths.</li>
    <li><b>Random Search:</b> run using different initial random noise and select the best final result by the verifier.</li>
    <li><b>Zero-Order Search:</b> run under different random noise around a pivot noise and select the best final result by the verifier, the best one is then served as a new pivot for next round search.</li>
    <li><b>Search Over Paths:</b> run under different random noise to a specific step, sample noises for each noisy sample and simulate forward process, then perform denoising and select the best candiate using the verifier, continue this process until finish denoising.</li>
    <li><b>Scaling through search</b> leads to substantial improvement across model sizes.</li>
    <li>No single <b>verifier-algorithm configuration</b> is universally optimal.</li>
    <li><b>Inference-time search</b> further improves performance of the model which has already been fine-tuned.</li>
    <li><b>Fewer denoising steps but more searching iterations</b> enables efficient convergence but lower final performance.</li>
    <li>With a fixed inference compute budget, performing <b>search on small models</b> can outperform larger models without search.</li>
</ul>
fig: fig1.png 400
cap: <b>Scale with search</b> is more effective than scale with denoising steps.
fig: fig2.png 400
cap: <b>Random Search performs the best</b> because it has larger space that converges the fastest.
""",
},
{
"title": "Diffusion Model Alignment Using Direct Preference Optimization",
"author": "Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik",
"organization": "Salesforce AI, Stanford University",
"date": "20231121",
"venue": "CVPR 2024",
"pdf_url": "https://arxiv.org/pdf/2311.12908",
"code_url": "https://github.com/SalesforceAIResearch/DiffusionDPO/",
"name": "Diffusion-DPO",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "Diffusion-DPO.ipynb",
"info": "**",
"summary": """It adapts <b>Direct Preference Optimization (DPO)</b> from large language models to diffusion models.""",
"details": 
"""
<ul>
    <li> <b>Model & dataset.</b> It trains SD1.5 and SDXL1.0 on <i>Pick-a-Pic</i> human preference data consisting of 850K pairs from 59K unique prompts.
    <li> <b>Evaluations</b> are performed on Pick-a-Pic validation set, Partiprompt, and HPS v2.
</ul>
""",
},
{
"title": "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation",
"author": "Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, Robin Rombach",
"organization": "Stability AI",
"date": "20240318",
"venue": "SIGGRAPH Asia 2024",
"pdf_url": "https://export.arxiv.org/pdf/2403.12015",
"code_url": "",
"name": "SD3-Turbo",
"comment": "",
"category": "Acceleration",
"jupyter_notes": "",
"info": "**",
"summary": """It performs distillation of diffusion models in <b>latent space</b> using <b>teacher-synthetic data</b> and optimizing adversarial loss with <b>teacher as discriminator</b>.""",
"details": 
"""
fig: fig1.png 500
cap: <b>ADD:</b> (1) An adversarial loss for deceiving a discriminator (DINO v2); (2) A distillation loss for matching denoised output to that of a teacher. <b>The proposed LADD:</b> (1) Use <i>teacher-generated images</i> as the student input; (2) Use <i>the teacher</i> as the discrinimator. <b>Advantages:</b> (1) It is <i>efficient</i> to distill model in latent space; (2) Diffusion model as the discriminator provides <i>noise-level feedback</i>, handles <i>multi-aspect ratio data</i>.
fig: fig2.png 700
cap: (1) Training on <b>synthetic data</b> works better than real data. (2) Training on synthetic data only needs the <b>adversarial loss</b>. CS: CLIPScore.
fig: fig3.png 700
cap: Training using <b>LADD performs better than LCM</b>.
fig: fig4.png 700
cap: <b>Student model size</b> significant impacts performance, while the benefits of teacher models and data quality plateau.
fig: fig5.png 500
cap: Use LoRA for DPO-traning, and apply <b>DPO-LoRA</b> after LADD training.
""",
},
{
"title": "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment",
"author": "Dhruba Ghosh, Hanna Hajishirzi, Ludwig Schmidt",
"organization": "University of Washington, Allen Institute for AI, LAION",
"date": "20231017",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2310.11513",
"code_url": "https://github.com/djghosh13/geneval/",
"name": "GenEval",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """An <b>object-focused</b> framework for image generation evaluation.""",
"details": 
"""
fig: fig1.png 600
cap: <b>GenEval pipeline.</b> Detect objects using Mask2Former detector and evaluate attributes of them.
fig: fig2.png 550
cap: <b>Evaluation perspectives.</b>
""",
},
{
"title": "VBench: Comprehensive Benchmark Suite for Video Generative Models",
"author": "Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, Ziwei Liu",
"organization": "Nanyang Technological University, Shanghai Artificial Intelligence Laboratory, The Chinese University of Hong Kong, Nanjing University",
"date": "20231129",
"venue": "CVPR 2024",
"pdf_url": "https://arxiv.org/pdf/2311.17982",
"code_url": "https://github.com/Vchitect/VBench/",
"name": "Vbench",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """It evaluates video generation from 16 dimensions within the perspectives of video quality and video-prompt consistency.""",
"details": 
"""
<ul>
    <li> <b>Content Categories:</b> animal, architecture, food, human, lifestyle, plant, scenary, vehicles.
    <li> <b>Temporal quality-subject consistency:</b> DINO feature similarity across frames.
    <li> <b>Temporal quality-background consistency:</b> CLIP feature similarity across frames.
    <li> <b>Temporal quality-temporal flickering:</b> mean absolute difference across frames.
    <li> <b>Temporal quality-motion smoothness:</b> use video frame interpolation model to evaluate motion smoothness.
    <li> <b>Temporal quality-dynamic degree:</b> use RAFT to estimate degree of dynamics.
    <li> <b>Frame-wise quality-aesthetic quality:</b> use LAION aesthetic predictor.
    <li> <b>Frame-wise quality-imaging quality:</b> use MUSIQ image quality predictor.
    <li> <b>Semantics-object class:</b> use GRiT to detect classes.
    <li> <b>Semantics-multiple objects:</b> detect success rate of generating all objects.
    <li> <b>Semantics-human action:</b> use UMT to detect specific actions.
    <li> <b>Semantics-color:</b> use GRiT for color captioning.
    <li> <b>Semantics-spatial relationship:</b> use rule-based evaluation.
    <li> <b>Semantics-scene:</b> use Tag2Text for scene captioning.
    <li> <b>Style-appearance style:</b> use CLIP feature similarity.
    <li> <b>Style-temporal style:</b> use ViCLIP to calculate video feature and temporal style description feature similarity.
    <li> <b>Overall consistency:</b> use ViCLIP to evaluate overall semantics and style consistency.
</ul>
""",
},
{
"title": "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation",
"author": "Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu",
"organization": "The University of Hong Kong, Huawei Noah's Ark Lab",
"date": "20230712",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2307.06350",
"code_url": "https://github.com/Karine-Huang/T2I-CompBench/",
"name": "T2I-CompBench",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """It uses 6000 prompts to evaluate model capability on compositional generation, including attribute binding, object relationship, complex compositions.""",
"details": 
"""
<ul>
    <li> <b>Attribute binding prompts:</b> at least two objects with two attributes from color, shape, texture.
    <li> <b>Object relationship prompts:</b> at least two objects with spatial relationship or non-spatial relationship.
    <li> <b>Complex compositions prompts:</b> more than two objects or more than two sub-categories.
</ul>
fig: fig1.png 700
cap: <b>Evaluation methods.</b> Use disentangled BLIP-VQA to evaluate attribute binding, UniDet-based metric to evaluate spatial relationship, CLIPScore to evaluate non-spatial relationship, and 3-in-1 metric (average score of the three metrics) to evaluate complex compositions.
""",
},
{
"title": "CLIPScore: A Reference-free Evaluation Metric for Image Captioning",
"author": "Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi",
"organization": "Allen Institute for AI, University of Washington",
"date": "20210418",
"venue": "EMNLP 2021",
"pdf_url": "https://arxiv.org/pdf/2104.08718",
"code_url": "https://github.com/jmhessel/clipscore",
"name": "CLIPScore",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """It proposes a reference-free metric mainly focusing on semantic alignment for image generation evaluation.""",
"details": 
"""
<ul>
    <li> It calculates the <b>cosine similarity between a caption and an image</b>, multiplying the result by 2.5 (some use 1.).
    <li> It is <b>sensitive</b> to adversarially constructed image captions.
    <li> It <b>generalizes well</b> on never-before-seen images.
    <li> It frees from the shortcomings of <b>n-gram matching</b> that disfavors good captions with new words and favors captions with familiar words.
</ul>
""",
},
{
"title": "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation",
"author": "Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy",
"organization": "Tel Aviv University, Stability AI",
"date": "20230502",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2305.01569",
"code_url": "https://github.com/yuvalkirstain/PickScore/",
"name": "PickScore",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """Pick-a-Pic: use a web app to collect user preferences; PickScore: train a CLIP-based model on preference data for image generation evaluation.""",
"details": 
"""
""",
},
{
"title": "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation",
"author": "Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong",
"organization": "Tsinghua University, Zhipu AI, Beijing University of Posts and Telecommunications",
"date": "20230412",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2304.05977",
"code_url": "https://github.com/THUDM/ImageReward/",
"name": "ImageReward",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """It trains BLIP on 137K human preference image pairs for preference evaluation and use it to tune models by Reward Feedback Learning (ReFL).""",
"details": 
"""
fig: fig1.png 600
cap: <b>Training pipeline.</b> (1) Use DiffusionDB prompts to generate images; (2) Rank; (3) Train model on ranking data; (4) Tune model via ReFL.
fig: fig2.png 300
cap: <b>ReFL algorithm.</b>
""",
},
{
"title": "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation",
"author": "Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong",
"organization": "Tsinghua University, Zhipu AI, Beijing University of Posts and Telecommunications",
"date": "20230412",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2304.05977",
"code_url": "https://github.com/THUDM/ImageReward/",
"name": "ReFL",
"comment": "",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "",
"summary": """It trains BLIP on 137K human preference image pairs for preference evaluation and use it to tune models by Reward Feedback Learning (ReFL).""",
"details": 
"""
fig: fig1.png 600
cap: <b>Training pipeline.</b> (1) Use DiffusionDB prompts to generate images; (2) Rank; (3) Train model on ranking data; (4) Tune model via ReFL.
fig: fig2.png 300
cap: <b>ReFL algorithm.</b>
""",
},
{
"title": "SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL",
"author": "Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, Yu-Gang Jiang",
"organization": "Fudan University, ByteDance Seed",
"date": "20250415",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2504.11455",
"code_url": "https://github.com/wdrink/SimpleAR/",
"name": "SimpleAR",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """A vanilla, open-sourced AR model (0.5B) for 1K text-to-image generation, trained by pre-training, SFT, RL (GRPO), and acceleration.""",
"details": 
"""
<ul>
    <li> <b>Structure.</b> Use <i>Qwen</i> structure and taking <i>Cosmos</i> as the visual tokenizer with 64K codebook and 16 ratio downsampling.
    <li> <b>Training stages.</b> (1) Pre-training on 512 resolution; (2) SFT on 1024 resolution; (3) RL on 1024 resolution.
    <li> Use <b>LLM initialization</b> does not improve DPG-Bench performance.
    <li> Use <b>2D RoPE</b> will not improve performance, but is necessary for dynamic resolution generation.
    <li> Use <b>GRPO</b> with CLIP as the reward model improves more than using HPS v2.
</ul>
""",
},
{
"title": "Seedream 3.0 Technical Report",
"author": "ByteDance Seed Vision Team",
"organization": "ByteDance",
"date": "20250415",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2504.11346",
"code_url": "",
"name": "Seedream 3.0",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """<b>ByteDance Seed Vision Team</b>'s text-to-image generation model, improving Seedream 2.0 by representation alignment, larger reward models.""",
"details": 
"""
<ul>
    <li> Propose <b>defect-aware training:</b> stop gradient on watermarks, subtitles, overlaid text, mosaic pattern.
    <li> Introduce a <b>representation alignment loss</b>: cosine distance between the feature of MMDiT and DINOv2-L.
    <li> Find <b>scaling property of VLM-based reward model</b>.
    <li> Other improvements: (1) mixed-resolution training; (2) <b>cross-modality RoPE</b>; (3) diverse aesthetic captions in SFT.
</ul>
""",
},
{
"title": "Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model",
"author": "ByteDance Seaweed Team",
"organization": "ByteDance",
"date": "20250411",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2504.08685",
"code_url": "",
"name": "Seaweed-7B",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """<b>ByteDance Seaweed Team</b>'s text-to-video and image-to-video generation model (7B), trained on O(100M) videos using 665K H100 GPU hours.""",
"details": 
"""
fig: fig2.png 450
cap: <b>VAE</b> with compression ratio of 16x16x4 (48 channels) or 8x8x4 (16 channels). Using L1 + KL + LPIPS + adversarial losses. Using an <i>image discriminator and a video discriminator</i> is better than using either one. <i>Compressing using VAE outperforms patchification in DiT, and faster</i>.
fig: fig3.png 200
cap: <b>VAE training stages</b> for images and videos.
fig: fig4.png 800
cap: Use <b>mixed resolution & durations & frame rate</b> VAE training converges slower but performs better than training on a low resolution.
fig: fig6.png 650
cap: <b>Full attention</b> enjoys training scalability.
fig: fig5.png 250
cap: <b>Figure 5.</b> The proposed <b>hybrid-stream</b> is better than dual-stream (MMDiT).
fig: fig7.png 500
cap: <b>4-stage pre-training.</b> (1) <b>Multi-task pre-training:</b> text-to-video, image-to-video, video-to-video. Input features and conditions are channel-concatenated, with a binary mask indicating the condition. Ratio of image-to-video is 20% during pre-training, and increases to 50%-75% detached for fine-tuning. (2) <b>SFT:</b> use 700K good videos and 50K top videos; The semantic alignment ability drops a little. (3) <b>RLHF:</b> lr=1e-7, beta=100, select win-lose from 4 candidates. (4) <b>Distillation:</b> trajectory segmented consistency distillation + CFG distillation + adversarial training, distill to 8 steps.
""",
},
{
"title": "Wan: Open and Advanced Large-Scale Video Generative Models",
"author": "Tongyi Wanxiang",
"organization": "Alibaba",
"date": "20250326",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2503.20314",
"code_url": "https://github.com/Wan-Video/Wan2.1/",
"name": "Wan",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "",
"summary": """<b>Alibaba Tongyi Wanxiang</b>'s text-to-video and image-to-video generation models (14B) with DiT structure.""",
"details": 
"""
<p><b>Data procssing pipeline</b>. <i>Fundamental dimensions:</i> text, aesthetic, NSFW score, watermark and logo, black border, overexposure, synthetic image, blur, duration and resolution. <i>Visual quality:</i> clustering, scoring. <i>Motion quality:</i> optimal motion, medium-quality motion, static videos, camera-driven motion, low-quality motion, shaky camera footage. <i>Visual text data:</i> hundreds of millions of text-containing images by rendering Chinese characters on a pure white background and large amounts from real-world data. <i>Captions:</i> celebrities, landmarks, movie characters, object counting, OCR, camera angle and motion, categories, relational understanding, re-caption, editing instruction caption, group image description, human-annotated captions.</p>
fig: fig1.png 450 fig2.png 450
cap: <b>VAE</b> with 127M parameters and 8x8x4 compression ratio. Three-stage training: (1) Train 2D image VAE. (2) Train 3D causal VAE with 128x128x5. (3) Adversarial training with a 3D discriminator. It optimizes <i>L = L_1 loss + KL loss + L_LPIPS</i> loss. It replace all GroupNorm layers with RMSNorm layers to preserve temporal causality. It then applies <i>temporal cache</i> to save GPU memory by employing the causality.
fig: fig3.png 500
cap: <b>Architecture</b>. Text prompt encoded by umT5 is injected by cross-attention; timestep is embedded by MLP; using flow-matching loss.
fig: fig4.png 550
cap: <b>I2V framework.</b> Image condition is incorporated through channel-concat and <i>CLIP image encodings</i>.
""",
},
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
"info": "",
"summary": """<b>StepFun</b>'s image-to-video generation model (30B), trained upon Step-Video-T2V, by incorporating conditions of motion and channel-concat image.""",
"details": 
"""
fig: fig1.png 600
cap: <b>Image condition:</b> channel-concat of <i>noise-augmented</i> image condition. <b>Motion condition:</b> optical flow-based motion + timestep.
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
"info": "",
"summary": """<b>ByteDance Sead Vision Team</b> 's image generation model that employs MMDiT structure and has Chinese-English bilingual capability.""",
"details": 
"""
<ul>
    <li> <b>Structure innovation:</b> <i>self-developed bilingual LLM</i> and ByT5 as text encoders; <i>self-developed VAE</i>; learned positional embeddings on text tokens and scaled 2D RoPE on image tokens.
    <li> <b>Training stages:</b> pre-training => continue training => supervised fine-tuning => human feedback alignment.
    <li> <b>Inference stages:</b> user prompt => prompt engineering => text encoding => generation => refinement => output.
    <li> <b>User experience platform:</b> Doubao (豆包) & Dreamina (即梦).
</ul>
fig: fig3.png 600
cap: <b>Model structure</b> is similar to MMDiT (SD3).
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
"info": "",
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
"info": "",
"summary": """<b>StepFun</b>'s open-sourced model (30B) with DiT structure for text-to-video generation.""",
"details": 
"""
fig: fig1.png 500
cap: <b>Structure.</b> A VAE with a 8x8x4 compression ratio and 16 feature channels, bilingual text encoders (HunyuanCLIP and Step-LLM), DiT with RoPE-3D and QK-Norm, and a DPO pipeline. Text prompt conditions are incorporated into DiT by cross-attention modules.
fig: fig2.png 400
cap: <b>VAE</b> compresses videos by 16x16x8 with 16 feature channels.
fig: fig4.png 500
cap: <b>DPO framework.</b> Use training data prompts and handcrafted prompts to generate samples, which are scored through human annotation or reward models. Diffusion-DPO method is adapted here by reducing beta and increasing learning rate for achieving faster convergence.
fig: fig5.png 900
cap: <b>Data pool with 2B video-text pairs & 3.8B image-text pairs.</b> <i>Filters:</i> video segmentation, video quality assessment, aesthetic score, NSFW score, watermark detection, subtitle detection, saturation score, blur score, black border detection, video motion assessment, K-means-based concept balancing, and CLIP score alignment. <i>Video captioning:</i> short caption, dense caption, and original title.
fig: fig3.png 550
cap: <b>Pre-training stages.</b>
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
"info": "",
"summary": """It improves VAR by applying <b>bitwise modeling</b> that makes vocabulary "infinity" to open up new posibilities of discrete text-to-image generation.""",
"details": 
"""
fig: fig1.png 450
cap: <b>Viusal tokenization and quantization.</b> Instead of predicting <i>2**d</i> indices, infinite-vocabulary classifier predicts <i>d</i> bits instead.
fig: fig3.png 350
cap: <b>Infinity</b> is fast and better.
fig: fig4.png 300
cap: <b>Tokenizer</b> outperforms continuous SD VAE.
fig: fig5.png 450
cap: <b>Inifinite-Vocabulary Classifier</b> needs low memory but performs better.
fig: fig8.png 400
cap: <b>Self-correction</b> mitigates the train-test discrepancy.
fig: fig6.png 700
cap: <b>Vocabulary scales well.</b>
fig: fig7.png 650
cap: <b>Scaling up model size.</b> There is strong correlation between validation loss and evaluation metrics (as observed by Fluid).
fig: fig2.png 700
cap: Using <b>2D RoPE</b> outperforms using APE.
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
"info": "",
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
"info": "",
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
"info": "",
"summary": """It shows auto-regressive models with <b>continuous tokens beat discrete tokens counterpart</b>, and finds some empirical observations during scaling.""",
"details": 
"""
fig: fig1.png 450
cap: <b>Image tokenizer:</b> discrete (VQGAN) or continuous (VAE). <b>Text tokenizer:</b> discrete (T5-XXL). <b>Model structure:</b> transformer with cross-attention modules attending to text embeddings. <b>Loss:</b> cross-entropy loss on text tokens and diffusion loss on image tokens.
fig: fig2.png 600
cap: <b>Scaling</b> behavior of validation loss on <b>model size</b>.
fig: fig3.png 600
cap: <b>Random-order masks</b> on <b>continuous image tokens</b> perform the best. Continuous prefers random order, discrete prefers raster order.
fig: fig4.png 550
cap: Random-order masks on continuous tokens <b>scale with training computes</b>.
fig: fig5.png 550
cap: Strong correlation between <b>validation loss</b> and <b>evaluation metrics</b>.
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
"info": "",
"summary": """It proposes diffusion transformer (16B) with <b>Mixture-of-Experts</b> by inserting experts into DiT blocks for image generation.""",
"details": 
"""
<ul>
    <li> Incorporating <b>shared expert routing</b> improves convergence and performance, but the improvement is little when using more than one.
    <li> <b>Increasing experts</b> reduces loss but introduces more loss spikes.
</ul>
fig: fig1.png 550
cap: <b>Structure.</b> It is built upon DiT and replaces MLP within Transformer blocks by sparsely activated mixture of MLPs as experts.
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
"info": "",
"summary": """It shows that applying "next-token prediction" to <b>vanilla autoregressive language models</b> can achieve good  image generation performance.""",
"details": 
"""
<ul>
    <li> It trains a <b>discrete visual tokenizer</b> that is competitive to the continuous ones, e.g., SD VAE, SDXL VAE, Consistency Decoder from OpenAI.
    <li> It shows that <b>vanilla autoregressive models</b>, e.g., LlaMA, without visual inductive biases can serve as the basis of image generation system.
    <li> <b>Training data.</b> 50M subset of LAION-COCO and 10M internal high aesthetics quality images.
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
"info": "**",
"summary": """It improves auto-regressive image generation on image quality, inference speed, data efficiency, and scalability, by proposing <b>next-scale prediction</b>.""",
"details": 
"""
fig: fig1.png 650
cap: <b>Next-scale prediction.</b> Start from 1x1 token map; at each step, it predicts the next higher-resolution token map given all previous ones.
fig: fig3.png 750
cap: <b>Training pipeline of tokenzier and VAR.</b>  Tokenzier (similar to VQ-VAE): the same architecture and training data (OpenImages), using codebook of 4096 and spatial downsample ratio of 16. VAR: the standard transformer with AdaLN; not use RoPE, SwiGLU MLP, RMS Norm.
fig: fig4.png 650
cap: <b>Encoding & decoding of tokenizer.</b>
fig: fig2.png 350
cap: VAR shows good <b>scaling behavior</b>, and significantly outperforms DiT.
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
"info": "**",
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
fig: fig1.png 500
cap: <b>SDXL Structure.</b>
""",
},
{
"title": "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis",
"author": "Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, Hongsheng Li",
"organization": "CUHK, SenseTime Research, Shanghai Jiao Tong University, Centre for Perceptual and Interactive Intelligence",
"date": "20230615",
"venue": "arXiv 2023",
"pdf_url": "https://arxiv.org/pdf/2306.09341",
"code_url": "https://github.com/tgxs002/HPSv2/",
"name": "HPS v2",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """It proposes HPD v2: 798K human preferences on 433K pairs of images; HPS v2: fine-tuned CLIP on HPD v2 for image generation evaluation.""",
"details": 
"""
fig: fig1.png 750
cap: <b>Training pipeline.</b> (1) Clean prompts from COCO captions and DiffusionDB by ChatGPT; (2) Generate images using 9 image generation models; (3) Rank and annotate each pair of images by humans; (4) Finetune CLIP and obtain a preference model to give HPS v2 evaluation score.
""",
},
{
"title": "Human Preference Score: Better Aligning Text-to-Image Models with Human Preference",
"author": "Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, Hongsheng Li",
"organization": "CUHK, SenseTime Research, Shanghai Jiao Tong University, Centre for Perceptual and Interactive Intelligence, Shanghai AI Lab",
"date": "20230325",
"venue": "ICCV 2023",
"pdf_url": "https://arxiv.org/pdf/2303.14420",
"code_url": "https://github.com/tgxs002/align_sd/",
"name": "HPS",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """It fine-tunes CLIP on annotated 98K SD generated images from 25K prompts for image generation evaluation.""",
"details": 
"""
fig: fig1.png 650
cap: <b>Train score model:</b> the same as CLIP except for the sample with the highest preference is taken as the positive; <b>Finetune image generation model using the score model:</b> append a special token to the prompts of worse images for training; remove that token during inference.
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
"jupyter_notes": "DiT.ipynb",
"info": "**",
"summary": """It replaces the conventional U-Net structure with <b>transformer</b> for scalable image generation, the timestep and condition are injected by adaLN-Zero.""",
"details": 
"""
fig: fig1.png 700
cap: <b>Model structure.</b> Use <i>adaLN-Zero</i> structure to inject timestep and class condition performs better than using cross-attention or in-context.
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
"info": "",
"summary": """It proposes a transformer-based video generation model (9B) that performs <b>auto-regressive</b> frame  generation and recursive frame interpolatation""",
"details": 
"""
fig: fig1.png 450
cap: <b>Model structure & training.</b> CogVideo is trained upon CogView2. It generates frames autoregressively and interpolates them recursively.
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
"comment": "It significantly advances visual generation field by enabling efficient, high-quality synthesis via latent-space diffusion. It has over 20,000 citations (as of Jul 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": """It achieves efficient high-resolution image generation by applying diffusion and denoising processes in the <b>compressed VAE latent space</b>.""",
"details": 
"""
fig: fig1.png 350
cap: <b>Strucuture.</b> The conditions are injected by cross-attention.
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
"jupyter_notes": "CFG.ipynb",
"info": "**",
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
"comment": "It shows that a simple, theoretically grounded denoising process can rival and even surpass GANs in sample quality, sparking an explosion of diffusion-based research and aplications across images, videos, audio, and beyond. It has over 20,000 citations (as of Jul 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "DDPM.ipynb",
"info": "**",
"summary": """It proposes <b>denoising diffusion probabilistic models</b> that iteratively denoises data from random noise.""",
"details": 
"""
fig: fig1.png 400
cap: <b>Diffusion (forward) & denoising (reverse) processes.</b>
fig: fig2.png 550
cap: <b>Training & sampling algorithms.</b>
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
"info": "**",
"summary": """In order to generate large scale images efficiently, it improves VQ-VAE by employing a <b>hierarchical organization</b>.""",
"details": 
"""
<ul>
    <li><b>Structure:</b> (1) a top-level encoder to learn top-level priors from images; (2) a bottom-level encoder to learn bottom-level priors from images and top-level priors; (3) a decoder to generate images from both top-level and bottom-level priors.</li>
    <li><b>Training stage 1:</b> training the top-level encoder and the bottom-level encoder to encode images onto the two levels of discrete latent space.</li>
    <li><b>Training stage 2:</b> training PixelCNN to predict bottom-level priors from top-level priors, while fixing the two encoders.</li>
    <li><b>Sampling:</b> (1) sampling a top-level prior; (2) predicting bottom-level prior from the top-level prior using the trained PixelCNN; (3) generating images from both the top-level and the bottom-level priors by the trained decoder.</li>
</ul>
fig: fig1.png 850
cap: <b>Training (left) & sampling (right) frameworks.</b>
fig: fig2.png 500
cap: <b>Training & sampling algorithms.</b>
""",
},
{
"title": "FVD: A new Metric for Video Generation",
"author": "Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, Sylvain Gelly",
"organization": "Johannes Kepler University, IDSIA, Google Brain",
"date": "20190504",
"venue": "ICLR workshop 2019",
"pdf_url": "https://openreview.net/pdf?id=rylgEULtdN",
"code_url": "",
"name": "FVD",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "",
"summary": """Extend FID for video generation evaluation by replacing 2D InceptionNet with pre-trained Inflated 3D convnet.""",
"details": 
"""
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
"info": "**",
"summary": """It proposes <b>vector quantised variational autoencoder</b> to generate discrete codes while the prior is also learned.""",
"details": 
"""
<ul>
    <li><b>Posterior collapse problem:</b> a strong decoder and a strong KL constraint could make the learned posterior <i>q(z|x)</i> very close to prior <i>p(z)</i>, so that the conditional generation task collapses to an unconditional generation task.</li>
    <li><b>How VQ-VAE avoids the collapse problem by employing discrete codes/latents?</b> (1) It learns <i>q(z|x)</i> by choosing one from some candidates rather than directly generating a simple prior; (2) The learned <i>q(z|x)</i> is continuous but <i>p(z)</i> is discrete, so the encoder can not be "lazy".</li>
    <li><b>Optimization objectives:</b> (1) The decoder is optimized by a recontruction loss; (2) The encoder is optimized by a reconstruction loss and a matching loss; (3) The embedding is optimized by a matching loss.</li>
    <li><b>How to back-propagate gradient with quantization exists? Straight-Through Estimator:</b> directly let the graident of loss to the quantized embedding equal to the gradient of loss to the embedding that before being quantized.</li>
</ul>
fig: fig1.png 900
cap: <b>Model structure.</b>
""",
},
{
"title": "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium",
"author": "Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter",
"organization": "Johannes Kepler University Linz",
"date": "20170626",
"venue": "NeurIPS 2017",
"pdf_url": "https://arxiv.org/pdf/1706.08500",
"code_url": "",
"name": "FID",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """Calculate <b>Fréchet distance</b> between Gaussian distributions of InceptionNet features of real-world and synthetic data for image generation evaluation.""",
"details": 
"""
""",
},
{
"title": "Improved Techniques for Training GANs",
"author": "Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen",
"organization": "OpenAI",
"date": "20160610",
"venue": "NeurIPS 2016",
"pdf_url": "https://arxiv.org/pdf/1606.03498",
"code_url": "https://github.com/openai/improved-gan/",
"name": "Inception Score",
"comment": "",
"category": "Datasets & Evaluation",
"jupyter_notes": "",
"info": "**",
"summary": """Calculate <b>KL divergence between p(y|x) and p(y)</b> that aims to minimize the entropy across predictions and maximize the entropy across predictions of classes for image generation evaluation.""",
"details": 
"""
""",
},
]
