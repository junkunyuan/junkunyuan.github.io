LIST = dict()
LIST["file"] = "nlp.html"
LIST["title"] = "Natural Language Processing (NLP)"
LIST["description"] = "Understand and generate human language."
LIST["categories"] = ["Foundation Algorithms & Models", "Reinforcement Learning"]
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
"title": "Large Language Diffusion Models",
"author": "Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li",
"organization": "Renmin University of China, Ant Group",
"date": "20250214",
"venue": "arXiv 2025",
"pdf_url": "https://arxiv.org/pdf/2502.09992",
"code_url": "https://github.com/ML-GSAI/LLaDA",
"name": "LLaDA",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces a <b>masked diffusion language model</b> (8B) that matches strong autoregressive LLMs while inherently enabling bidirectional reasoning.
""",
"details": 
"""
<ul>
    <li> It argues that the <b>generative modeling</b> is to learn \(\max_{\\theta}\mathbb{E}_{p_{data}(x)}\log p_{\\theta}(x)\), which <b>not necessarily be auto-regressive</b>.</li>
    <li><b>Intruction-following</b> and <b>in-context learning</b> is also not an exclusive advantage of autoregressive models.</li>
    <li>Auto-regressive models have disadvantages such as <b>high computational costs</b> due to <b>token-by-token generation</b>, and limitations in <b>reversal reasoning</b> due to <b>left-to-right generation</b>.</li>
    <li><b>LLaDA (8B) with 4096 tokens</b> is pre-trained fram scratch on 2.3T tokens using 0.13M H800 GPU hours, followed by SFT on 4.5M pairs.</li>
    <li>LLaDA <b>does not use a causal mask</b>, as it sees the entire context.</li>
    <li>LLaDA uses <b>vanilla multi-head attention</b>, as it is incompatible with KV cache.</li>
</ul>
fig: fig1.png 900
cap: <b>Pre-training.</b> Tokens are independently randomly masked by probability of \(t\sim U[0,1]\), the model predicts the masked tokens by minimizing the cross-entropy loss. <b>SFT.</b> Only response tokens are possibly masked. <b>Inference.</b> Stimulate a diffusion process from \(t=1\) to \(t=0\).
""",
},
{
"title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
"author": "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn",
"organization": "Stanford University, CZ Biohub",
"date": "20230529",
"venue": "NeurIPS 2023",
"pdf_url": "https://arxiv.org/pdf/2305.18290",
"code_url": "",
"name": "DPO",
"comment": "It offers a simple, RL-free recipe to turn human preference data into aligned language models with equal or better performance than RLHF while eliminating reward-model training and heavy hyper-parameter tuning overhead. It has over 5,000 citations (as of Sep, 2025).",
"category": "Reinforcement Learning",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It introduces DPO, a <b>single-stage, RL-free</b> algorithm that directly optimizes a language model on preference data by reparameterizing the Bradley-Terry objective into a simple classification loss.
""",
"details": 
"""
""",
},
{
"title": "Attention Is All You Need",
"author": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
"organization": "Google Brain, Google Research, University of Toronto",
"date": "20170612",
"venue": "NeurIPS 2017",
"pdf_url": "https://arxiv.org/pdf/1706.03762",
"code_url": "",
"name": "Transformer",
"comment": "It revolutionized deep learning by introducing the Transformer architecture, which replaced recurrence with self-attention, enabling massively parallel training and becoming the foundational model for virtually all modern large-scale language systems. It has 192,000 citations (as of Sep, 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "transformer.ipynb",
"info": "**",
"summary": 
"""
It introduces sequence transduction architecture <b>relying solely on multi-head self-attention</b>, dramatically reducing training time.
""",
"details": 
"""
<ul>
    <li>Details to be added</li>
</ul>
""",
},
]