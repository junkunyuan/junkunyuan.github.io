LIST = dict()
LIST["file"] = "paper_reading_list/language_generation.html"
LIST["title"] = "Language Generation"
LIST["description"] = "Generate text."
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
    <li>
</ul>
""",
},
]