LIST = dict()
LIST["file"] = "reinforcement_learning.html"
LIST["title"] = "Reinforcement Learning"
LIST["description"] = "Learn to make decisions in an environment by maximizing long-term rewards."
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
# "summary": 
# """
# """,
# "details": 
# """
# """,
# },
{
"title": "Reinforcement Learning: An Introduction",
"author": "Richard S. Sutton, Andrew G. Barto",
"organization": "University of Massachusetts Amherst, Carnegie Mellon University",
"date": "19980101",
"venue": "Cambridge 1998",
"pdf_url": "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf",
"code_url": "",
"name": "RL Introduction",
"comment": "It systematizes the foundations of RL by unifying dynamic programming, Monte Carlo methods, and temporal-difference learning into a coherent framework, establishing the theoretical and algorithmic basis for modern RL research. It has over 8,0000 citations (as of Aug 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "Markov Decision Process.ipynb",
"info": "**",
"summary": 
"""
It formalizes the core concepts, algorithms, and theoretical foundations of RL, establishing it as a coherent and accessible discipline.
""",
"details": 
"""
""",
},
{
"title": "An Empirical Evaluation of Thompson Sampling",
"author": "Olivier Capelle, Lihong Li",
"organization": "Yahoo! Research",
"date": "20111212",
"venue": "NeurIPS 2011",
"pdf_url": "https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf",
"code_url": "",
"name": "Thompson Sampling",
"comment": "",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "Multi-Armed Bandit.ipynb",
"info": "**",
"summary": 
"""
It introduces the <b>first large-scale empirical demonstration</b> that Thompson sampling achieves SOTA in real-world bandit problems.
""",
"details": 
"""
""",
},
{
"title": "Finite-time Analysis of the Multiarmed Bandit Problem",
"author": "Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer",
"organization": "University of Technology Graz, Univerisity of Milan, University Dortmund",
"date": "20020501",
"venue": "Machine Learning 2002",
"pdf_url": "https://link.springer.com/content/pdf/10.1023/a:1013689704352.pdf",
"code_url": "",
"name": "ε-greedy & UCB",
"comment": "It fundamentally shifted bandit research by providing the first distribution-free, finite-horizon regret bounds that enabled practical, anytime performance guarantees and sparked a wave of refined algorithms and analyses. It has over 9,000 citations (as of Aug 2025).",
"category": "Foundation Algorithms & Models",
"jupyter_notes": "Multi-Armed Bandit.ipynb",
"info": "**",
"summary": 
"""
It proposes index-based and ε-greedy policies that achieve <b>finite-time logarithmic regret bounds</b> for multi-armed bandit with <b>bounded rewards</b>.
""",
"details": 
"""
""",
},
]