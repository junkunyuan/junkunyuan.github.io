PAPER = dict()
PAPER["file"] = "research.html"
PAPER["title"] = "Research"
PAPER["description"] = "Things about research."
PAPER["categories"] = ["A Guide to Research"]
PAPER["papers"] = [
# {
# "title": "",
# "venue": "",
# "name": "",
# "category": "",
# "details": 
# """
# """,
# "author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
# },
{
"title": "Progress Management",
"venue": "Update and discuss research progress.",
"name": "progress management",
"category": "A Guide to Research",
"details": 
"""
<ul>
    <li> <b>Keep progress updated.</b> Maintain a progress document and update it whenever you achieve new pregress. <a href="resource/else/research progress example.pptx">Example document</a>.
    <li> <b>Hold regular discussions with collaborators.</b> Share progress summaries, achievements, challenges, insights, and next steps with collaborators. To make meeting efficient, clarify: (1) What progress needs to be shared (or not)? (2) Which issues should be discussed? (3) What team members should prepare (e.g., reviewing relevant papers in advance).
    <li> <b>Update progress frequently.</b> Write concise notes on achievements and issues. Keep record clear and well-structured, ensuring quick access and review for the team. Ask yourself: (1) What should I share with the team? (2) Which results / issues are important to recall later?
</ul>
fig: fig1.png 450
cap: Example slide of <b>weekly progress summary</b>.
""",
"author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
},
{
"title": "Paper Review",
"venue": "Track the latest works and discover an open problem.",
"name": "paper review",
"category": "A Guide to Research",
"details": 
"""
<ol>
    <li> <b>Find a research topic.</b> Read recent papers and answer: (1) <i>What is the value of this topic?</i> (2) <i>Are there any well-known researchers or institutions involved in this topic?</i> &nbsp;&nbsp;  Some paper sources: <a href="https://proceedings.neurips.cc//">NeurIPS</a>, <a href="https://openreview.net/group?id=ICLR.cc">ICLR</a>, <a href="https://proceedings.mlr.press/">ICML</a>, <a href="https://openaccess.thecvf.com/menu">CVPR & ICCV & ECCV</a>, <a href="https://aclanthology.org/">ACL & EMNLP</a>.
    <li> <b>Find an open problem.</b> Summarize motivations & contributions of related works and answer: (1) <i>Is there any open problem or bottleneck in this topic? (you may summarize it based on the Future Work or Limitation section of these papers)</i> (2) <i>Is this problem central or just a side detail for this topic?</i> (3) <i>What impact will solving this problem have on this research field? Does solving it advance just the method, or the entire field?</i>
    <li> <b>Find related solutions.</b> Search for relevant solutions (in related papers) to the open problem you find and answer: (1) <i>What they have solved and what have not solved?</i> (2) <i>Is your idea / solution novel or trivial / incremental / just a tweak?</i>
</ol>
fig: fig1.png 450 fig2.png 450
cap: Example slides of <b>"find a research topic" (left)</b> and <b>related work summarization (right)</b>.
""",
"author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
},
{
"title": "Experiment",
"venue": "Prepare, implement, and monitor experiments.",
"name": "experiment",
"category": "A Guide to Research",
"details": 
"""
<ul>
    <li> <b>Code.</b> Choose a suitable codebase from open-source projects related to your topic. It should be flexible, reliable, and easy to use.
    <li> <b>Data.</b> Select training datasets and evaluation benchmarks, typically following those used in the most relevant works to ensure credibility. Preferably, use datasets on which the chosen codebase has already been tested by previous works.
    <li> <b>Reproduce baseline results.</b> Run the codebase on the selected dataset and verify that you can reproduce the results reported in prior works. This step is very important, so ensure you can either reproduce the results or provide a well-reasoned explanation if reproduction fails.
    <li> <b>Implement ideas.</b> Introduce your methods by modifying or adding as little code as possible, keeping implementation clean and transparent.
    <li> <b>Document experiments.</b> Maintain an experiment log and update it whenever you start / complete a new experiment. <a href="resource/else/experiment progress example.xlsx">Example document</a>.
</ul>
fig: fig1.png 800
cap: Example log of <b>experiment progress</b>.
""",
"author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
},
{
"title": "Write Paper",
"venue": "Instructions for paper writing.",
"name": "write paper",
"category": "A Guide to Research",
"details": 
"""
<b>Writing a paper is the most crucial part of research.</b> Through the paper, you tell people: "What problem are you solving, why are you solving it, how did you solve it, what are your final results and findings, and what is the significance and inspiration of your work to the research community?"<br><br>

<b>There is no one-size-fits-all format for paper writing.</b> The following guidance is based solely on my personal experience. Everyone can develop unique and creative ideas for their papers based on their own thinking and insights and the distinctiveness of each research project.

<ul>
    <li> <b>Prepare LaTex templete.</b> Download LaTex templete from official websites of the target conference or jounal and read instructions carefully.
    <li> <b>Build online project.</b> Maintain an online LaTex project so collaborators can help write and polish the paper. <a href="https://www.texpage.com/console">TeXPage</a> is recommended for users in China as <a href="https://overleaf.com/project">Overleaf</a> can sometimes be unstable due to heavy traffic. Upload the LaTex templete and share the projecct with collaborators.
    <li> <b>General writing principle.</b> Adhere to Occam's Razor—write only what is essential, and avoid irrelevant or redundant expressions. Constantly remind yourself: "If you remove this sentence, does it affect the clarity of your paper? Is there a more concise way to express the same idea?" Avoid exaggerated or false statements, as they not only damage your reputation but also give reviewers opportunities to challenge your work.
    <li> <b>Abstract.</b> A good abstract (about 200 words) should clearly convey: (1) The problem you address. (maybe 1 sentence) (2) How you solve it. (maybe 1 sentence) (3) High-level idea and method with important components. (maybe 2-3 sentences) (4) The key results achieved. (maybe 1 sentence) <b>Note:</b> avoid excessive background or motivation—reserve that for Introduction.
    <li> <b>Introduction.</b> It should cover: (1) Background of the topic. (2) Motivation of the research. (3) Motivation of your idea and method. (4) High-level overview of your method with key components. (5) Experiments with major results and findings. (6) The value of the work to this research field. (7) A brief summary of contributions (optional). <b>Note:</b> avoid excessive background or method details—focus on the core contributions.
    <li> <b>Related Works.</b> Provide context by addressing: (1) What previous works have done? (2) What they have overlooked or left unresolved? <b>Note:</b> avoid listing works haphazardly; instead, thoughtfully select representative works and present them in a logical, well-organized manner.
    <li> <b>Method.</b> Clearly present your proposed method by first explaining the high-level motivation and idea. Use figures to illustrate the overall framework and modules. <b>Note:</b> do not overcomplicate your method (such as overusing ineffective symbols or formulas) for sophistication's sake—clarity, simplicity, effectiveness, and scalablity are key values in AI research (think about ResNet, Dropout, Mixup, YOLO, etc.).
    <li> <b>Experiment.</b> This section should address: (1) Datasets / benchmarks used. (2) Implementation details sufficient for reproducibility. (3) Main results compared to relevant and strong baseline methods. (4) Ablation studies to show the role of each component. (5) In-depth visualization, analysis, and insights beyond the numbers. <b>Note:</b> focus on telling readers what the results mean, not just what they are. 
    <li> <b>Conclusion.</b> Summarize the achievement, limitations, and broader implications of this work. Highlight future directions that others can build on. <b>Note:</b> keep this forward-looking, while Abstract and Introduction focus on current contributions.
</ul>
fig: fig1.png 100 fig2.png 550 fig3.png 250
cap: <b>(left)</b> Example of <b>LaTex structure</b>. Each chapter can be added by writing `\input{chapters/abstract}` in `main.tex`. <b>(middle)</b> How to <b>cite a paper</b> by obtaining bib info and inserting it into `main.bib`. <b>(right)</b> The added bib info should be clean so that you can cite it by `~\cite{CLIP}`.
""",
"author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
},
{
"title": "Open Source",
"venue": "Make the project open-source.",
"name": "open source",
"category": "A Guide to Research",
"details": 
"""
<ul>
    <li> <b>Importance of open source.</b> Open source is essential for reproducibility, knowledge sharing, and AI innovation. An open-sourced project also serves as a visible proof of your coding skills, attracting attention to your work and strengthening your research reputation and network.
    <li> <b>Code.</b> Ensure your code is clean: (1) Remove unused files, debug prints, or personal content. (2) Use a clear directory structure (e.g., `data/`, `model/`, `train.py`). (3) Add meaningful comments. (4) Verify that releasing the code does not violate any policies. Provide a concise README, including: (1) A brief introduction. (2) Running environment and dependancies (datasets, checkpoints, etc.). (3) Training and evaluation instructions. (4) Acknowledgement of other open-source projects used. (5) Citation instructions. (6) Contribution guidelines. (7) License.
    <li> <b>Data.</b> Ensure your data is clean: (1) Obtain permission to share it. (2) Remove personal information. (3) Ensure compliance with regulations if human data is involved. (4) Filter out harmful, offensive, or biased content. (5) Organize files clearly (`train/`, `val/`, `test/`). Write a concise README, including: (1) A brieft introduction. (2) Instructions to download, load, and use it. (3) Known biases and risks. (4) License.
</ul>
""",
"author": "","organization": "","date": "", "comment": "", "pdf_url": "","code_url": "", "jupyter_notes": "", "info": "", "summary": """""",
},
]