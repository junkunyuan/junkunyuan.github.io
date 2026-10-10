---
name: paper-polishing
description: Polish artificial intelligence papers, section by section.
---

# Paper Polishing

Polish **artificial intelligence papers**, section by section.

- **Before any polish.** The user names two things up front, or nothing is touched: the output (revision notes or direct edits) and the mode (major polish, minor polish, or reviewer). A request missing either gets the question back, never a guess; nothing is polished until both are named.

- **Standard.** Aim for the writing quality of ICLR, NeurIPS, ICML, CVPR, and Nature: clear, precise, concise, and calm. Start from the one message each section must convey and cut any sentence that does not advance it; settle what is new and what supports it before touching the wording.

- **References.** Study the original text of [MoCo](https://arxiv.org/abs/1911.05722), [MAE](https://arxiv.org/abs/2111.06377), and [MeanFlow](https://arxiv.org/abs/2505.13447). Adapt their structure, argumentation, sentence rhythm, and use of evidence; let them guide decisions as references, never serve as templates, and never copy their sentences. Framing follows the [strategy cards](https://github.com/Michael-Jiahao-Zhang/game-the-llm-reviewer/blob/main/skills/game-the-llm-reviewer/references/strategies.md).

- **Language and claims.** Plain words, short declarative sentences, active voice with "we", precise verbs ("outperforms by 2.1 AP"), no hype or filler. Match each claim to its evidence, one name per concept, one definition per abbreviation, present tense for what the paper shows.

- **Framing.** Recast a supported difference as an achievement ("removes the need for D", not "does not require D"), a comparison as an effect ("cuts error from 10 to 6.2"), a scope as coverage beside its untested boundary; move a key result earlier in the abstract; no self-dismissal.

- **Equivalence check.** A reframing keeps every comparator, condition, unit, uncertainty, and adverse finding: no new "first", no significance without a test, no best case as an average, no "may" turned into "does". Revert an edit that shifts what a reader can judge, even if stronger.

- **Abstract.** Open with the thesis as a finding ("This paper shows that X is a scalable Y"); then gap, core idea, why it works, two or three headline numbers with dataset and metric, the broader implication. One paragraph of 150–220 words; no citations, undefined abbreviations, or equations.

- **Introduction.** Reach the problem within three sentences and frame the gap as a question the reader can hold; answer it by analysis, then state the idea in one sentence before any detail. Cite a teaser figure early, give results as numbers, prefer prose to a contributions list, close modestly.

- **Related work.** Organize by theme, a bold run-in heading per paragraph ("Masked language modeling."), not one paper per sentence: summarize the idea, cite key work, end by placing this paper against it. Generous to prior work, never dismissive; keep it short.

- **Method.** Open with a one-paragraph overview and its figure, then build from concept to formulation to implementation. Name each component once; state every target’s source and every loss’s computation; define symbols at first use; give each choice its rationale and ablation.

- **Experiments.** State the setup first: datasets, metrics, backbone, budget, baselines. Each ablation answers one question, named by it ("Masking ratio."): numbers first, interpretation second. Keep comparisons fair or say when not, and ablations apart from system comparisons.

- **Conclusion and appendix.** One or two short paragraphs: the thesis against the evidence, a broader implication, honest limitations, no new results. The appendix enables reproduction: hyperparameters, data, architecture, compute, protocols, each cited from the main text.

- **Figures and tables.** Captions stand alone: a bold message, then what is shown, what colors and symbols mean, the setting, the conclusion stated. Table captions sit above, takeaway first; booktabs rules, default row in gray, best in bold, consistent units, decimals, method order.

- **Revision notes.** Return the text with review notes after the phrase, `\yjk{…}` (orange, `\color[RGB]{200,135,0}`, printed `[Junkun: …]`); nothing else changes. Each note states the problem, then the advice, and the fix when clear. Leave a passage alone if no edit keeps it true.

- **Direct edits.** The text is changed in place, and every line of an edited passage runs full. A last line under half the column is cut until it disappears, one over it is filled with real detail (a condition, a number, a reason), never padding; aim at nine-tenths, never at the cost of clarity.

- **Polish modes.** **Major:** structure and content are open; a key problem (a thesis the evidence does not carry, a section out of order) gets a large rewrite, a small one a small edit. **Minor:** structure and content stay as they are; only small edits and wording, sentence by sentence.

- **Reviewer mode.** Read as a strict ICLR, NeurIPS, ICML, or CVPR reviewer: list concerns from major (novelty over prior work, missing baselines or ablations, claims beyond the evidence, unfair comparisons) to minor (clarity, notation, typos, figures), each with a direction for revision.
