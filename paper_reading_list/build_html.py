import pandas as pd
from tqdm import tqdm
from datetime import datetime
from resource.main_content import MAIN_CONTENT
from resource import DOMAINS
from resource.utils import get_venue_all, border_color_generator, convert_fig_cap_to_figure

time_now = datetime.now().strftime('%B %d, %Y at %H:%M')

PREFIX = \
f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="resource/html.css" type="text/css">
    <link rel="shortcut icon" href="resource/my_photo.jpg">
    <title>Paper Reading List</title>
    <meta name="description" content="Paper Reading List">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <div id="layout-content" style="margin-top:25px">
<body>
"""
INTRODUCTION = \
f"""
<h1 id="top">title</h1>
<p class="larger"><b>total_paper_num domain_name.</b></p>
<p>Curated by <a href="https://junkunyuan.github.io/">Junkun Yuan</a>.</p>
<p>Click <a href="paper_reading_list.html">here</a> to go back to main contents.</p>
<p><font color=#B0B0B0>Last updated on time_now (UTC+8).</font></p>
"""
SUFFIX  = \
"""
</body>
</html>
"""


def build_main_content_all_domains(domains, num_papers):
    content = """<hr><p id="table" class="larger"><b>Table of contents:</b></p><ul>"""
    for domain, num_paper in zip(domains, num_papers):
        paper_num_display = "" if num_paper == 0 else f"({num_paper} papers)"
        content += f"""<li><a class="no_dec" href={domain["file"]}>{domain["title"]}</a>{paper_num_display} &nbsp; <font color=#B0B0B0>{domain["description"]}</font></li>"""
    content += "</ul>"
    return content


def build_main_content_of_each_domain(domain):
    ## Load papers
    papers = {"title": list(), "author": list(), "organization": list(), "date": list(), 
                "venue": list(), "pdf_url": list(), "code_url": list(), "name": list(), "comment": list(), 
                "category": list(), "jupyter_notes": list(), "summary": list(), "details": list()}
    for paper in tqdm(domain["papers"], desc=f"Read {domain['title']}"):
        for key in papers.keys():
            papers[key].append(paper[key])
    papers = pd.DataFrame(papers)

    ## Build table of contents
    catalog = """<hr><p id='table' class="larger"><b>Table of contents:</b></p><ul>"""
    for category in domain["categories"]:
        paper_choose = papers[papers["category"].str.contains(category)]
        paper_choose = paper_choose.sort_values(by="date", ascending=True)
        paper_names = paper_choose["name"].to_list()
        names = ""
        for i, name in enumerate(paper_names):
            if i == len(paper_names) - 1:
                names += f"""<a class="no_dec" href="#{name}"><font color=#B0B0B0>{name}</font></a>"""
            else:
                names += f"""<a class="no_dec" href="#{name}"><font color=#B0B0B0>{name}</font></a> <font color=#B0B0B0>&nbsp;|&nbsp;</font> """
        catalog += f"""<li><a class="no_dec" id="{category}" href="#{category}-table"><b>{category}:</b></a> {names}</li>"""
    catalog += """</ul>"""
    
    ## Build contents
    content_all_domain = ""
    color_bar_generator = border_color_generator()
    for category in domain["categories"]:
        color_bar = next(color_bar_generator)
        paper_choose = papers["category"].str.contains(category)
        if paper_choose.sum() == 0:
            continue
        paper_choose = papers[paper_choose]
        paper_choose = paper_choose.sort_values(by="date", ascending=False)
        content_cate = f"""<h2 id="{category}-table"><a class="no_dec" href="#{category}">{category}</a></h2>"""
        for _, paper in paper_choose.iterrows():
            code = f"""&nbsp;&nbsp;|&nbsp;&nbsp; <a href="{paper['code_url']}">code</a>""" if len(paper['code_url']) > 0 else ""

            venue = f"""<a href="{paper["pdf_url"]}">{paper["venue"]}</a>"""
            venue_all = get_venue_all(paper["venue"])
            date = datetime.strptime(paper["date"], "%Y%m%d").strftime("%b %d, %Y")
            comment = f"""<p class="paper_detail"><font color=#FF000>{paper["comment"]}</font></p>""" if paper["comment"] else ""
            jupyter_note = ""
            if paper.get("jupyter_notes", ""):
                jupyter_note = \
                f""" 
                <p><a href="https://github.com/junkunyuan/junkunyuan.github.io/blob/main/paper_reading_list/resource/jupyters/{paper["jupyter_notes"]}" class="note">(see notes in jupyter)</a></p>
                """
            details = paper["details"].replace("<img src='", f"<img src='resource/figs/{paper["name"]}/{paper["name"]}-")
            
            if "fig:" in details:
                details = convert_fig_cap_to_figure(details, paper["name"])

            author = f"""<p class="paper_detail">{paper["author"]}</p>""" if paper["author"] else ""
            organization = f"""<p class="paper_detail">{paper["organization"]}</p>""" if paper["organization"] else ""

            debug = ""
            if "new paper" in paper["title"].lower():
                debug = f"""<p>{paper["summary"]}</p><p>{details}</p>"""
                debug = debug.replace("data-src", "src")

            content_cate += \
            f"""
            <p class="little_split" id='{paper["name"]}'></p>
            <div style="border-left: 8px solid {color_bar}; padding-left: 10px">
            <div style="height: 0.3em;"></div>
            <p class="paper_title" onclick="toggleTable('{paper["name"]}-{category}-details')"><i>{paper["title"]}</i></p>
            {author}
            {organization}
            <p class="paper_detail"><b>{date} &nbsp; {paper["name"]}</b> {code} &nbsp;&nbsp;|&nbsp;&nbsp; {venue} &nbsp; <font color=#D0D0D0>{venue_all}</font></p>
            {comment}
            {debug}
            <div id='{paper["name"]}-{category}-details' class="info_detail">
                <hr class="dashed">
                <p><font color=#202020>{paper["summary"]}</font></p>
                {jupyter_note}
                <p>{details}</p>
            </div>
            <div style="height: 0.05em;"></div>
            </div>
            <p class="little_split"></p>
            """

        content_all_domain += content_cate
    content_all_domain += \
    """
    <script>
        function toggleTable(tableId) {
            const container = document.getElementById(tableId);
            const button = container.previousElementSibling;
            const isVisible = window.getComputedStyle(container).display !== 'none';
            if (!isVisible) {
                const images = container.querySelectorAll('img');
                images.forEach(img => {
                    if (img.dataset.src !== '') {
                        img.src = img.dataset.src;
                    }
                });
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
                
            }
        }
    </script>

    <button id="backToTop" title="back to top">↑</button>
    <script>
        const button = document.getElementById("backToTop");
        window.addEventListener("scroll", () => {
        if (document.documentElement.scrollTop > 300) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
        });

        function updateButtonPosition() {
        const bodyRect = document.body.getBoundingClientRect();
        const windowWidth = window.innerWidth;
        const rightOffset = Math.max((windowWidth - bodyRect.width) / 2, 10);
        button.style.right = rightOffset + "px";
        }

        window.addEventListener("resize", updateButtonPosition);
        window.addEventListener("load", updateButtonPosition);

        button.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    </script>
    """
    return catalog + content_all_domain

if __name__ == "__main__":
    intro_temp = INTRODUCTION.replace("time_now", time_now)
    
    ## Build html of all domains
    num_papers = list()
    for domain in DOMAINS:
        num_paper = 0 if domain["title"] == "Coding and Engineering" else len(domain["papers"]) 
        num_papers.append(num_paper)
        paper_num_display = "" if num_paper == 0 else f"<font color='#D93053'>{num_paper}</font> papers on "
        intro = intro_temp.replace("total_paper_num", paper_num_display)
        intro = intro.replace("title", domain["title"])
        intro = intro.replace("domain_name", domain["title"])
        papers_content = build_main_content_of_each_domain(domain)
        content_domain = PREFIX + intro + papers_content + SUFFIX
        with open(domain["file"], "w", encoding="utf-8-sig") as f:
            f.write(content_domain)
    
    ## Build main content
    intro = intro_temp.replace("total_paper_num", f"<font color='#D93053'>{sum(num_papers)}</font> papers on ")
    intro = intro.replace("title", MAIN_CONTENT["title"])
    intro = intro.replace("domain_name", MAIN_CONTENT["title"])
    intro = intro.replace("""<p>Click <a href="paper_reading_list.html">here</a> to go back to main contents.</p>""", "")
    content = build_main_content_all_domains(DOMAINS, num_papers)
    main_content = PREFIX + intro + content + SUFFIX
    with open(MAIN_CONTENT["file"], "w", encoding="utf-8-sig") as f:
        f.write(main_content)
    