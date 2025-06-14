import pandas as pd
from datetime import datetime

domains = dict()

## --------------------------------------------------------------------------------
## Visual Generative Models
## --------------------------------------------------------------------------------
visual_generative_models = dict()
visual_generative_models["file"] = "visual_generative_models.html"
visual_generative_models["prefix"] = \
    """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="jemdoc_reading_papers.css" type="text/css">
        <link rel="shortcut icon" href="../resource/citations.jpg">
        <title>JunkunYuan's Reading Papers</title>
        <meta name="description" content="Junkun Yuan&#39;s Reading Papers">
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <div id="layout-content" style="margin-top:25px"></div>
    </head>
    <body>
    """
visual_generative_models["title"] = \
    f"""
    <h1 id="top">Visual Generative Models</h1>
    <b><font size=3>Reading papers on visual generative models. Curated by Junkun Yuan (yuanjk0921@outlook.com)</font></b>
    <br><br>
    Last updated on {datetime.now().strftime('%A, %B %d %Y at %I:%M %p')}.
    <br><br>
    <a href="reading_papers.html">[back to main contents]</a>
    <br>
    <hr>
    """
visual_generative_models["suffix"] = \
    """
    </body>
    </html>
    """
visual_generative_models["categories"] = {
    "Survey & Insight": "survey-and-insight",
    "Foundation Model & Algorithm": "foundation-model-and-algorithm",
    "Fine-Tuning": "finetuning",
    "Reinforcement Learning": "reinforcement-learning",
    "Acceleration": "acceleration",
    "Inference-Time Improvement": "inference-time-improvement",
    "Downstream Task": "downstream_task",
    "Evaluation": "evaluation"
}
domains["visual_generative_models"] = visual_generative_models
## --------------------------------------------------------------------------------

def build_main_content(df, categories):
    table_of_content = [f"""<li><a href="#{c_id}">{c}</a></li>""" for c, c_id in categories.items()]
    table_of_content = "".join(table_of_content)

    cate_content = ""
    for category, category_id in categories.items():
        main_content_cate = f"""<h3 id="{category_id}">{category}</h3>"""
        
        df_cate = df[df["category"].str.contains(category)]
        df_cate = df_cate.sort_values(by="date", ascending=False)

        items = ""
        for _, item in df_cate.iterrows():
            if isinstance(item["project"], str) and item["project"].startswith("https://github.com/"):
                project_name = item["project"].split("https://github.com/")[1]
                project_name = project_name[:-1] if project_name.endswith("/") else project_name
                project = \
                f"""
                <a href="{item["project"]}"><img src="https://img.shields.io/github/stars/{project_name}.svg?style=social&label=Star" alt="Star" style="vertical-align: middle;" /></a>
                """
            else:
                project = ""
            if isinstance(item["jupyter_note"], str) and ".ipynb" in item["jupyter_note"]:
                jupyter_note = \
                f""" 
                <br><a href="https://github.com/junkunyuan/junkunyuan.github.io/blob/main/reading_papers/jupyters/{item["jupyter_note"]}" class="note">(notes in jupyter)</a>
                """
            else:
                jupyter_note = ""
            items += \
                f"""
                <tr>
                    <td>
                        <a id="{item["date"]}-{item["model"].replace("<br>", "--")}-item" class="date">{item["date"]}</a>
                        <br><a href="#{item["date"]}-{item["model"].replace("<br>", "--")}" class="model">
                            {item["model"]}
                            {jupyter_note}
                        </a>
                    </td>
                    <td>
                        <a href="{item["paper_url"]}"><span class="paper">
                            {item["paper"]} <b><i>({item["publication"]})</i></b>
                        </span></a>
                        {project}
                    </td>
                    <td><span class="summary">
                        {item["summary"]}
                    </span></td>
                </tr>
                """      
        main_content_cate += \
            f"""
            <table>
                <colgroup>
                    <col style="width: 16%;">
                    <col style="width: 36%;">
                    <col style="width: 48%;">
                </colgroup>
                <thead>
                    <tr>
                        <th>Date & Model</th>
                        <th>Paper & Publication & Project</th>
                        <th>Summary</th>
                    </tr>
                </thead>
                <tbody>
                    {items}
                </tbody>
            </table>
            <a href="#top">[back to top]</a>
            <br><br>
            <hr>
            """
        cate_content += main_content_cate

    main_content = \
    f"""
    <b>Contents:</b>
    <ul>
        <li>
            <a href="#summary">Summary</a>
            <ul>
                {table_of_content}
            </ul>
        </li>
        <li><a href="#papers">Papers & Reading Notes</a></li>
    </ul>
    <h2 id="summary">Summary</h2>
    {cate_content}
    """

    return main_content

def build_details(df):
    df = df.sort_values(by="date", ascending=False)
    detail_items = ""
    all_items = len(df)
    for idx, item in df.iterrows():
        if isinstance(item["detail"], str) and len(item["detail"]) > 5:
            detail = \
                f"""
                <button onclick="toggleTable('{item["date"]}-{item["model"].replace("<br>", "--")}-table')">Read more</button>
                <div id='{item["date"]}-{item["model"].replace("<br>", "--")}-table' class="table-container">
                {item["detail"]}
                </div>
                """
        else:
            detail = ""
        
        if isinstance(item["jupyter_note"], str) and ".ipynb" in item["jupyter_note"]:
            jupyter_note = f"""<p><a href="https://github.com/junkunyuan/junkunyuan.github.io/blob/main/reading_papers/jupyters/{item["jupyter_note"]}" class="note">(notes in jupyter)</a></p>"""
        else:
            jupyter_note = ""
        
        if isinstance(item["project"], str) and item["project"].startswith("https://github.com/"):
                project_name = item["project"].split("https://github.com/")[1]
                project_name = project_name[:-1] if project_name.endswith("/") else project_name
                project = \
                f"""
                <a href="{item["project"]}"><img src="https://img.shields.io/github/stars/{project_name}.svg?style=social&label=Star" alt="Star" style="vertical-align: middle;" /></a>
                """
        else:
            project = ""

        detail_items += \
            f"""
            <h4 id="{item["date"]}-{item["model"].replace("<br>", "--")}">
            [{all_items - idx}] &nbsp;{item["paper"]} <i>({item["publication"]})</i>
            </h4>
            {jupyter_note}
            <p><i><b>Date:</b></i> {item["date"]}</p>
            <p><i><b>Project:</b></i> {item["model"].replace("<br>", ", ")} {project}</p>
            <p><i><b>Authors:</b></i> {item["author"]}</p>
            <p><i><b>Organizations:</b></i> {item["organization"]}</p>
            <p><i><b>Summary:</b></i> {item["summary"]}</p>
            {detail}
            <p><a href="#{item["date"]}-{item["model"].replace("<br>", "--")}-item">[back to item]</a> &nbsp; <a href="#top">[back to top]</a> </p>
            <hr>
            """
    details = """<h2 id="papers">Papers & Reading Notes</h2>""" + detail_items + \
        """
        <script>
            function toggleTable(tableId) {
                const container = document.getElementById(tableId);
                const button = container.previousElementSibling;
                const isVisible = window.getComputedStyle(container).display !== 'none';
                
                if (!isVisible) {
                    const images = container.querySelectorAll('.lazy-load');
                    images.forEach(img => {
                        if (!img.src && img.dataset.src) {
                            img.src = img.dataset.src;
                        }
                    });
                    container.style.display = 'block';
                    button.textContent = 'Close';
                } else {
                    container.style.display = 'none';
                    button.textContent = 'Read more';
                }
            }
        </script>
        """
    return details

if __name__ == "__main__":
    for domain_name, domain in domains.items():
        ## Load paper pool
        df = pd.read_csv(f"{domain_name}.csv")
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

        main_content = build_main_content(df, domain["categories"])
        details = build_details(df)
        html_content = domain["prefix"] + domain["title"].replace("Reading papers", f"<font color='#D93053'>{len(df)}</font> papers") + main_content + details + domain["suffix"]
        with open(domain["file"], 'w', encoding="utf-8-sig") as f: 
            f.write(html_content)
    