"""HTML scaffolding shared between the main page and the reading-list pages.
`css_href` lets each generator point at the same stylesheet using its own
relative path."""

from __future__ import annotations

from datetime import datetime


def format_now() -> str:
    """Return a build-time timestamp like 'May 13, 2026 at 16:42 (UTC-7)'.
    Uses the build machine's local timezone."""
    dt = datetime.now().astimezone()
    date_part = dt.strftime("%B %d, %Y at %H:%M")
    offset = dt.utcoffset() or _zero_delta()
    total_min = int(offset.total_seconds() / 60)
    sign = "+" if total_min >= 0 else "-"
    hours, minutes = divmod(abs(total_min), 60)
    tz = f"UTC{sign}{hours}" if minutes == 0 else f"UTC{sign}{hours}:{minutes:02d}"
    return f"{date_part} ({tz})"


def _zero_delta():
    from datetime import timedelta
    return timedelta(0)


THEME_INIT_SCRIPT: str = (
    '    <script>'
    "(function(){try{var t=localStorage.getItem('theme');"
    "if(t==='light'||t==='dark')"
    "document.documentElement.setAttribute('data-theme',t);}catch(e){}})();"
    '</script>\n'
)


def page_head(title: str, css_href: str, favicon_href: str, extra: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="description" content="{title}">
    <link rel="stylesheet" href="{css_href}">
    <link rel="shortcut icon" href="{favicon_href}">
{THEME_INIT_SCRIPT}{extra}</head>
<body>
<div id="layout-content">
"""


PAGE_FOOT: str = "</div>\n</body>\n</html>\n"


READING_LIST_HEAD_EXTRA: str = (
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js" data-manual></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-bash.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-json.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/keep-markup/prism-keep-markup.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/toolbar/prism-toolbar.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/show-language/prism-show-language.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js"></script>\n'
    '    <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/line-highlight/prism-line-highlight.min.css" rel="stylesheet">\n'
    '    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/line-highlight/prism-line-highlight.min.js"></script>\n'
    '    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n'
    '    <script>document.addEventListener("DOMContentLoaded",function(){if(window.Prism)Prism.highlightAll();});</script>\n'
)


THEME_TOGGLE: str = """
<button id="themeToggle" type="button" aria-label="Toggle color theme" title="Theme: auto" data-mode="auto">
    <svg class="icon-auto" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4 A8 8 0 0 1 12 20 Z" fill="currentColor" stroke="none"/></svg>
    <svg class="icon-light" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
    <svg class="icon-dark" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" fill="currentColor" stroke="none"/></svg>
</button>
<script>
(function () {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    const order = ["auto", "light", "dark"];
    const labels = { auto: "Theme: follow system", light: "Theme: light", dark: "Theme: dark" };
    function read() {
        try { const v = localStorage.getItem("theme"); return (v === "light" || v === "dark") ? v : "auto"; }
        catch (e) { return "auto"; }
    }
    function apply(mode) {
        if (mode === "auto") {
            document.documentElement.removeAttribute("data-theme");
            try { localStorage.removeItem("theme"); } catch (e) {}
        } else {
            document.documentElement.setAttribute("data-theme", mode);
            try { localStorage.setItem("theme", mode); } catch (e) {}
        }
        btn.dataset.mode = mode;
        btn.title = labels[mode];
        btn.setAttribute("aria-label", labels[mode]);
    }
    apply(read());
    btn.addEventListener("click", () => {
        const cur = btn.dataset.mode || "auto";
        apply(order[(order.indexOf(cur) + 1) % order.length]);
    });
})();
</script>
"""


BACK_TO_TOP: str = """
<button id="backToTop" aria-label="Back to top" title="Back to top">↑</button>
<script>
(function () {
    const btn = document.getElementById("backToTop");
    if (!btn) return;
    window.addEventListener("scroll", () => {
        btn.classList.toggle("is-visible", document.documentElement.scrollTop > 300);
    });
    btn.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
})();
</script>
"""


ET_AL_SCRIPT: str = """
<script>
(function () {
    function toggle(uid) {
        document.querySelectorAll('[data-uid="' + uid + '"]').forEach(el => {
            if (el.classList.contains("et-al")) el.hidden = !el.hidden;
            else if (el.classList.contains("et-al-hidden")) el.hidden = !el.hidden;
        });
    }
    document.addEventListener("click", e => {
        const t = e.target.closest(".et-al, .et-al-hidden");
        if (t) toggle(t.dataset.uid);
    });
    document.addEventListener("keydown", e => {
        if (e.key !== "Enter" && e.key !== " ") return;
        const t = e.target.closest(".et-al");
        if (t) { e.preventDefault(); toggle(t.dataset.uid); }
    });
})();
</script>
"""


PAPER_TOGGLE_SCRIPT: str = """
<script>
function toggleTable(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("is-open");
}
</script>
"""


TOC_TOGGLE_SCRIPT: str = """
<script>
(function () {
    const header = document.getElementById("table");
    const content = document.getElementById("toc-content");
    if (!header || !content) return;
    const indicator = header.querySelector(".toc-indicator");
    function setOpen(open) {
        content.classList.toggle("is-open", open);
        if (indicator) indicator.textContent = open ? "▼" : "▶";
        header.setAttribute("aria-expanded", open ? "true" : "false");
    }
    setOpen(false);
    header.addEventListener("click", () => {
        setOpen(!content.classList.contains("is-open"));
    });
    header.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen(!content.classList.contains("is-open"));
        }
    });
})();
</script>
"""


SEARCH_SCRIPT: str = """
<input type="search" id="paperSearch" class="search"
       placeholder="Filter papers by title / author / venue / name…"
       aria-label="Filter papers">
<script>
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("paperSearch");
    if (!input) return;
    const papers = Array.from(document.querySelectorAll(".paper[data-search]"));
    input.addEventListener("input", () => {
        const q = input.value.trim().toLowerCase();
        papers.forEach(p => {
            const hit = !q || p.dataset.search.includes(q);
            p.classList.toggle("is-hidden", !hit);
        });
    });
});
</script>
"""
