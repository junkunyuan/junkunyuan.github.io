/* Behavior for every page. Each block does nothing when the element it is
   about is absent, so one file serves the home page and the reading lists.

   Loaded with `defer`; the only script that has to run earlier is the theme
   applier in the <head>, which would otherwise let a light flash through. */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- theme */
  var themeBtn = document.getElementById("themeToggle");
  if (themeBtn) {
    var order = ["auto", "light", "dark"];
    var labels = {
      auto: "Theme: follow system",
      light: "Theme: light",
      dark: "Theme: dark",
    };
    var read = function () {
      try {
        var v = localStorage.getItem("theme");
        return v === "light" || v === "dark" ? v : "auto";
      } catch (e) {
        return "auto";
      }
    };
    var apply = function (mode) {
      if (mode === "auto") {
        document.documentElement.removeAttribute("data-theme");
        try { localStorage.removeItem("theme"); } catch (e) {}
      } else {
        document.documentElement.setAttribute("data-theme", mode);
        try { localStorage.setItem("theme", mode); } catch (e) {}
      }
      themeBtn.dataset.mode = mode;
      themeBtn.title = labels[mode];
      themeBtn.setAttribute("aria-label", labels[mode]);
    };
    apply(read());
    themeBtn.addEventListener("click", function () {
      var cur = themeBtn.dataset.mode || "auto";
      apply(order[(order.indexOf(cur) + 1) % order.length]);
    });
  }

  /* ---------------------------------------------------------- back to top */
  var topBtn = document.getElementById("backToTop");
  if (topBtn) {
    window.addEventListener("scroll", function () {
      topBtn.classList.toggle("is-visible", document.documentElement.scrollTop > 300);
    });
    topBtn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* --------------------------------------------------------------- et al. */
  var toggleAuthors = function (uid) {
    document.querySelectorAll('[data-uid="' + uid + '"]').forEach(function (el) {
      if (el.classList.contains("et-al") || el.classList.contains("et-al-hidden")) {
        el.hidden = !el.hidden;
      }
    });
  };
  document.addEventListener("click", function (e) {
    var t = e.target.closest(".et-al, .et-al-hidden");
    if (t) toggleAuthors(t.dataset.uid);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" && e.key !== " ") return;
    var t = e.target.closest(".et-al");
    if (t) {
      e.preventDefault();
      toggleAuthors(t.dataset.uid);
    }
  });

  /* -------------------------------------------------- expand a note's body */
  document.addEventListener("click", function (e) {
    var title = e.target.closest(".paper--toggle .paper__title");
    if (!title) return;
    var body = title.closest(".paper").querySelector(".info_detail");
    if (body) body.classList.toggle("is-open");
  });

  /* --------------------------------------- open a note without clicking it */
  /* Two ways in, and neither touches the notes themselves. A link straight to a
     note opens it — the cross-page search produces exactly those links, and
     landing on a collapsed card meant clicking again to see what you searched
     for. And `?open=all`, or `?open=<name>`, opens notes while you are still
     writing them: the switch lives in the address bar, so there is nothing to
     remember to take out before publishing. */
  var openNote = function (paper) {
    var body = paper && paper.querySelector(".info_detail");
    if (body) body.classList.add("is-open");
  };

  var openFromHash = function () {
    if (!location.hash) return;
    var el;
    try { el = document.querySelector(location.hash); } catch (e) { return; }
    var paper = el && el.closest(".paper");
    if (!paper) return;
    openNote(paper);
    paper.scrollIntoView();
  };

  var openFromQuery = function () {
    var m = /[?&]open=([^&]*)/.exec(location.search);
    if (!m) return;
    var want = decodeURIComponent(m[1]).trim().toLowerCase();
    document.querySelectorAll(".paper--toggle").forEach(function (p) {
      var n = p.querySelector(".paper__name");
      if (want === "all" || (n && n.textContent.trim().toLowerCase() === want)) openNote(p);
    });
  };

  openFromQuery();
  openFromHash();
  window.addEventListener("hashchange", openFromHash);

  /* ---------------------------------------------------- table of contents */
  var tocHead = document.getElementById("table");
  var tocBody = document.getElementById("toc-content");
  if (tocHead && tocBody) {
    var indicator = tocHead.querySelector(".toc-indicator");
    var setOpen = function (open) {
      tocBody.classList.toggle("is-open", open);
      if (indicator) indicator.textContent = open ? "▼" : "▶";
      tocHead.setAttribute("aria-expanded", open ? "true" : "false");
    };
    setOpen(false);
    tocHead.addEventListener("click", function () {
      setOpen(!tocBody.classList.contains("is-open"));
    });
    tocHead.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(!tocBody.classList.contains("is-open"));
      }
    });
  }

  /* ---------------------------------------------------------------- filter */
  /* Types on this page's notes, and lists what matched on the other pages —
     the index is a script (window.NOTE_INDEX), because a page opened from the
     filesystem is not allowed to fetch a sibling file. */
  /* Two kinds of term: a bare word, which matches anywhere, and `field:value`,
     which matches only that field. Every term has to match — typing three tags
     narrows, it does not widen. Field names map to the single-letter markers
     `_search_blob` writes into data-search. */
  var FIELD = { title: "t", author: "a", org: "o", organization: "o",
                venue: "v", name: "n", cat: "c", category: "c", tag: "g" };

  var queryTerms = function (q) {
    return q.toLowerCase().split(/\s+/).filter(Boolean).map(function (term) {
      var m = /^([a-z]+):(.+)$/.exec(term);
      return m && FIELD[m[1]] ? { field: FIELD[m[1]], value: m[2] } : { value: term };
    });
  };

  /* A field can appear more than once — a paper wears several `|g:` tags — so
     collect every segment and let any of them satisfy the term. Segments run to
     the next "|", which is why the generator keeps that character out of values. */
  var fieldSegments = function (blob, letter) {
    var out = [], re = new RegExp("\\|" + letter + ":([^|]*)", "g"), m;
    while ((m = re.exec(blob))) out.push(m[1]);
    return out;
  };

  var matchesAll = function (blob, terms) {
    return terms.every(function (t) {
      if (!t.field) return blob.indexOf(t.value) !== -1;
      return fieldSegments(blob, t.field).some(function (seg) {
        return seg.indexOf(t.value) !== -1;
      });
    });
  };

  var search = document.getElementById("paperSearch");
  if (search) {
    var papers = Array.prototype.slice.call(
      document.querySelectorAll(".paper[data-search]")
    );
    var here = new Set(papers.map(function (p) { return p.id; }));
    var elsewhere = document.createElement("div");
    elsewhere.className = "search-elsewhere";
    search.parentNode.insertBefore(elsewhere, search.nextSibling);

    var page = location.pathname.split("/").pop();
    search.addEventListener("input", function () {
      var q = search.value.trim().toLowerCase();
      var terms = queryTerms(q);
      papers.forEach(function (p) {
        p.classList.toggle("is-hidden", !matchesAll(p.dataset.search, terms));
      });

      elsewhere.innerHTML = "";
      if (q.length < 2 || !window.NOTE_INDEX) return;
      var hits = window.NOTE_INDEX.filter(function (r) {
        return matchesAll(r.s, terms) && r.u.split("#")[0] !== page;
      });
      if (!hits.length) {
        if (!papers.length) {
          var none = document.createElement("p");
          none.className = "search-elsewhere__head";
          none.textContent = "No note matches that.";
          elsewhere.appendChild(none);
        }
        return;
      }
      /* The contents page has no notes of its own, so this list is the whole
         result rather than an aside — say so, and show more of it. */
      var global = papers.length === 0;
      var limit = global ? 60 : 24;
      var head = document.createElement("p");
      head.className = "search-elsewhere__head";
      head.textContent = global
        ? hits.length + (hits.length === 1 ? " note matches:" : " notes match:")
        : hits.length + " more on other pages:";
      elsewhere.appendChild(head);
      hits.slice(0, limit).forEach(function (r) {
        var a = document.createElement("a");
        a.href = r.u;
        a.className = "search-elsewhere__hit";
        a.innerHTML =
          "<b>" + r.n + '</b> <span class="venue">(' + r.v + ")</span> — " + r.d;
        elsewhere.appendChild(a);
      });
      if (hits.length > limit) {
        var more = document.createElement("p");
        more.className = "search-elsewhere__head";
        more.textContent = "…and " + (hits.length - limit) + " more.";
        elsewhere.appendChild(more);
      }
    });
  }

  /* ------------------------------------------------------ copy a code block */
  /* Prism's toolbar used to provide this; the code is highlighted at build
     time now, so the button is ours. */
  document.querySelectorAll("pre.highlight").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-copy";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var text = pre.querySelector("code").innerText;
      var done = function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1200);
      };
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(done, function () {});
        return;
      }
      var ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); done(); } catch (e) {}
      document.body.removeChild(ta);
    });
    pre.appendChild(btn);
  });
})();
