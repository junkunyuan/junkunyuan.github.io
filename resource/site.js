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
    // Open by default (2026-11-02). A lineage figure inside cannot measure its
    // boxes while the contents are hidden, so opening redraws its arrows.
    var setOpen = function (open) {
      tocBody.classList.toggle("is-open", open);
      if (indicator) indicator.textContent = open ? "▼" : "▶";
      tocHead.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) window.dispatchEvent(new Event("resize"));
    };
    setOpen(true);
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

  /* ------------------------------------------------------ lineage figure */
  /* The build lays the labels out in a grid (sitegen/lineage.py): a column per
     research line, a row per venue year, oldest at the top. The arrows are
     drawn here from the boxes the browser set, so they follow the real font
     and redraw when the width changes. They run only through the empty
     channels of the grid — vertical runs in the gaps between columns,
     horizontal runs in the gaps between rows — so no line crosses a label:
       - a successor in its parent's own cell is indented under it and joined
         by a tree branch (a second parent in the cell, with labels between,
         goes round through the channels instead);
       - a successor further down the same column gets a straight arrow down;
       - any other arrow leaves the parent's right edge into the gap after its
         column, runs along a gap between two rows to the gap before the
         successor's column, and enters its left edge;
       - concurrent papers share a dashed frame, and an arrow to a successor of
         any of them leaves the frame, once.
     Lines should not cross each other either, so the routing searches: each
     arrow takes the row gap where it crosses fewest other lines (then the
     shortest), and runs sharing a gap are parallel in the order that crosses
     least. Arrows of one parent share their runs and branch near each
     successor; arrows of several parents into one paper share its last run,
     so they merge before it. Where a crossing no route avoids remains, the longer line breaks
     around it, so it reads as passing, not as a junction. Hovering a paper
     lights its own arrows and fades the rest. */
  var NS = "http://www.w3.org/2000/svg";
  var GAP = 3;  // half the break a line leaves where another crosses it
  // A polyline with rounded corners; leg i breaks around every point in gaps[i]
  // (an x on a horizontal leg, a y on a vertical one).
  var rounded = function (pts, gaps) {
    var d = "M" + pts[0][0] + "," + pts[0][1];
    var leg = function (a, b, i, x, y) {
      var across = Math.abs(a[1] - b[1]) < 0.5;
      var dir = across ? Math.sign(b[0] - a[0]) : Math.sign(b[1] - a[1]);
      ((gaps && gaps[i]) || []).slice().sort(function (u, v) { return dir * (u - v); })
        .forEach(function (g) {
          d += across ? " L" + (g - dir * GAP) + "," + a[1] + " M" + (g + dir * GAP) + "," + a[1]
                      : " L" + a[0] + "," + (g - dir * GAP) + " M" + a[0] + "," + (g + dir * GAP);
        });
      d += " L" + x + "," + y;
    };
    for (var i = 1; i < pts.length - 1; i++) {
      var p = pts[i - 1], c = pts[i], n = pts[i + 1];
      var r = Math.min(6, (Math.abs(c[0] - p[0]) + Math.abs(c[1] - p[1])) / 2,
                       (Math.abs(n[0] - c[0]) + Math.abs(n[1] - c[1])) / 2);
      var ux = Math.sign(c[0] - p[0]), uy = Math.sign(c[1] - p[1]);
      var vx = Math.sign(n[0] - c[0]), vy = Math.sign(n[1] - c[1]);
      leg(p, c, i - 1, c[0] - ux * r, c[1] - uy * r);
      d += " Q" + c[0] + "," + c[1] + " " + (c[0] + vx * r) + "," + (c[1] + vy * r);
    }
    var last = pts[pts.length - 1];
    leg(pts[pts.length - 2], last, pts.length - 2, last[0], last[1]);
    return d;
  };
  var permutations = function (a) {
    if (a.length <= 1) return [a];
    var out = [];
    a.forEach(function (x, i) {
      permutations(a.slice(0, i).concat(a.slice(i + 1))).forEach(function (r) {
        out.push([x].concat(r));
      });
    });
    return out;
  };
  document.querySelectorAll(".lineage").forEach(function (fig) {
    var canvas = fig.querySelector(".lineage__canvas");
    var svg = fig.querySelector(".lineage__edges");
    var edges = JSON.parse(fig.getAttribute("data-edges") || "[]");
    // Hovered: the paper under the pointer and the concurrent frame around it.
    var hovered = [];
    // The plan — each arrow's row gap and each channel's order — depends on the
    // grid's rows and columns, not on its width: it is searched once and reused
    // on every resize, which then only re-measures.
    var plan = null;
    var mark = function () {
      fig.classList.toggle("lineage--focus", hovered.length > 0);
      var near = {};
      hovered.forEach(function (id) { near[id] = true; });
      svg.querySelectorAll("path.lineage__edge").forEach(function (p) {
        var on = hovered.indexOf(p.dataset.from) >= 0 || hovered.indexOf(p.dataset.to) >= 0;
        p.classList.toggle("is-on", on);
        if (on) { near[p.dataset.from] = true; near[p.dataset.to] = true; }
      });
      canvas.querySelectorAll("[data-node]").forEach(function (el) {
        var group = el.parentNode.closest(".lineage__group");
        var on = !!near[el.getAttribute("data-node")];
        el.classList.toggle("is-on", on || !!(group && near[group.getAttribute("data-node")]));
      });
    };
    canvas.addEventListener("mouseover", function (ev) {
      var next = [ev.target.closest("[data-node]"), ev.target.closest(".lineage__group")]
        .filter(Boolean).map(function (el) { return el.getAttribute("data-node"); });
      if (next.join() !== hovered.join()) { hovered = next; mark(); }
    });
    canvas.addEventListener("mouseleave", function () { hovered = []; mark(); });
    var draw = function () {
      if (!canvas.offsetWidth) return;  // hidden: nothing to measure, so no plan either
      var box = canvas.getBoundingClientRect();
      var rel = function (el) {
        var r = el.getBoundingClientRect();
        return { l: r.left - box.left, r: r.right - box.left, t: r.top - box.top,
                 b: r.bottom - box.top, y: (r.top + r.bottom) / 2 - box.top };
      };
      // Column spans (c = -1 is the year column) and row spans of the grid.
      var colL = {}, colR = {}, rowT = {}, rowB = {};
      canvas.querySelectorAll("[data-r]").forEach(function (cell) {
        var q = rel(cell), r = +cell.dataset.r, c = +cell.dataset.c;
        colL[c] = colL[c] === undefined ? q.l : Math.min(colL[c], q.l);
        colR[c] = colR[c] === undefined ? q.r : Math.max(colR[c], q.r);
        if (cell.dataset.span) return;  // a frame across rows tells nothing of one row
        rowT[r] = rowT[r] === undefined ? q.t : Math.min(rowT[r], q.t);
        rowB[r] = rowB[r] === undefined ? q.b : Math.max(rowB[r], q.b);
      });
      var rows = Object.keys(rowT).map(Number);
      var lastRow = Math.max.apply(null, rows);
      // The usable span of a channel: the gap after column c keeps 9px free on
      // its right for the arrowhead into the successor.
      var gapSpan = function (c) {
        var right = colL[c + 1] === undefined ? colR[c] + 20 : colL[c + 1];
        return [colR[c] + 3, right - 9];
      };
      var rowSpan = function (k) { return [rowB[k] + 3, rowT[k + 1] - 3]; };
      var labels = [].map.call(canvas.querySelectorAll("[data-node]"), function (el) {
        var q = rel(el), cell = el.closest("[data-r]");
        var name = el.querySelector(".lineage__name");  // a frame: its first paper's
        if (name) q.y = rel(name).y;  // the name line, not the counts under it
        q.el = el; q.row = +cell.dataset.r; q.col = +cell.dataset.c;
        return q;
      });
      var find = function (id) {
        return labels.filter(function (q) { return q.el.getAttribute("data-node") === id; })[0];
      };
      var clearDown = function (x, y1, y2, skip) {
        return labels.every(function (q) {
          var inside = skip.some(function (k) { return k.el !== q.el && k.el.contains(q.el); });
          return skip.indexOf(q) >= 0 || inside || q.l > x || q.r < x || q.b < y1 || q.t > y2;
        });
      };
      svg.setAttribute("width", canvas.offsetWidth);
      svg.setAttribute("height", canvas.offsetHeight);
      svg.querySelectorAll("path.lineage__edge").forEach(function (p) { p.remove(); });

      // The routes: fixed ones inside a column, and ones through the channels.
      var routes = [];
      edges.forEach(function (e) {
        var p = find(e[0]), s = find(e[1]);
        if (!p || !s) return;
        var rt = { e: e, p: p, s: s }, x = p.l + 9;
        if (p.row === s.row && p.col === s.col && clearDown(x, p.b, s.y, [p, s])) {
          rt.pts = [[x, p.b], [x, s.y], [s.l - 1, s.y]];
        } else if (p.col === s.col && p.row !== s.row && clearDown(x, p.b, s.t, [p, s])) {
          rt.pts = s.l > x + 3 ? [[x, p.b], [x, s.y], [s.l - 1, s.y]] : [[x, p.b], [x, s.t - 1]];
        } else {
          rt.g1 = p.col;                    // the gap after the parent's column
          rt.g2 = s.col - 1;                // the gap before the successor's
          rt.direct = rt.g1 === rt.g2;      // neighbors: one vertical run does it
        }
        routes.push(rt);
      });
      var flexible = routes.filter(function (rt) { return !rt.pts; });

      // Parallel runs: each channel holds its parents in an order.
      var order = {};
      var at = function (key, src, span) {
        var list = order[key] || [src], i = Math.max(0, list.indexOf(src));
        var step = list.length > 1 ? Math.min(5, (span[1] - span[0]) / (list.length - 1)) : 0;
        return (span[0] + span[1]) / 2 + (i - (list.length - 1) / 2) * step;
      };
      // A run in a channel belongs to a parent on the way out ("s:") and to a
      // successor on the way in ("d:"): every arrow into one paper shares its
      // last vertical run, so arrows to a common successor merge there, and
      // the figure carries one line into each paper, not one per parent.
      // The last run into a paper belongs to the paper when several parents
      // reach it through that gap — they merge there — and to the parent when it
      // is the only one: then one parent's arrows to several papers of a column
      // share one run down and branch off near each, as a tree does.
      var inbound = {};
      flexible.forEach(function (rt) {
        var k = (rt.direct ? rt.g1 : rt.g2) + "|" + rt.e[1];
        (inbound[k] = inbound[k] || {})[rt.e[0]] = true;
      });
      flexible.forEach(function (rt) {  // settled once: shape() runs thousands of times
        var k = (rt.direct ? rt.g1 : rt.g2) + "|" + rt.e[1];
        rt.into = Object.keys(inbound[k]).length > 1 ? "d:" + rt.e[1] : "s:" + rt.e[0];
      });
      var entry = function (rt) { return rt.into; };
      var runs = function (rt) {
        var out = "s:" + rt.e[0], into = entry(rt);
        return rt.direct ? [["g" + rt.g1, into]]
                         : [["g" + rt.g1, out], ["k" + rt.k, out], ["g" + rt.g2, into]];
      };
      var shape = function (rt) {
        if (rt.pts) return rt.pts;
        var p = rt.p, s = rt.s, into = entry(rt), out = "s:" + rt.e[0];
        if (rt.direct) {
          var x = at("g" + rt.g1, into, gapSpan(rt.g1));
          return [[p.r + 2, p.y], [x, p.y], [x, s.y], [s.l - 1, s.y]];
        }
        var x1 = at("g" + rt.g1, out, gapSpan(rt.g1));
        var x2 = at("g" + rt.g2, into, gapSpan(rt.g2));
        var y = at("k" + rt.k, out, rowSpan(rt.k));
        return [[p.r + 2, p.y], [x1, p.y], [x1, y], [x2, y], [x2, s.y], [s.l - 1, s.y]];
      };
      var segments = function (rt) {
        var pts = shape(rt), out = [];
        for (var i = 0; i + 1 < pts.length; i++) out.push({ rt: rt, a: pts[i], b: pts[i + 1], i: i });
        return out;
      };
      // Where two lines of different parents and different successors cross;
      // lines that share an end meet there, which is a junction, not a crossing.
      // Two such lines running along each other count too: they read as one.
      var crossing = function (u, v) {
        if (u.rt.e[0] === v.rt.e[0] || u.rt.e[1] === v.rt.e[1]) return null;
        var uh = Math.abs(u.a[1] - u.b[1]) < 0.5, vh = Math.abs(v.a[1] - v.b[1]) < 0.5;
        if (uh === vh) {
          var k = uh ? 1 : 0, along = uh ? 0 : 1;
          if (Math.abs(u.a[k] - v.a[k]) > 1.5) return null;
          var lo = Math.max(Math.min(u.a[along], u.b[along]), Math.min(v.a[along], v.b[along]));
          var hi = Math.min(Math.max(u.a[along], u.b[along]), Math.max(v.a[along], v.b[along]));
          return hi - lo > 1 ? { overlap: true } : null;
        }
        var h = uh ? u : v, w = uh ? v : u, x = w.a[0], y = h.a[1];
        var inside = x > Math.min(h.a[0], h.b[0]) + 0.5 && x < Math.max(h.a[0], h.b[0]) - 0.5 &&
                     y > Math.min(w.a[1], w.b[1]) + 0.5 && y < Math.max(w.a[1], w.b[1]) - 0.5;
        return inside ? { h: h, x: x, y: y } : null;
      };
      // What the eye counts: arrows of one parent run together, so where they
      // cross a line together it is one crossing, not one per arrow.
      var spots = function (pairs) {
        var seen = {}, n = 0;
        pairs.forEach(function (c) {
          var key = c.overlap ? "o" + n : Math.round(c.x) + "," + Math.round(c.y);
          if (!seen[key]) { seen[key] = true; n++; }
        });
        return n;
      };
      // The rest of the figure, sorted for bisection: horizontal lines by height,
      // vertical ones by x, and both by their coordinate for lines on top of each other.
      var index = function (segs) {
        var across = [], down = [], same = {};
        segs.forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5;
          (h ? across : down).push(u);
          var k = (h ? "h" : "v") + Math.round(h ? u.a[1] : u.a[0]);
          (same[k] = same[k] || []).push(u);
        });
        across.sort(function (u, v) { return u.a[1] - v.a[1]; });
        down.sort(function (u, v) { return u.a[0] - v.a[0]; });
        return { all: segs, across: across, down: down, same: same,
                 ys: across.map(function (u) { return u.a[1]; }),
                 xs: down.map(function (u) { return u.a[0]; }) };
      };
      var bisect = function (list, v) {
        var lo = 0, hi = list.length;
        while (lo < hi) { var m = (lo + hi) >> 1; if (list[m] <= v) lo = m + 1; else hi = m; }
        return lo;
      };
      // The crossings one arrow makes with an indexed rest of the figure.
      var against = function (mine, idx) {
        var found = [];
        mine.forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5;
          var lo = Math.min(h ? u.a[0] : u.a[1], h ? u.b[0] : u.b[1]);
          var hi = Math.max(h ? u.a[0] : u.a[1], h ? u.b[0] : u.b[1]);
          var keys = h ? idx.xs : idx.ys, list = h ? idx.down : idx.across;
          for (var i = bisect(keys, lo); i < keys.length && keys[i] < hi; i++) {
            var c = crossing(u, list[i]);
            if (c) found.push(c);
          }
          (idx.same[(h ? "h" : "v") + Math.round(h ? u.a[1] : u.a[0])] || []).forEach(function (v) {
            var c = crossing(u, v);
            if (c) found.push(c);
          });
        });
        return spots(found);
      };
      var countBetween = function (a, b) {
        var found = [];
        a.forEach(function (u) { b.forEach(function (v) { var c = crossing(u, v); if (c) found.push(c); }); });
        return spots(found);
      };
      var all = function () {
        var segs = [];
        routes.forEach(function (rt) { if (rt.pts || rt.direct || rt.k !== undefined) segs = segs.concat(segments(rt)); });
        return segs;
      };
      // All crossings of the figure: each vertical line is checked only against
      // the horizontal lines whose height lies within it (sorted, bisected), and
      // lines of one orientation only against those on the same coordinate.
      // A crossing is a junction, not a crossing, where an arrow from one line's
      // parent to the other line's successor bends: there the first line feeds
      // the second, as the frame's arrow to Shortcut Models feeds Consistency
      // Models' arrow while the frame's trunk runs on.
      var turns = {};
      var junction = function (c, u, v) {
        var at = turns[Math.round(c.x) + "," + Math.round(c.y)];
        return !!at && (at[u.rt.e[0] + ">" + v.rt.e[1]] || at[v.rt.e[0] + ">" + u.rt.e[1]]);
      };
      var total = function () {
        var segs = all(), across = [], down = [], same = {}, found = [];
        turns = {};
        segs.forEach(function (u) {
          var k = Math.round(u.b[0]) + "," + Math.round(u.b[1]);
          (turns[k] = turns[k] || {})[u.rt.e[0] + ">" + u.rt.e[1]] = true;
        });
        segs.forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5;
          (h ? across : down).push(u);
          var k = (h ? "h" : "v") + Math.round(h ? u.a[1] : u.a[0]);
          (same[k] = same[k] || []).push(u);
        });
        across.sort(function (u, v) { return u.a[1] - v.a[1]; });
        var ys = across.map(function (u) { return u.a[1]; });
        down.forEach(function (v) {
          var y1 = Math.min(v.a[1], v.b[1]), y2 = Math.max(v.a[1], v.b[1]), lo = 0, hi = ys.length;
          while (lo < hi) { var m = (lo + hi) >> 1; if (ys[m] <= y1) lo = m + 1; else hi = m; }
          for (var i = lo; i < ys.length && ys[i] < y2; i++) {
            var c = crossing(across[i], v);
            if (c && !junction(c, across[i], v)) found.push(c);
          }
        });
        Object.keys(same).forEach(function (k) {
          var g = same[k];
          for (var i = 0; i < g.length; i++)
            for (var j = i + 1; j < g.length; j++) {
              var c = crossing(g[i], g[j]);
              if (c) found.push(c);
            }
        });
        return spots(found);
      };
      // The ink of the whole figure: the length of every line, where lines lie
      // on top of each other counted once — a trunk shared costs nothing more.
      var ink = function () {
        var lines = {};
        all().forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5, at = Math.round(h ? u.a[1] : u.a[0]);
          var lo = Math.min(h ? u.a[0] : u.a[1], h ? u.b[0] : u.b[1]);
          var hi = Math.max(h ? u.a[0] : u.a[1], h ? u.b[0] : u.b[1]);
          (lines[(h ? "h" : "v") + at] = lines[(h ? "h" : "v") + at] || []).push([lo, hi]);
        });
        var n = 0;
        Object.keys(lines).forEach(function (k) {
          var end = -Infinity;
          lines[k].sort(function (a, b) { return a[0] - b[0]; }).forEach(function (iv) {
            if (iv[1] > end) { n += iv[1] - Math.max(iv[0], end); end = iv[1]; }
          });
        });
        return n;
      };

      // Two steps, repeated: (1) a row gap for every arrow that crosses columns —
      // fewest conflicts with the lines already there, then shortest; (2) the
      // order of parents in every shared channel — all orders at once while
      // there are few, else one channel at a time.
      var gaps = rows.filter(function (k) { return k < lastRow; });
      var yOf = function (run) { return find(run.slice(2)).y; };  // the paper a run belongs to
      var big = flexible.length > 20;  // a large figure gets a cheaper search
      // Where a crossing remains, the longer of the two lines breaks around it,
      // so the shorter reads as passing through. A line here is everything lying
      // on that spot — a trunk of several arrows breaks as one — and its length
      // is that of its longest arrow. A break keeps clear of corners; with no
      // room on the longer line the shorter one breaks instead, which the search
      // counts as awkward and avoids where it can.
      var span = function (pts) {
        var n = 0;
        for (var i = 0; i + 1 < pts.length; i++)
          n += Math.abs(pts[i + 1][0] - pts[i][0]) + Math.abs(pts[i + 1][1] - pts[i][1]);
        return n;
      };
      var roomy = function (u, at) {
        var ax = Math.abs(u.a[1] - u.b[1]) < 0.5 ? 0 : 1;
        return at - Math.min(u.a[ax], u.b[ax]) >= GAP + 6 && Math.max(u.a[ax], u.b[ax]) - at >= GAP + 6;
      };
      // Every crossing pair, found as total() finds them: a vertical line against
      // the horizontal ones within its height, sorted and bisected.
      var pairs = function (segs) {
        var across = [], down = [], found = [];
        segs.forEach(function (u) { (Math.abs(u.a[1] - u.b[1]) < 0.5 ? across : down).push(u); });
        across.sort(function (u, v) { return u.a[1] - v.a[1]; });
        var ys = across.map(function (u) { return u.a[1]; });
        down.forEach(function (v) {
          var y1 = Math.min(v.a[1], v.b[1]), y2 = Math.max(v.a[1], v.b[1]), lo = 0, hi = ys.length;
          while (lo < hi) { var m = (lo + hi) >> 1; if (ys[m] <= y1) lo = m + 1; else hi = m; }
          for (var i = lo; i < ys.length && ys[i] < y2; i++) {
            var c = crossing(across[i], v);
            if (c) { c.w = v; found.push(c); }
          }
        });
        return found;
      };
      var breaks = function () {
        var segs = all(), gaps = new Map(), awkward = 0, seen = {}, lengths = new Map();
        var line = {}, turned = {};  // segments by orientation and coordinate; where arrows bend
        segs.forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5, k = (h ? "h" : "v") + Math.round(h ? u.a[1] : u.a[0]);
          (line[k] = line[k] || []).push(u);
          if (!lengths.has(u.rt)) lengths.set(u.rt, span(shape(u.rt)));
          var b = Math.round(u.b[0]) + "," + Math.round(u.b[1]);
          (turned[b] = turned[b] || {})[u.rt.e[0] + ">" + u.rt.e[1]] = true;
        });
        var through = function (across, x, y) {  // the segments of one orientation over (x, y)
          var at = across ? y : x, along = across ? x : y, ax = across ? 0 : 1;
          return (line[(across ? "h" : "v") + Math.round(at)] || []).filter(function (u) {
            return along > Math.min(u.a[ax], u.b[ax]) && along < Math.max(u.a[ax], u.b[ax]);
          });
        };
        var longest = function (l) {
          return Math.max.apply(null, l.map(function (u) { return lengths.get(u.rt); }));
        };
        pairs(segs).forEach(function (c) {
          var spot = Math.round(c.x) + "," + Math.round(c.y);
          if (seen[spot]) return;
          seen[spot] = true;
          var across = [through(true, c.x, c.y), c.x], down = [through(false, c.x, c.y), c.y];
          // Where one of the lines also branches into the other — an arrow from a
          // parent on one line to a successor on the other — the spot is a
          // junction, and a break there would read as the line ending.
          var bends = turned[spot] || {};
          var meet = across[0].some(function (u) {
            return down[0].some(function (v) {
              return bends[u.rt.e[0] + ">" + v.rt.e[1]] || bends[v.rt.e[0] + ">" + u.rt.e[1]];
            });
          });
          if (meet) return;
          var both = longest(across[0]) >= longest(down[0]) ? [across, down] : [down, across];
          var fits = both.map(function (o) {
            return o[0].every(function (u) { return roomy(u, o[1]); });
          });
          var pick = fits[0] ? both[0] : fits[1] ? both[1] : null;
          if (!fits[0]) awkward++;
          if (!pick) return;
          pick[0].forEach(function (u) {
            var per = gaps.get(u.rt) || {};
            (per[u.i] = per[u.i] || []).push(pick[1]);
            gaps.set(u.rt, per);
          });
        });
        return { gaps: gaps, awkward: awkward };
      };
      var cost = function () {
        return total() * 1e5 + ink() + (big ? 0 : breaks().awkward * 40);
      };
      // The ink one arrow adds: its length, less where it lies on a line of the
      // same parent or into the same successor — the cheap form for large figures.
      var added = function (mine, others) {
        var n = 0;
        mine.forEach(function (u) {
          var h = Math.abs(u.a[1] - u.b[1]) < 0.5, ax = h ? 0 : 1, at = h ? 1 : 0;
          var lo = Math.min(u.a[ax], u.b[ax]), hi = Math.max(u.a[ax], u.b[ax]), shared = 0;
          others.forEach(function (v) {
            if (v.rt.e[0] !== u.rt.e[0] && v.rt.e[1] !== u.rt.e[1]) return;
            if ((Math.abs(v.a[1] - v.b[1]) < 0.5) !== h || Math.abs(v.a[at] - u.a[at]) > 0.5) return;
            var o = Math.min(hi, Math.max(v.a[ax], v.b[ax])) - Math.max(lo, Math.min(v.a[ax], v.b[ax]));
            if (o > shared) shared = o;
          });
          n += hi - lo - shared;
        });
        return n;
      };
      var key = function (rt) { return rt.e[0] + ">" + rt.e[1]; };
      var members = {};
      if (plan) {
        flexible.forEach(function (rt) { rt.k = plan.k[key(rt)]; });
        order = plan.order;
      }
      for (var step = 0; !plan && step < (big ? 1 : 3); step++) {
        flexible.forEach(function (rt) {
          if (rt.direct) return;
          var keep = rt.k;
          rt.k = undefined;
          var others = big ? index(all()) : null, best = null;
          var kin = big ? others.all.filter(function (v) {
            return v.rt.e[0] === rt.e[0] || v.rt.e[1] === rt.e[1];
          }) : null;
          // The gaps between the two ends' rows, and one beyond each: a route
          // that wanders further only grows longer.
          var lo = Math.min(rt.p.row, rt.s.row) - 1, hi = Math.max(rt.p.row, rt.s.row);
          gaps.filter(function (k) { return k >= lo && k <= hi; }).forEach(function (k) {
            rt.k = k;
            // A frame across rows fills the gap between them in its column: no
            // run may pass through it.
            var run = shape(rt), y = run[2][1], xa = Math.min(run[2][0], run[3][0]), xb = Math.max(run[2][0], run[3][0]);
            if (labels.some(function (q) { return q.t < y && q.b > y && q.r > xa && q.l < xb; })) return;
            // Small figures weigh each gap by the whole figure, so a crossing a
            // trunk already makes costs nothing more; large ones by this arrow.
            var mine = big ? segments(rt) : null;
            var c = big ? against(mine, others) * 1e5 + added(mine, kin) : cost();
            if (!best || c < best.c) best = { k: k, c: c };
          });
          rt.k = best ? best.k : keep;
        });
        members = {};
        flexible.forEach(function (rt) {
          runs(rt).forEach(function (kr) {
            var list = members[kr[0]] = members[kr[0]] || [];
            if (list.indexOf(kr[1]) < 0) list.push(kr[1]);
          });
        });
        Object.keys(order).forEach(function (key) { if (!members[key]) delete order[key]; });
        Object.keys(members).forEach(function (key) {  // keep a known order; newcomers: lower further left
          var known = (order[key] || []).filter(function (u) { return members[key].indexOf(u) >= 0; });
          var fresh = members[key].filter(function (u) { return known.indexOf(u) < 0; })
            .sort(function (u, v) { return yOf(v) - yOf(u); });
          order[key] = known.concat(fresh);
        });
        var shared = Object.keys(members).filter(function (key) { return members[key].length > 1; });
        if (big) {  // a large figure reorders only the channels a crossing runs through, at most 30
          var hot = {};
          pairs(all()).forEach(function (c) {
            [c.h.rt.e[0], c.h.rt.e[1], c.w.rt.e[0], c.w.rt.e[1]].forEach(function (id) { hot[id] = true; });
          });
          shared = shared.filter(function (key) {
            return members[key].some(function (run) { return hot[run.slice(2)]; });
          }).sort(function (a, b) { return members[b].length - members[a].length; }).slice(0, 30);
        }
        var options = shared.map(function (key) {
          var list = members[key];
          return list.length <= 4 && !big ? permutations(list) : big ? [
            list.slice().sort(function (u, v) { return yOf(u) - yOf(v); }),
            list.slice().sort(function (u, v) { return yOf(v) - yOf(u); })] : [
            order[key], list.slice().sort(function (u, v) { return yOf(u) - yOf(v); }),
            list.slice().sort(function (u, v) { return yOf(v) - yOf(u); })];
        });
        var combos = options.reduce(function (n, o) { return n * o.length; }, 1);
        var now = cost();
        if (combos <= 128) {
          var best = shared.map(function (key) { return order[key]; });
          for (var t = 0; t < combos; t++) {
            var rest = t;
            shared.forEach(function (key, i) { order[key] = options[i][rest % options[i].length]; rest = Math.floor(rest / options[i].length); });
            var c = cost();
            if (c < now) { now = c; best = shared.map(function (key) { return order[key]; }); }
          }
          shared.forEach(function (key, i) { order[key] = best[i]; });
        } else {
          for (var round = 0; round < (big ? 1 : 3); round++) {
            shared.forEach(function (key, i) {
              var keep = order[key];
              options[i].forEach(function (o) {
                order[key] = o;
                var c = cost();
                if (c < now) { now = c; keep = o; }
              });
              order[key] = keep;
            });
          }
        }
      }

      if (!plan) {
        plan = { k: {}, order: order };
        flexible.forEach(function (rt) { plan.k[key(rt)] = rt.k; });
      }

      var gaps = breaks().gaps;

      routes.forEach(function (rt) {
        var path = document.createElementNS(NS, "path");
        path.setAttribute("d", rounded(shape(rt), gaps.get(rt)));
        path.setAttribute("class", "lineage__edge");
        path.setAttribute("marker-end", "url(#lineage-arrow)");
        path.dataset.from = rt.e[0];
        path.dataset.to = rt.e[1];
        svg.appendChild(path);
      });
      mark();
    };
    draw();
    var queued = false;
    window.addEventListener("resize", function () {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { queued = false; draw(); });
    });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  });
})();
