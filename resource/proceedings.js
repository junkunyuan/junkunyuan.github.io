/* The conference-proceedings search (paper_reading_list/proceedings.html).

   The papers arrive as one data script per venue (resource/proceedings/<key>.js,
   written by sitegen/proceedings.py), each pushing its venue onto
   window.PROCEEDINGS; this script runs after them and searches them as you type.

   The query reads like the reading list's search box — every term has to
   match, a bare word matches anywhere, `field:value` only that field — plus
   `"quoted phrases"` (a full name) and `-word` (exclude). The query and the
   venue/track chips live in the URL's hash, so a search is a link: the contents
   page links `#venue=<key>`, and every author name is an `#q=author:"…"` link.

   Citation counts, where Semantic Scholar has them, come in a second, small
   script per venue (window.PROCEEDINGS_CITATIONS[key].counts, one per paper in
   the venue's order, -1 for none), so a refresh of the counts never rewrites
   the titles. They ride the "paper" link as the notes' counts do, and sort. */
(function () {
  "use strict";
  var input = document.getElementById("proceedingsSearch");
  var V = window.PROCEEDINGS;
  if (!input || !V || !V.length) return;

  var FIELD = { title: "t", author: "a", venue: "v", track: "k" };
  var FIRST = 50, PAGE = 100, AUTHORS_SHOWN = 12;
  var $ = function (id) { return document.getElementById(id); };
  var list = $("proceedingsResults"), more = $("proceedingsMore");

  // Case and accents never decide a match: "alche" finds "Alché".
  var norm = function (s) {
    return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  };
  // "|" delimits the fields of a paper's blob, so no value may carry one.
  var clean = function (s) { return norm(s).replace(/\|/g, " "); };
  var esc = function (s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  };
  var fmt = function (n) { return n.toLocaleString("en-US"); };
  var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];
  // "20261104" -> "November 04, 2026", the footer's spelling of a day.
  var day = function (s) {
    return s && s.length === 8 ? MONTHS[+s.slice(4, 6) - 1] + " " + s.slice(6) + ", " + s.slice(0, 4) : "";
  };
  var CITES = window.PROCEEDINGS_CITATIONS || {};

  // One record per paper, with a field-tagged blob as the notes have:
  // |t:title |a:author |a:author |v:venue key |k:track
  var papers = [];
  V.forEach(function (v, vi) {
    var parts = v.venue.split(" ");
    v.abbr = parts.slice(0, -1).join(" ") || v.venue;
    v.year = parts.length > 1 ? parts[parts.length - 1] : "";
    var cites = CITES[v.key];
    v.cited = cites ? "Citations from Semantic Scholar, " + day(cites.as_of) : "";
    v.papers.forEach(function (r, j) {
      var track = r[0] >= 0 ? v.tracks[r[0]] : "";
      papers.push({
        v: vi, k: r[0], title: r[1], authors: r[2], track: track,
        cites: cites && cites.counts[j] !== undefined ? cites.counts[j] : -1,
        url: v.url_prefix + r[3],
        blob: "|t:" + clean(r[1]) + r[2].map(function (a) { return "|a:" + clean(a); }).join("") +
              "|v:" + clean(v.venue + " " + v.key) + (track ? "|k:" + clean(track) : "")
      });
    });
    v.papers = null;  // the records hold everything now
  });

  var parse = function (q) {
    var terms = [], re = /(-?)(?:([a-z]+):)?(?:"([^"]*)"?|(\S+))/g, m;
    q = q.toLowerCase();
    while ((m = re.exec(q))) {
      var field = m[2] && FIELD[m[2]] ? FIELD[m[2]] : null;
      var value = m[3] !== undefined ? m[3] : m[4];
      if (m[2] && !field) value = m[2] + ":" + value;  // an unknown prefix is part of the word
      value = clean(value).trim();
      if (value) terms.push({ not: m[1] === "-", field: field, value: value });
    }
    return terms;
  };

  var hit = function (blob, t) {
    if (!t.field) return blob.indexOf(t.value) !== -1;
    var key = "|" + t.field + ":", from = 0, at, end;
    while ((at = blob.indexOf(key, from)) !== -1) {
      end = blob.indexOf("|", at + key.length);
      if (end === -1) end = blob.length;
      if (blob.slice(at + key.length, end).indexOf(t.value) !== -1) return true;
      from = end;
    }
    return false;
  };
  var matches = function (p, terms) {
    for (var i = 0; i < terms.length; i++) {
      if (hit(p.blob, terms[i]) === terms[i].not) return false;
    }
    return true;
  };

  // Mark the matches in the original text: walk it one character at a time and
  // keep where each normalized character came from.
  var marked = function (text, values) {
    if (!values.length) return esc(text);
    var flat = "", from = [], i, j, n;
    for (i = 0; i < text.length; i++) {
      n = norm(text[i]);
      for (j = 0; j < n.length; j++) { flat += n[j]; from.push(i); }
    }
    var ranges = [];
    values.forEach(function (v) {
      var at = 0, pos;
      while ((pos = flat.indexOf(v, at)) !== -1) {
        ranges.push([from[pos], from[pos + v.length - 1] + 1]);
        at = pos + v.length;
      }
    });
    if (!ranges.length) return esc(text);
    ranges.sort(function (a, b) { return a[0] - b[0]; });
    var merged = [ranges[0].slice()];
    ranges.slice(1).forEach(function (r) {
      var last = merged[merged.length - 1];
      if (r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
      else merged.push(r.slice());
    });
    var out = "", pos2 = 0;
    merged.forEach(function (r) {
      out += esc(text.slice(pos2, r[0])) + "<mark>" + esc(text.slice(r[0], r[1])) + "</mark>";
      pos2 = r[1];
    });
    return out + esc(text.slice(pos2));
  };

  var state = { venue: -1, track: -1, sort: "title", hits: [], shown: 0, title: [], author: [] };
  var anyCites = papers.some(function (p) { return p.cites >= 0; });

  var authorLink = function (name) {
    return '<a href="#q=' + encodeURIComponent('author:"' + name + '"') + '">' +
           marked(name, state.author) + "</a>";
  };
  var authorsHtml = function (p, all) {
    var names = p.authors, cut = names.length;
    if (!all && names.length > AUTHORS_SHOWN + 2) {
      cut = AUTHORS_SHOWN;
      // A match past the cut shows the whole list.
      for (var i = AUTHORS_SHOWN; i < names.length; i++) {
        var n = clean(names[i]);
        if (state.author.some(function (v) { return n.indexOf(v) !== -1; })) { cut = names.length; break; }
      }
    }
    var html = names.slice(0, cut).map(authorLink).join(", ");
    if (cut < names.length) {
      html += ', <button type="button" class="proceedings__more-authors">+' +
              (names.length - cut) + " more</button>";
    }
    return html;
  };

  var card = function (p, index) {
    var v = V[p.v];
    var link = function (href, label, tip) {
      return '<a href="' + esc(href) + '"' + (tip ? ' title="' + esc(tip) + '"' : "") +
             ' target="_blank" rel="noopener">' + label + "</a>";
    };
    var paper = p.cites >= 0
      ? link(p.url, 'paper <span class="count">(' + fmt(p.cites) + ")</span>", v.cited)
      : link(p.url, "paper");
    return '<div class="paper" data-i="' + index + '">' +
      '<p class="paper__title">' + link(p.url, marked(p.title, state.title)) + "</p>" +
      '<p class="paper__detail paper__authors">' + authorsHtml(p, false) + "</p>" +
      '<p class="paper__detail"><b class="paper__venue" title="' + esc(v.name) + '">' + esc(v.abbr) +
      "</b> " + esc(v.year) + (p.track ? " · " + esc(p.track) : "") + " · " + paper + "</p></div>";
  };

  var showMore = function (count) {
    var end = Math.min(state.hits.length, state.shown + count), html = "";
    for (var i = state.shown; i < end; i++) html += card(state.hits[i], i);
    list.insertAdjacentHTML("beforeend", html);
    state.shown = end;
    more.hidden = end >= state.hits.length;
    more.textContent = "Show " + fmt(Math.min(PAGE, state.hits.length - end)) + " more of " +
                       fmt(state.hits.length - end);
  };

  var chip = function (idx, label, n, on, tip) {
    return '<button type="button" class="proceedings__chip" data-idx="' + idx + '" aria-pressed="' + on + '"' +
           (tip ? ' title="' + esc(tip) + '"' : "") + ">" + esc(label) +
           '<span class="proceedings__chip-count">' + fmt(n) + "</span></button>";
  };

  var saveHash = function () {
    var parts = [];
    if (input.value.trim()) parts.push("q=" + encodeURIComponent(input.value.trim()));
    if (state.venue >= 0) parts.push("venue=" + V[state.venue].key);
    if (state.venue >= 0 && state.track >= 0) {
      parts.push("track=" + encodeURIComponent(V[state.venue].tracks[state.track]));
    }
    if (state.sort !== "title") parts.push("sort=" + state.sort);
    var url = location.pathname + location.search + (parts.length ? "#" + parts.join("&") : "");
    try { history.replaceState(null, "", url); } catch (e) { /* a file:// page may refuse */ }
  };

  var run = function () {
    var terms = parse(input.value);
    var plus = terms.filter(function (t) { return !t.not; });
    var values = function (f) {
      return plus.filter(function (t) { return !t.field || t.field === f; })
                 .map(function (t) { return t.value; });
    };
    state.title = values("t");
    state.author = values("a");
    var perVenue = V.map(function () { return 0; }), perTrack = {}, hits = [];
    for (var i = 0; i < papers.length; i++) {
      var p = papers[i];
      if (!matches(p, terms)) continue;
      perVenue[p.v]++;
      if (state.venue >= 0 && p.v !== state.venue) continue;
      if (state.venue >= 0) perTrack[p.k] = (perTrack[p.k] || 0) + 1;
      if (state.track >= 0 && p.k !== state.track) continue;
      hits.push(p);
    }
    // Most cited first; a paper without a count goes last; ties keep the title order.
    if (state.sort === "citations") hits.sort(function (a, b) { return b.cites - a.cites; });
    $("proceedingsSort").hidden = !anyCites;
    [].forEach.call($("proceedingsSort").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.sort === state.sort));
    });
    $("proceedingsVenues").innerHTML = V.map(function (v, i) {
      return chip(i, v.venue, perVenue[i], state.venue === i, v.name);
    }).join("");
    var tracks = state.venue >= 0 ? V[state.venue].tracks : [];
    $("proceedingsTracks").innerHTML = tracks.length > 1 ? tracks.map(function (t, i) {
      return chip(i, t, perTrack[i] || 0, state.track === i, "");
    }).join("") : "";
    var where = state.venue >= 0 ? " in " + V[state.venue].venue +
                (state.track >= 0 ? " · " + V[state.venue].tracks[state.track] : "") : "";
    $("proceedingsCount").textContent =
      fmt(hits.length) + (hits.length === 1 ? " paper" : " papers") + where;
    state.hits = hits;
    state.shown = 0;
    list.innerHTML = "";
    showMore(FIRST);
    saveHash();
  };

  // #q=…&venue=iclr2026&track=Oral
  var loadHash = function () {
    var params = {};
    location.hash.replace(/^#/, "").split("&").forEach(function (kv) {
      var at = kv.indexOf("=");
      if (at > 0) {
        try { params[kv.slice(0, at)] = decodeURIComponent(kv.slice(at + 1)); } catch (e) { /* skip */ }
      }
    });
    input.value = params.q || "";
    state.venue = -1;
    V.forEach(function (v, i) { if (v.key === params.venue) state.venue = i; });
    state.track = state.venue >= 0 ? V[state.venue].tracks.indexOf(params.track || "") : -1;
    state.sort = params.sort === "citations" && anyCites ? "citations" : "title";
  };

  var timer = null;
  input.addEventListener("input", function () {
    clearTimeout(timer);
    timer = setTimeout(run, 80);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { input.value = ""; run(); }
  });
  document.addEventListener("keydown", function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || "");
    if (e.key === "/" && !typing) { e.preventDefault(); input.focus(); input.select(); }
  });
  $("proceedingsVenues").addEventListener("click", function (e) {
    var b = e.target.closest(".proceedings__chip");
    if (!b) return;
    var i = +b.dataset.idx;
    state.venue = state.venue === i ? -1 : i;
    state.track = -1;
    run();
  });
  $("proceedingsTracks").addEventListener("click", function (e) {
    var b = e.target.closest(".proceedings__chip");
    if (!b) return;
    var i = +b.dataset.idx;
    state.track = state.track === i ? -1 : i;
    run();
  });
  list.addEventListener("click", function (e) {
    var b = e.target.closest(".proceedings__more-authors");
    if (!b) return;
    var card_ = b.closest(".paper");
    card_.querySelector(".paper__authors").innerHTML = authorsHtml(state.hits[+card_.dataset.i], true);
  });
  $("proceedingsSort").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b || b.dataset.sort === state.sort) return;
    state.sort = b.dataset.sort;
    run();
  });
  more.addEventListener("click", function () { showMore(PAGE); });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting && !more.hidden) showMore(PAGE);
    }, { rootMargin: "800px" }).observe(more);
  }
  // An author link, the back button, or a link from the contents page.
  window.addEventListener("hashchange", function () {
    loadHash();
    run();
    if (input.getBoundingClientRect().top < 0) input.scrollIntoView();
  });

  loadHash();
  run();
})();
