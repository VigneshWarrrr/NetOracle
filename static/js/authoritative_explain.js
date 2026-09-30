/* Explanation section of the authoritative forecast page only (presentation layer).
 *
 * Renders, from the existing API response and never from a request of its own:
 *   - the predicted-MITRE-stage attribution list (explanations.current_state_evidence.top_stage_features),
 *     formatted exactly like the server-rendered attack-risk list (dashboard.views.format_attribution /
 *     build_attribution_rows, mirrored below);
 *   - a temporal view of each list's top features across the observed windows t-5..t
 *     (explanations.temporal_evidence.attack_risk_attribution / stage_attribution).
 * Values are shown as returned (unsigned Gradient x Input magnitudes); nothing is recomputed.
 * Sample 0 comes from json_script data in the page; later samples from the replay controller's
 * "af-replay:sample" event (the response it already fetched).
 */
(function () {
    "use strict";

    var section = document.querySelector(".af-explain");
    if (!section) { return; }

    var stageRowsEl = section.querySelector('[data-explain="stage-rows"]');
    var heatmapEl = section.querySelector('[data-explain="heatmap"]');
    var legendEl = section.querySelector('[data-explain="legend"]');
    var summaryEl = section.querySelector('[data-explain="heatmap-summary"]');
    var tableHead = section.querySelector('[data-explain="table-head"]');
    var tableBody = section.querySelector('[data-explain="table-body"]');
    var toggleButtons = Array.prototype.slice.call(section.querySelectorAll("[data-explain-target]"));

    var DEFAULT_WINDOWS = ["t-5", "t-4", "t-3", "t-2", "t-1", "t"];
    var state = null;
    var target = "risk";

    /* ---------------- formatting: identical to authoritative_replay.js / dashboard.views ---------------- */

    function splitDecimal(str) {
        var neg = str.charAt(0) === "-";
        if (neg) { str = str.slice(1); }
        var parts = str.split(".");
        return { neg: neg, int: parts[0] || "0", frac: parts[1] || "" };
    }

    function shortestPlain(x) {
        var m = x.toExponential().match(/^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/);
        var digits = m[2] + (m[3] || "");
        var point = parseInt(m[4], 10) + 1;
        var intPart, fracPart;
        if (point <= 0) { intPart = "0"; fracPart = new Array(-point + 1).join("0") + digits; }
        else if (point >= digits.length) { intPart = digits + new Array(point - digits.length + 1).join("0"); fracPart = ""; }
        else { intPart = digits.slice(0, point); fracPart = digits.slice(point); }
        return m[1] + intPart + (fracPart ? "." + fracPart : "");
    }

    function roundDecimal(str, n, mode) {
        var d = splitDecimal(str);
        var frac = d.frac + new Array(n + 2).join("0");
        var kept = (d.int + frac.slice(0, n)).split("").map(Number);
        var next = Number(frac.charAt(n));
        var rest = frac.slice(n + 1).replace(/0+$/, "");
        var roundUp;
        if (next > 5 || (next === 5 && rest.length > 0)) { roundUp = true; }
        else if (next === 5) { roundUp = mode === "half-up" ? true : (kept[kept.length - 1] % 2 === 1); }
        else { roundUp = false; }
        if (roundUp) {
            var i = kept.length - 1;
            while (i >= 0) { if (kept[i] === 9) { kept[i] = 0; i -= 1; } else { kept[i] += 1; break; } }
            if (i < 0) { kept.unshift(1); }
        }
        var all = kept.join("");
        var intDigits = all.slice(0, all.length - n).replace(/^0+(?=\d)/, "") || "0";
        var fracDigits = all.slice(all.length - n);
        var isZero = /^[0]*$/.test(intDigits + fracDigits);
        return (d.neg && !isZero ? "-" : "") + intDigits + (n > 0 ? "." + fracDigits : "");
    }

    function pyFixed(x, n) { return roundDecimal(x.toFixed(100), n, "half-even"); }
    function pyExp2(x) { return x.toExponential(2).replace(/e([+-])(\d)$/, "e$10$2"); }

    function pyRepr(x) {
        if (x === 0) { return (1 / x < 0) ? "-0.0" : "0.0"; }
        var m = x.toExponential().match(/^(-?\d(?:\.\d+)?)e([+-])(\d+)$/);
        var exp = parseInt(m[2] + m[3], 10);
        if (exp < -4 || exp >= 16) { return m[1] + "e" + m[2] + (m[3].length < 2 ? "0" + m[3] : m[3]); }
        var plain = shortestPlain(x);
        return plain.indexOf(".") === -1 ? plain + ".0" : plain;
    }

    // dashboard.views.format_attribution
    function formatAttribution(value) {
        if (value === 0) { return "0"; }
        if (Math.abs(value) >= 1e-3) { return pyFixed(value, 4); }
        return pyExp2(value);
    }

    // dashboard.views.build_attribution_rows
    function buildAttributionRows(features) {
        var values = features.map(function (item) { return Number(item.importance); });
        var top = values.length ? Math.max.apply(null, values) : 0;
        return features.map(function (item, i) {
            var value = values[i];
            var relative = top > 0 ? (value / top * 100) : 0;
            return { rank: i + 1, feature: item.feature, value: value, display: formatAttribution(value),
                     raw: pyRepr(value), relative_percent: pyFixed(relative, 0).replace(/\.$/, "") };
        });
    }

    /* ---------------- data ---------------- */

    function readJson(id) {
        var el = document.getElementById(id);
        if (!el) { return undefined; }
        try { return JSON.parse(el.textContent); } catch (e) { return undefined; }
    }

    function isMatrix(m, rows, cols) {
        return Array.isArray(m) && m.length === rows && m.every(function (r) { return Array.isArray(r) && r.length === cols; });
    }

    function setData(d) {
        var names = Array.isArray(d.names) ? d.names : [];
        state = {
            names: names,
            windows: Array.isArray(d.windows) && d.windows.length === 6 ? d.windows : DEFAULT_WINDOWS,
            risk: { top: Array.isArray(d.riskTop) ? d.riskTop : [], temporal: isMatrix(d.riskTemporal, 6, names.length) ? d.riskTemporal : null },
            stage: { top: Array.isArray(d.stageTop) ? d.stageTop : [], temporal: isMatrix(d.stageTemporal, 6, names.length) ? d.stageTemporal : null },
            stageClass: d.stageClass, stageStep: d.stageStep, stageMethod: d.stageMethod
        };
        renderStagePanel();
        renderTemporal();
    }

    /* ---------------- stage attribution list (same markup as the attack-risk list) ---------------- */

    function setText(name, text) {
        var el = section.querySelector('[data-explain="' + name + '"]');
        if (el) { el.textContent = text; }
    }

    function renderStagePanel() {
        setText("stage-class", state.stageClass || "n/a");
        setText("stage-step", typeof state.stageStep === "number" ? "t+" + (state.stageStep + 1) : "n/a");
        setText("stage-method", state.stageMethod || "n/a");
        var rows = buildAttributionRows(state.stage.top);
        var frag = document.createDocumentFragment();
        if (!rows.length) {
            var empty = document.createElement("li");
            empty.className = "af-explain-empty";
            empty.textContent = "No stage attribution in this response.";
            frag.appendChild(empty);
        }
        rows.forEach(function (row) {
            var li = document.createElement("li");
            li.className = "af-attr-row";
            li.style.setProperty("--share", row.relative_percent + "%");
            var rank = document.createElement("span"); rank.className = "af-attr-rank"; rank.textContent = String(row.rank);
            var name = document.createElement("strong"); name.className = "af-attr-name"; name.textContent = row.feature; name.title = row.feature;
            var vals = document.createElement("span"); vals.className = "af-attr-vals";
            var raw = document.createElement("small"); raw.className = "af-attr-raw"; raw.title = "Exact raw value: " + row.raw; raw.textContent = row.display;
            var share = document.createElement("b"); share.className = "af-attr-share"; share.textContent = row.relative_percent + "% of top";
            vals.appendChild(raw); vals.appendChild(document.createTextNode(" · ")); vals.appendChild(share);
            var bar = document.createElement("i"); bar.className = "af-attr-bar"; bar.setAttribute("aria-hidden", "true");
            li.appendChild(rank); li.appendChild(name); li.appendChild(vals); li.appendChild(bar);
            frag.appendChild(li);
        });
        stageRowsEl.replaceChildren(frag);
    }

    /* ---------------- temporal evidence: top features x observed windows ---------------- */

    // Sequential single-hue ramp, read from the theme tokens (--af-heat-lo / --af-heat-hi) at render time
    // so it follows the light/dark theme; an exact 0 is styled separately (hatched), never as a shade.
    var DEFAULT_LO = [224, 242, 254], DEFAULT_HI = [7, 89, 133];
    function rampColor(name, fallback) {
        var m = getComputedStyle(section).getPropertyValue(name).match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
        return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : fallback;
    }
    function shade(t) {
        var lo = rampColor("--af-heat-lo", DEFAULT_LO), hi = rampColor("--af-heat-hi", DEFAULT_HI);
        var c = lo.map(function (v, i) { return Math.round(v + (hi[i] - v) * t); });
        return "rgb(" + c.join(", ") + ")";
    }

    function currentTarget() { return state[target]; }

    function temporalRows() {
        var t = currentTarget();
        if (!t.temporal) { return []; }
        return t.top.map(function (item) {
            var idx = state.names.indexOf(item.feature);
            return { feature: item.feature, values: idx < 0 ? null : t.temporal.map(function (w) { return w[idx]; }) };
        }).filter(function (r) { return r.values; });
    }

    function renderTemporal() {
        toggleButtons.forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-explain-target") === target ? "true" : "false"); });
        var rows = temporalRows();
        var label = target === "risk" ? "attack-risk score" : "predicted MITRE stage";
        if (!rows.length) {
            heatmapEl.replaceChildren(textNode("p", "af-explain-empty", "No temporal attribution in this response."));
            legendEl.replaceChildren(); summaryEl.textContent = ""; tableHead.replaceChildren(); tableBody.replaceChildren();
            return;
        }
        var max = 0, maxAt = null;
        rows.forEach(function (r) { r.values.forEach(function (v, w) { if (v > max) { max = v; maxAt = { feature: r.feature, window: state.windows[w] }; } }); });

        // grid: feature column + six window columns
        var grid = document.createElement("div");
        grid.className = "af-explain-grid-heat";
        grid.appendChild(textNode("span", "af-explain-heat-corner", "Feature"));
        state.windows.forEach(function (w, i) {
            grid.appendChild(textNode("span", "af-explain-heat-col" + (i === 5 ? " af-explain-heat-col-current" : ""), w));
        });
        rows.forEach(function (r) {
            var name = textNode("span", "af-explain-heat-name", r.feature);
            name.title = r.feature;
            grid.appendChild(name);
            r.values.forEach(function (v, w) {
                var cell = document.createElement("span");
                cell.className = "af-explain-heat-cell" + (w === 5 ? " af-explain-heat-cell-current" : "") + (v === 0 ? " af-explain-heat-cell-zero" : "");
                cell.style.backgroundColor = v === 0 ? "" : shade(max > 0 ? v / max : 0);
                cell.title = r.feature + " · " + state.windows[w] + " · API value " + formatAttribution(v) + " (exact " + pyRepr(v) + ")";
                cell.setAttribute("data-value", String(v));
                grid.appendChild(cell);
            });
        });
        heatmapEl.replaceChildren(grid);
        heatmapEl.setAttribute("aria-label", "Attribution of the top " + rows.length + " features for the " + label + " across observed windows " + state.windows[0] + " to " + state.windows[5]);

        legendEl.replaceChildren(
            textNode("span", "af-explain-legend-label", "0"),
            textNode("span", "af-explain-legend-ramp", ""),
            textNode("span", "af-explain-legend-label", formatAttribution(max)),
            textNode("span", "af-explain-legend-note", "Shade is relative to the largest value shown; hatched cells are exactly 0.")
        );
        summaryEl.textContent = "Showing the " + label + " attribution for its " + rows.length + " top features (ranked at window t) across " +
            state.windows[0] + " … " + state.windows[5] + ". Largest value shown: " + formatAttribution(max) +
            (maxAt ? " (" + maxAt.feature + ", " + maxAt.window + ")." : ".");

        var head = document.createElement("tr");
        head.appendChild(th("Feature"));
        state.windows.forEach(function (w) { head.appendChild(th(w)); });
        tableHead.replaceChildren(head);
        var frag = document.createDocumentFragment();
        rows.forEach(function (r) {
            var tr = document.createElement("tr");
            var nameCell = document.createElement("th"); nameCell.scope = "row"; nameCell.textContent = r.feature;
            tr.appendChild(nameCell);
            r.values.forEach(function (v) {
                var td = document.createElement("td"); td.textContent = formatAttribution(v); td.title = "Exact raw value: " + pyRepr(v);
                tr.appendChild(td);
            });
            frag.appendChild(tr);
        });
        tableBody.replaceChildren(frag);
    }

    function textNode(tag, cls, text) { var el = document.createElement(tag); el.className = cls; el.textContent = text; return el; }
    function th(text) { var el = document.createElement("th"); el.scope = "col"; el.textContent = text; return el; }

    /* ---------------- wiring ---------------- */

    toggleButtons.forEach(function (b) {
        b.addEventListener("click", function () {
            target = b.getAttribute("data-explain-target");
            if (state) { renderTemporal(); } // re-render only; no request
        });
    });

    document.addEventListener("af-replay:sample", function (ev) {
        var data = ev.detail && ev.detail.data;
        var ex = (data && data.explanations) || {};
        var cur = ex.current_state_evidence || {};
        var tem = ex.temporal_evidence || {};
        var msp = ex.mitre_stage_prediction || {};
        setData({ names: ex.feature_names, windows: tem.window_labels, riskTop: cur.top_attack_risk_features, stageTop: cur.top_stage_features,
                  riskTemporal: tem.attack_risk_attribution, stageTemporal: tem.stage_attribution,
                  stageClass: msp.explained_stage_class, stageStep: msp.explained_step_index, stageMethod: msp.explanation_method });
    });

    setData({ names: readJson("af-explain-names"), windows: readJson("af-explain-windows"),
              riskTop: readJson("af-explain-risk-top"), stageTop: readJson("af-explain-stage-top"),
              riskTemporal: readJson("af-explain-risk-temporal"), stageTemporal: readJson("af-explain-stage-temporal"),
              stageClass: readJson("af-explain-stage-class"), stageStep: readJson("af-explain-stage-step"), stageMethod: readJson("af-explain-stage-method") });
})();
