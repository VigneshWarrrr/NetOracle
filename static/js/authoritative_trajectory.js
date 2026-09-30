/* Network-state trajectory chart for the authoritative forecast page only.
 *
 * Plots one feature across 12 steps: the six observed input windows t-5..t
 * (explanations.temporal_evidence.raw_feature_values) followed by the model's six
 * predicted future states t+1..t+6 (future_state_rollout.predicted_state_raw_units).
 * Values are plotted exactly as given (straight segments, no smoothing, no derived
 * values; predicted values below 0 are marked, never altered, because the rollout is an
 * unconstrained regression output). Sample 0 comes from the server-rendered page; later samples come from the
 * response the replay controller already fetched ("af-replay:sample" event), so this
 * chart never makes a request of its own. Plain SVG, no chart library.
 */
(function () {
    "use strict";

    var section = document.getElementById("af-trajectory");
    if (!section) { return; }

    var SVG_NS = "http://www.w3.org/2000/svg";
    var DEFAULT_OBSERVED_LABELS = ["t-5", "t-4", "t-3", "t-2", "t-1", "t"];
    var DEFAULT_FORECAST_LABELS = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"];
    // Layout only (never affects values): taller plot on wide screens, slimmer y-axis gutter on narrow ones.
    var HEIGHT = 300;
    var MARGIN = { top: 40, right: 16, bottom: 34, left: 72 };
    function layoutFor(width) {
        return width < 480
            ? { height: 272, margin: { top: 40, right: 8, bottom: 32, left: 52 } }
            : { height: width >= 900 ? 320 : HEIGHT, margin: MARGIN };
    }

    var chartEl = section.querySelector("[data-trajectory-chart]");
    var selectEl = section.querySelector("#af-trajectory-feature");
    var summaryEl = section.querySelector("[data-trajectory-summary]");
    var tableBody = section.querySelector("[data-trajectory-table]");

    var state = { names: [], observed: null, predicted: null, observedLabels: DEFAULT_OBSERVED_LABELS, forecastLabels: DEFAULT_FORECAST_LABELS };
    var selectedName = null;
    var focusIndex = null;
    var geometry = null; // positions of the last render, for hover/keyboard
    var tooltip = null;

    /* ---------------- data ---------------- */

    function readJsonScript(id) {
        var el = document.getElementById(id);
        if (!el) { return null; }
        try { return JSON.parse(el.textContent); } catch (e) { return null; }
    }

    function isMatrix(m, rows, cols) {
        return Array.isArray(m) && m.length === rows && m.every(function (r) { return Array.isArray(r) && r.length === cols; });
    }

    function setData(names, observed, predicted, observedLabels, forecastLabels) {
        var ok = Array.isArray(names) && names.length > 0 && isMatrix(observed, 6, names.length) && isMatrix(predicted, 6, names.length);
        state.names = ok ? names : [];
        state.observed = ok ? observed : null;
        state.predicted = ok ? predicted : null;
        state.observedLabels = Array.isArray(observedLabels) && observedLabels.length === 6 ? observedLabels : DEFAULT_OBSERVED_LABELS;
        state.forecastLabels = Array.isArray(forecastLabels) && forecastLabels.length === 6 ? forecastLabels : DEFAULT_FORECAST_LABELS;
        syncSelect();
        render();
    }

    function syncSelect() {
        var names = state.names;
        var current = Array.prototype.map.call(selectEl.options, function (o) { return o.textContent; });
        if (current.length !== names.length || current.some(function (n, i) { return n !== names[i]; })) {
            var frag = document.createDocumentFragment();
            names.forEach(function (name, i) {
                var opt = document.createElement("option");
                opt.value = String(i);
                opt.textContent = name;
                frag.appendChild(opt);
            });
            selectEl.replaceChildren(frag);
        }
        var idx = selectedName !== null ? names.indexOf(selectedName) : -1;
        if (idx < 0) { idx = 0; } // default: first feature
        if (names.length) {
            selectEl.value = String(idx);
            selectedName = names[idx];
        }
        selectEl.disabled = names.length === 0;
    }

    // The 12 plotted points for the selected feature, straight from the two API matrices.
    function points() {
        var f = Number(selectEl.value);
        var pts = [];
        for (var i = 0; i < 6; i += 1) {
            pts.push({ label: state.observedLabels[i], kind: "observed", value: state.observed[i][f] });
        }
        for (var j = 0; j < 6; j += 1) {
            pts.push({ label: state.forecastLabels[j], kind: "forecast", value: state.predicted[j][f] });
        }
        return pts;
    }

    /* ---------------- formatting ---------------- */

    var groupFmt = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 });
    var compactFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });

    // Display only; the exact value is kept in the table cell's title and in the plot geometry.
    function formatValue(v) {
        if (typeof v !== "number" || !isFinite(v)) { return "n/a"; }
        if (v === 0) { return "0"; }
        var a = Math.abs(v);
        if (a >= 1e15 || a < 1e-3) { return v.toPrecision(6).replace(/\.?0+e/, "e"); }
        return groupFmt.format(v);
    }

    function formatTick(v, step) {
        if (v === 0) { return "0"; }
        if (Math.abs(v) >= 10000) { return compactFmt.format(v); }
        if (Math.abs(step) < 1e-3) { return v.toExponential(1); }
        var decimals = Math.max(0, -Math.floor(Math.log10(step)));
        return Number(v.toFixed(decimals)).toLocaleString("en-US", { maximumFractionDigits: decimals });
    }

    function niceStep(range, count) {
        var raw = range / count;
        var mag = Math.pow(10, Math.floor(Math.log10(raw)));
        var norm = raw / mag;
        return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    }

    /* ---------------- rendering ---------------- */

    function el(name, attrs, parent) {
        var node = document.createElementNS(SVG_NS, name);
        Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
        if (parent) { parent.appendChild(node); }
        return node;
    }

    function text(parent, attrs, content) {
        var t = el("text", attrs, parent);
        t.textContent = content;
        return t;
    }

    function isNum(v) { return typeof v === "number" && isFinite(v); }

    // A forecast point is flagged when the model output itself is below 0 (display marking only).
    function isNegativeForecast(p) { return p.kind === "forecast" && isNum(p.value) && p.value < 0; }

    function showMessage(message) {
        geometry = null;
        var p = document.createElement("p");
        p.className = "af-trajectory-empty";
        p.textContent = message;
        chartEl.replaceChildren(p);
        summaryEl.textContent = "";
        tableBody.replaceChildren();
    }

    function render() {
        if (!state.observed) {
            showMessage("Feature trajectory data is not available for this sample.");
            return;
        }
        var pts = points();
        var name = state.names[Number(selectEl.value)];
        var finite = pts.filter(function (p) { return typeof p.value === "number" && isFinite(p.value); });
        if (!finite.length) {
            showMessage("No finite values for " + name + " in this sample.");
            return;
        }

        var width = Math.max(300, Math.floor(chartEl.clientWidth || 640));
        var layout = layoutFor(width);
        var HEIGHT = layout.height, MARGIN = layout.margin;
        var plotW = width - MARGIN.left - MARGIN.right;
        var plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
        var slot = plotW / pts.length;
        var xAt = function (i) { return MARGIN.left + (i + 0.5) * slot; };
        var boundaryX = MARGIN.left + 6 * slot;

        var vals = finite.map(function (p) { return p.value; });
        var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
        if (lo === hi) { var pad = Math.abs(lo) * 0.1 || 1; lo -= pad; hi += pad; }
        var step = niceStep(hi - lo, 4);
        var y0 = Math.floor(lo / step) * step, y1 = Math.ceil(hi / step) * step;
        var yAt = function (v) { return MARGIN.top + plotH - ((v - y0) / (y1 - y0)) * plotH; };

        var svg = el("svg", {
            width: width, height: HEIGHT, viewBox: "0 0 " + width + " " + HEIGHT, class: "af-trajectory-svg",
            role: "img", "aria-label": "Line chart of " + name + " across observed input windows t-5 to t and model-predicted states t+1 to t+6"
        });

        // forecast zone + boundary
        el("rect", { x: boundaryX, y: MARGIN.top, width: MARGIN.left + plotW - boundaryX, height: plotH, class: "af-trajectory-zone" }, svg);

        // y grid + ticks. Tick positions are y0 + k*step; a tick within float noise of 0 is
        // labelled/placed at exactly 0 (display only - plotted data are untouched).
        var tickCount = Math.round((y1 - y0) / step);
        for (var k = 0; k <= tickCount; k += 1) {
            var t = y0 + k * step;
            if (Math.abs(t) < step * 1e-9) { t = 0; }
            var ty = yAt(t);
            el("line", { x1: MARGIN.left, x2: MARGIN.left + plotW, y1: ty, y2: ty, class: "af-trajectory-grid" }, svg);
            text(svg, { x: MARGIN.left - 8, y: ty, class: "af-trajectory-tick" + (t === 0 ? " af-trajectory-tick-zero" : ""), "text-anchor": "end", "dominant-baseline": "middle" }, formatTick(t, step));
        }
        // zero reference line, distinct from the grid, whenever 0 lies inside the axis range
        if (y0 <= 0 && y1 >= 0) {
            var zy = yAt(0);
            el("line", { x1: MARGIN.left, x2: MARGIN.left + plotW, y1: zy, y2: zy, class: "af-trajectory-zero" }, svg);
        }

        // x labels
        pts.forEach(function (p, i) {
            text(svg, { x: xAt(i), y: MARGIN.top + plotH + 20, class: "af-trajectory-tick" + (p.kind === "forecast" ? " af-trajectory-tick-forecast" : ""), "text-anchor": "middle" }, p.label);
        });

        // Region labels in a band above the plot; the boundary label is drawn as a pill after insertion (needs text metrics).
        el("line", { x1: boundaryX, x2: boundaryX, y1: MARGIN.top - 26, y2: MARGIN.top + plotH, class: "af-trajectory-boundary" }, svg);
        var labelY = MARGIN.top - 14;
        text(svg, { x: MARGIN.left, y: labelY, class: "af-trajectory-zone-label", "dominant-baseline": "middle" }, "OBSERVED INPUT");
        var boundaryPill = el("g", { class: "af-trajectory-boundary-pill" }, svg);
        var pillBg = el("rect", { rx: 3, ry: 3, class: "af-trajectory-boundary-pill-bg" }, boundaryPill);
        var boundaryLabel = text(boundaryPill, { x: boundaryX + 10, y: labelY, class: "af-trajectory-boundary-label", "dominant-baseline": "middle" }, "FORECAST START");
        var forecastLabel = text(svg, { x: MARGIN.left + plotW, y: labelY, class: "af-trajectory-zone-label af-trajectory-zone-label-forecast", "text-anchor": "end", "dominant-baseline": "middle" }, "MODEL-PREDICTED STATE");

        // lines: straight segments between consecutive finite points
        function path(fromIdx, toIdx) {
            var d = "";
            for (var i = fromIdx; i <= toIdx; i += 1) {
                var v = pts[i].value;
                if (typeof v !== "number" || !isFinite(v)) { continue; }
                d += (d ? "L" : "M") + xAt(i).toFixed(2) + " " + yAt(v).toFixed(2);
            }
            return d;
        }
        // Two separate paths: observed ends at t, forecast starts at t+1 (no connector across the boundary).
        el("path", { d: path(0, 5), class: "af-trajectory-line-observed" }, svg);
        el("path", { d: path(6, 11), class: "af-trajectory-line-forecast" }, svg);

        var markers = [];
        pts.forEach(function (p, i) {
            if (!isNum(p.value)) { markers.push(null); return; }
            var negative = isNegativeForecast(p);
            var cls = p.kind === "observed" ? "af-trajectory-dot-observed" : "af-trajectory-dot-forecast";
            if (negative) {
                // anomalous marker: a hollow diamond (shape, not only colour) around the unchanged point
                var cx = xAt(i), cy = yAt(p.value), r = 6;
                markers.push(el("path", {
                    d: "M" + cx + " " + (cy - r) + "L" + (cx + r) + " " + cy + "L" + cx + " " + (cy + r) + "L" + (cx - r) + " " + cy + "Z",
                    class: cls + " af-trajectory-dot-negative", "data-step": p.label, "data-value": String(p.value), "data-negative": "true"
                }, svg));
                return;
            }
            markers.push(el("circle", {
                cx: xAt(i), cy: yAt(p.value), r: 4, class: cls,
                "data-step": p.label, "data-value": String(p.value)
            }, svg));
        });

        var crosshair = el("line", { x1: 0, x2: 0, y1: MARGIN.top, y2: MARGIN.top + plotH, class: "af-trajectory-crosshair", visibility: "hidden" }, svg);
        var hit = el("rect", { x: MARGIN.left, y: MARGIN.top, width: plotW, height: plotH, class: "af-trajectory-hit" }, svg);

        tooltip = document.createElement("div");
        tooltip.className = "af-trajectory-tooltip";
        tooltip.hidden = true;
        chartEl.replaceChildren(svg, tooltip);

        // Size the FORECAST START pill to its text; drop the right-hand region label if it would collide.
        try {
            var lb = boundaryLabel.getBBox();
            pillBg.setAttribute("x", lb.x - 6);
            pillBg.setAttribute("y", lb.y - 3);
            pillBg.setAttribute("width", lb.width + 12);
            pillBg.setAttribute("height", lb.height + 6);
            if (forecastLabel.getBBox().x < lb.x + lb.width + 16) { forecastLabel.remove(); }
        } catch (e) { forecastLabel.remove(); }

        geometry = { pts: pts, xAt: xAt, yAt: yAt, slot: slot, crosshair: crosshair, markers: markers, name: name, width: width, height: HEIGHT, top: MARGIN.top };
        hit.addEventListener("pointermove", function (ev) {
            var box = svg.getBoundingClientRect();
            var i = Math.floor((ev.clientX - box.left - MARGIN.left) / slot);
            showPoint(Math.max(0, Math.min(pts.length - 1, i)));
        });
        hit.addEventListener("pointerleave", function () { if (document.activeElement !== chartEl) { hidePoint(); } });
        if (focusIndex !== null && document.activeElement === chartEl) { showPoint(focusIndex); }

        renderSummary(name, pts);
        renderTable(pts);
    }

    function showPoint(i) {
        if (!geometry) { return; }
        var p = geometry.pts[i];
        focusIndex = i;
        var x = geometry.xAt(i);
        geometry.crosshair.setAttribute("x1", x);
        geometry.crosshair.setAttribute("x2", x);
        geometry.crosshair.setAttribute("visibility", "visible");
        geometry.markers.forEach(function (m, k) { setActive(m, k === i); });

        var kind = p.kind === "observed" ? "Observed input" : "Model-predicted state";
        tooltip.replaceChildren();
        // step + series chip
        var head = document.createElement("div");
        head.className = "af-trajectory-tooltip-head";
        var stepEl = document.createElement("strong");
        stepEl.textContent = p.label;
        var chip = document.createElement("span");
        chip.className = "af-trajectory-tooltip-kind af-trajectory-tooltip-kind-" + p.kind;
        chip.textContent = kind;
        head.appendChild(stepEl);
        head.appendChild(chip);
        var nameLine = document.createElement("span");
        nameLine.className = "af-trajectory-tooltip-feature";
        nameLine.textContent = geometry.name;
        var valueRow = document.createElement("div");
        valueRow.className = "af-trajectory-tooltip-row";
        var valueLabel = document.createElement("span");
        valueLabel.textContent = "API value";
        var valueLine = document.createElement("span");
        valueLine.className = "af-trajectory-tooltip-value";
        valueLine.textContent = formatValue(p.value);
        valueRow.appendChild(valueLabel);
        valueRow.appendChild(valueLine);
        tooltip.appendChild(head);
        tooltip.appendChild(nameLine);
        tooltip.appendChild(valueRow);
        if (isNegativeForecast(p)) {
            var note = document.createElement("span");
            note.className = "af-trajectory-tooltip-note";
            note.textContent = "Below 0 · unconstrained regression output, shown unmodified";
            tooltip.appendChild(note);
        }
        tooltip.hidden = false;
        var left = x + 12;
        if (left + tooltip.offsetWidth > geometry.width) { left = x - 12 - tooltip.offsetWidth; }
        var y = typeof p.value === "number" && isFinite(p.value) ? geometry.yAt(p.value) : geometry.top;
        tooltip.style.left = Math.max(0, left) + "px";
        tooltip.style.top = Math.max(0, Math.min(geometry.height - tooltip.offsetHeight, y - tooltip.offsetHeight / 2)) + "px";
    }

    function hidePoint() {
        focusIndex = null;
        if (!geometry) { return; }
        geometry.crosshair.setAttribute("visibility", "hidden");
        geometry.markers.forEach(function (m) { setActive(m, false); });
        tooltip.hidden = true;
    }

    function setActive(marker, on) {
        if (!marker) { return; }
        if (marker.tagName === "circle") { marker.setAttribute("r", on ? 6 : 4); }
        else { marker.classList.toggle("is-active", on); }
    }

    function range(pts) {
        var v = pts.map(function (p) { return p.value; }).filter(function (x) { return typeof x === "number" && isFinite(x); });
        if (!v.length) { return "n/a"; }
        var lo = Math.min.apply(null, v), hi = Math.max.apply(null, v);
        return lo === hi ? formatValue(lo) + " (constant)" : formatValue(lo) + " to " + formatValue(hi);
    }

    function renderSummary(name, pts) {
        var line = document.createElement("span");
        line.textContent = name + ": observed input range " + range(pts.slice(0, 6)) +
            " (t-5 … t); model-predicted range " + range(pts.slice(6)) + " (t+1 … t+6).";
        var parts = [line];
        var forecast = pts.slice(6);
        var negatives = forecast.filter(isNegativeForecast).length;
        var keyItem = section.querySelector("[data-trajectory-key-negative]");
        if (keyItem) { keyItem.hidden = negatives === 0; }
        if (negatives > 0) {
            var warn = document.createElement("span");
            warn.className = "af-trajectory-warning";
            warn.setAttribute("data-trajectory-warning", String(negatives));
            warn.textContent = "Some predicted values are below 0: " + negatives + " of " + forecast.length +
                " predicted values are below 0. The state rollout is an unconstrained regression output and is shown unmodified.";
            parts.push(warn);
        }
        summaryEl.replaceChildren.apply(summaryEl, parts);
    }

    function renderTable(pts) {
        var frag = document.createDocumentFragment();
        pts.forEach(function (p) {
            var tr = document.createElement("tr");
            var a = document.createElement("td"); a.textContent = p.label;
            var b = document.createElement("td");
            b.textContent = (p.kind === "observed" ? "Observed input" : "Model-predicted state") + (isNegativeForecast(p) ? " · below 0" : "");
            var c = document.createElement("td"); c.textContent = formatValue(p.value); c.title = "API value: " + String(p.value);
            if (isNegativeForecast(p)) { tr.className = "af-trajectory-row-negative"; }
            tr.appendChild(a); tr.appendChild(b); tr.appendChild(c);
            frag.appendChild(tr);
        });
        tableBody.replaceChildren(frag);
    }

    /* ---------------- wiring ---------------- */

    selectEl.addEventListener("change", function () {
        selectedName = state.names[Number(selectEl.value)];
        render();
    });

    chartEl.addEventListener("keydown", function (ev) {
        if (!geometry) { return; }
        var last = geometry.pts.length - 1;
        var i = focusIndex === null ? 0 : focusIndex;
        if (ev.key === "ArrowRight") { i = Math.min(last, i + 1); }
        else if (ev.key === "ArrowLeft") { i = Math.max(0, i - 1); }
        else if (ev.key === "Home") { i = 0; }
        else if (ev.key === "End") { i = last; }
        else if (ev.key === "Escape") { hidePoint(); return; }
        else { return; }
        ev.preventDefault();
        showPoint(i);
    });
    chartEl.addEventListener("focus", function () { if (focusIndex === null) { showPoint(0); } });
    chartEl.addEventListener("blur", hidePoint);

    // Later samples: reuse the response the replay controller already fetched.
    document.addEventListener("af-replay:sample", function (ev) {
        var data = ev.detail && ev.detail.data;
        var ex = (data && data.explanations) || {};
        var temporal = ex.temporal_evidence || {};
        var rollout = (data && data.future_state_rollout) || {};
        setData(ex.feature_names, temporal.raw_feature_values, rollout.predicted_state_raw_units,
            temporal.window_labels || (data && data.input_metadata && data.input_metadata.window_labels), rollout.horizon_labels);
    });

    if (typeof ResizeObserver === "function") {
        var lastWidth = 0;
        new ResizeObserver(function () {
            var w = Math.floor(chartEl.clientWidth);
            if (w && w !== lastWidth) { lastWidth = w; render(); }
        }).observe(chartEl);
    }

    // Sample 0: the server-rendered response embedded in the page.
    setData(readJsonScript("af-trajectory-names"), readJsonScript("af-trajectory-observed"), readJsonScript("af-trajectory-predicted"));
})();
