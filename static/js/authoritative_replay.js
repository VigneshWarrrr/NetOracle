/* Replay navigation for the authoritative forecast page only.
 *
 * The page is server-rendered for test sample 0. This script steps through other
 * samples of the frozen Phase 3.5 test split by calling the existing endpoint
 * (/api/authoritative-predict/?index=<n>) and rewriting only the per-sample values
 * that the template renders. Labels, disclosures and semantics text are never
 * generated here; every displayed value comes from the API response, formatted
 * exactly as the Django template / dashboard.views formatters do.
 */
(function () {
    "use strict";

    var toolbar = document.getElementById("af-replay");
    if (!toolbar || typeof window.fetch !== "function") {
        return;
    }

    var PLAY_DELAY_MS = 1500;
    var apiUrl = toolbar.getAttribute("data-api-url");
    var minIndex = parseInt(toolbar.getAttribute("data-min-index"), 10);
    var maxIndex = parseInt(toolbar.getAttribute("data-max-index"), 10);
    var page = toolbar.closest(".af-page") || document.body;

    var slider = toolbar.querySelector("[data-replay-slider]");
    var indexOut = toolbar.querySelector("[data-replay-index]");
    var statusEl = toolbar.querySelector("[data-replay-status]");
    var prevBtn = toolbar.querySelector('[data-replay-action="prev"]');
    var nextBtn = toolbar.querySelector('[data-replay-action="next"]');
    var playBtn = toolbar.querySelector('[data-replay-action="play"]');

    var shownIndex = parseInt(toolbar.getAttribute("data-index"), 10) || 0; // sample currently rendered
    var requestSeq = 0;
    var inflight = null;
    var playing = false;
    var playTimer = null;

    /* ---------------- number formatting (mirrors Django / Python) ---------------- */

    // Exact decimal expansion helpers operate on {neg, int, frac} digit strings.
    function splitDecimal(str) {
        var neg = str.charAt(0) === "-";
        if (neg) { str = str.slice(1); }
        var parts = str.split(".");
        return { neg: neg, int: parts[0] || "0", frac: parts[1] || "" };
    }

    // Shortest round-trip digits (same digits as Python repr / str), as a plain decimal string.
    function shortestPlain(x) {
        var m = x.toExponential().match(/^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/);
        var digits = m[2] + (m[3] || "");
        var exp = parseInt(m[4], 10);
        var point = exp + 1; // digits before the decimal point
        var intPart, fracPart;
        if (point <= 0) {
            intPart = "0";
            fracPart = new Array(-point + 1).join("0") + digits;
        } else if (point >= digits.length) {
            intPart = digits + new Array(point - digits.length + 1).join("0");
            fracPart = "";
        } else {
            intPart = digits.slice(0, point);
            fracPart = digits.slice(point);
        }
        return m[1] + intPart + (fracPart ? "." + fracPart : "");
    }

    // Round a plain decimal string to n fraction digits. mode: "half-up" (away from zero) or "half-even".
    function roundDecimal(str, n, mode) {
        var d = splitDecimal(str);
        var frac = d.frac + new Array(n + 2).join("0");
        var kept = (d.int + frac.slice(0, n)).split("").map(Number);
        var next = Number(frac.charAt(n));
        var rest = frac.slice(n + 1).replace(/0+$/, "");
        var roundUp;
        if (next > 5 || (next === 5 && rest.length > 0)) {
            roundUp = true;
        } else if (next === 5) {
            roundUp = mode === "half-up" ? true : (kept[kept.length - 1] % 2 === 1);
        } else {
            roundUp = false;
        }
        if (roundUp) {
            var i = kept.length - 1;
            while (i >= 0) {
                if (kept[i] === 9) { kept[i] = 0; i -= 1; } else { kept[i] += 1; break; }
            }
            if (i < 0) { kept.unshift(1); }
        }
        var all = kept.join("");
        var intDigits = all.slice(0, all.length - n).replace(/^0+(?=\d)/, "") || "0";
        var fracDigits = all.slice(all.length - n);
        var isZero = /^[0]*$/.test(intDigits + fracDigits);
        return (d.neg && !isZero ? "-" : "") + intDigits + (n > 0 ? "." + fracDigits : "");
    }

    // Django {{ value|floatformat:n }} (n > 0): Decimal(repr(value)) quantized with ROUND_HALF_UP.
    function floatformat(value, n) {
        if (value === null || value === undefined || value === "") { return ""; }
        var x = Number(value);
        if (!isFinite(x)) { return String(value); }
        return roundDecimal(shortestPlain(x), n, "half-up");
    }

    // Python f"{x:.nf}": correctly rounded from the exact binary value, ties to even.
    function pyFixed(x, n) {
        return roundDecimal(x.toFixed(100), n, "half-even");
    }

    // Python f"{x:.2e}".
    function pyExp2(x) {
        return x.toExponential(2).replace(/e([+-])(\d)$/, "e$10$2");
    }

    // Python repr(float).
    function pyRepr(x) {
        if (x === 0) { return (1 / x < 0) ? "-0.0" : "0.0"; }
        var m = x.toExponential().match(/^(-?\d(?:\.\d+)?)e([+-])(\d+)$/);
        var exp = parseInt(m[2] + m[3], 10);
        if (exp < -4 || exp >= 16) {
            return m[1] + "e" + m[2] + (m[3].length < 2 ? "0" + m[3] : m[3]);
        }
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
            return {
                rank: i + 1,
                feature: item.feature,
                display: formatAttribution(value),
                raw: pyRepr(value),
                relative_percent: pyFixed(relative, 0).replace(/\.$/, "")
            };
        });
    }

    /* ---------------- DOM updates ---------------- */

    function field(name) {
        return page.querySelector('[data-replay="' + name + '"]');
    }

    function setText(name, text) {
        var el = field(name);
        if (el) { el.textContent = text; }
    }

    function slice(value, n) {
        return value === null || value === undefined ? "" : String(value).slice(0, n);
    }

    function get(obj, path) {
        return path.split(".").reduce(function (acc, key) {
            return acc === null || acc === undefined ? undefined : acc[key];
        }, obj);
    }

    function renderStageSteps(trajectory) {
        var list = field("stage-steps");
        if (!list) { return; }
        var labels = trajectory.horizon_labels || [];
        var stages = trajectory.per_step_stage || [];
        var confs = trajectory.per_step_confidence || [];
        var count = Math.min(labels.length, stages.length, confs.length); // zip() semantics
        var frag = document.createDocumentFragment();
        for (var i = 0; i < count; i += 1) {
            var li = document.createElement("li");
            li.className = "af-timeline-row";
            var step = document.createElement("span");
            step.className = "af-step";
            step.textContent = labels[i];
            var stage = document.createElement("strong");
            stage.className = "af-stage-name";
            stage.textContent = stages[i];
            var conf = document.createElement("small");
            conf.className = "af-stage-conf";
            conf.title = "Softmax confidence of the predicted stage class; this is not an attack risk";
            conf.textContent = "stage conf. " + floatformat(confs[i], 3);
            li.appendChild(step);
            li.appendChild(stage);
            li.appendChild(conf);
            frag.appendChild(li);
        }
        list.replaceChildren(frag);
    }

    function renderAttributionRows(rows) {
        var list = field("attribution-rows");
        if (!list) { return; }
        var frag = document.createDocumentFragment();
        rows.forEach(function (row) {
            var li = document.createElement("li");
            li.className = "af-attr-row";
            li.style.setProperty("--share", row.relative_percent + "%");
            var rank = document.createElement("span");
            rank.className = "af-attr-rank";
            rank.textContent = String(row.rank);
            var name = document.createElement("strong");
            name.className = "af-attr-name";
            name.textContent = row.feature;
            var vals = document.createElement("span");
            vals.className = "af-attr-vals";
            var raw = document.createElement("small");
            raw.className = "af-attr-raw";
            raw.title = "Exact raw value: " + row.raw;
            raw.textContent = row.display;
            var share = document.createElement("b");
            share.className = "af-attr-share";
            share.textContent = row.relative_percent + "% of top";
            vals.appendChild(raw);
            vals.appendChild(document.createTextNode(" · "));
            vals.appendChild(share);
            var bar = document.createElement("i");
            bar.className = "af-attr-bar";
            bar.setAttribute("aria-hidden", "true");
            li.appendChild(rank);
            li.appendChild(name);
            li.appendChild(vals);
            li.appendChild(bar);
            frag.appendChild(li);
        });
        list.replaceChildren(frag);
    }

    function render(f) {
        var explanations = f.explanations || {};
        var stagePrediction = explanations.mitre_stage_prediction || {};
        var meta = f.input_metadata || {};
        var windowLabels = meta.window_labels || [];
        var rollout = f.future_state_rollout || {};
        var trajectory = f.mitre_stage_trajectory || {};
        var prov = f.provenance || {};

        setText("score", floatformat(get(f, "whole_horizon_attack_probability.value"), 4));
        setText("overall-stage", stagePrediction.overall_stage || "n/a");
        setText("overall-confidence", floatformat(stagePrediction.overall_confidence, 3));
        setText("source-file", meta.source_file == null ? "" : meta.source_file);
        setText("window-start", meta.window_start == null ? "" : meta.window_start);

        setText("feature-count", meta.feature_count == null ? "" : String(meta.feature_count));
        setText("window-count", String(windowLabels.length));
        setText("window-labels", windowLabels.join(", "));
        setText("rollout-note", rollout.note == null ? "" : rollout.note);
        setText("horizon-labels", (rollout.horizon_labels || []).join(", "));

        renderStageSteps(trajectory);
        setText("stage-semantics", trajectory.semantics == null ? "" : trajectory.semantics);

        setText("attribution-method", get(explanations, "future_attack_risk_prediction.explanation_method") || "Gradient × Input");
        renderAttributionRows(buildAttributionRows(get(explanations, "current_state_evidence.top_attack_risk_features") || []));

        setText("prov-model", prov.model_identifier == null ? "" : prov.model_identifier);
        setText("prov-checkpoint", prov.checkpoint_path == null ? "" : prov.checkpoint_path);
        setText("prov-checkpoint-sha", slice(prov.checkpoint_sha256, 16));
        setText("prov-scaler-sha", slice(prov.scaler_sha256, 16));
        setText("prov-schema-sha", slice(prov.feature_schema_sha256, 16));
        setText("prov-commit", slice(prov.code_commit_hash, 12));
        setText("prov-device", prov.device == null ? "" : prov.device);
        setText("prov-duration", floatformat(prov.inference_duration_seconds, 4));
    }

    /* ---------------- toolbar state ---------------- */

    function clamp(n) {
        return Math.min(maxIndex, Math.max(minIndex, n));
    }

    function setStatus(text, isError) {
        statusEl.textContent = text;
        statusEl.classList.toggle("af-replay-error", Boolean(isError));
    }

    function showIndex(n) {
        slider.value = String(n);
        indexOut.textContent = String(n);
    }

    function updateButtons(target) {
        prevBtn.disabled = target <= minIndex;
        nextBtn.disabled = target >= maxIndex;
    }

    function setLoading(on) {
        toolbar.classList.toggle("af-replay-busy", on);
        page.classList.toggle("af-replay-loading", on);
        page.setAttribute("aria-busy", on ? "true" : "false");
    }

    function setPlaying(on) {
        playing = on;
        playBtn.textContent = on ? "Pause" : "Play";
        playBtn.setAttribute("aria-pressed", on ? "true" : "false");
        if (!on && playTimer) {
            clearTimeout(playTimer);
            playTimer = null;
        }
    }

    function errorMessage(response, body) {
        var detail = body && (body.error || body.detail);
        if (detail) { return "HTTP " + response.status + ": " + detail; }
        return "HTTP " + response.status + (response.statusText ? " " + response.statusText : "");
    }

    function load(target) {
        target = clamp(target);
        if (target === shownIndex && !inflight) {
            showIndex(shownIndex);
            updateButtons(shownIndex);
            return;
        }
        if (inflight) { inflight.abort(); } // a newer request supersedes an older one
        var controller = new AbortController();
        var seq = ++requestSeq;
        inflight = controller;

        showIndex(target);
        updateButtons(target);
        setLoading(true);
        setStatus("Loading test sample " + target + "…", false);

        fetch(apiUrl + "?index=" + encodeURIComponent(target), {
            credentials: "same-origin",
            headers: { "Accept": "application/json" },
            signal: controller.signal
        }).then(function (response) {
            return response.json().catch(function () { return null; }).then(function (body) {
                if (!response.ok) { throw new Error(errorMessage(response, body)); }
                if (!body || !body.whole_horizon_attack_probability || !body.mitre_stage_trajectory) {
                    throw new Error("unexpected response format");
                }
                return body;
            });
        }).then(function (body) {
            if (seq !== requestSeq) { return; }
            render(body);
            shownIndex = target;
            toolbar.setAttribute("data-index", String(target));
            // Share the already-fetched response with other page-scoped views (no extra request).
            document.dispatchEvent(new CustomEvent("af-replay:sample", { detail: { index: target, data: body } }));
            setStatus("Showing test sample " + target + ".", false);
            if (playing) {
                if (target >= maxIndex) {
                    setPlaying(false);
                    setStatus("Showing test sample " + target + ". Reached the last test sample; playback stopped.", false);
                } else {
                    playTimer = setTimeout(function () { playTimer = null; load(shownIndex + 1); }, PLAY_DELAY_MS);
                }
            }
        }).catch(function (err) {
            if (seq !== requestSeq || (err && err.name === "AbortError")) { return; }
            setPlaying(false);
            showIndex(shownIndex);
            updateButtons(shownIndex);
            setStatus("Could not load test sample " + target + " (" + (err && err.message ? err.message : "network error") +
                "). Still showing test sample " + shownIndex + ".", true);
        }).then(function () {
            if (seq !== requestSeq) { return; }
            inflight = null;
            setLoading(false);
        });
    }

    function navigate(target) {
        setPlaying(false); // manual navigation always stops playback
        load(target);
    }

    /* ---------------- wiring ---------------- */

    prevBtn.addEventListener("click", function () { navigate(Number(slider.value) - 1); });
    nextBtn.addEventListener("click", function () { navigate(Number(slider.value) + 1); });
    playBtn.addEventListener("click", function () {
        if (playing) {
            setPlaying(false);
            if (!inflight) { setStatus("Showing test sample " + shownIndex + ". Playback paused.", false); }
            return;
        }
        if (shownIndex >= maxIndex) {
            setStatus("Showing test sample " + shownIndex + ". Already at the last test sample.", false);
            return;
        }
        setPlaying(true);
        load(shownIndex + 1);
    });
    // Dragging only previews the number; one request is made when the value is committed.
    slider.addEventListener("input", function () { indexOut.textContent = slider.value; });
    slider.addEventListener("change", function () { navigate(Number(slider.value)); });

    slider.min = String(minIndex);
    slider.max = String(maxIndex);
    showIndex(shownIndex);
    updateButtons(shownIndex);
    toolbar.hidden = false;
})();
