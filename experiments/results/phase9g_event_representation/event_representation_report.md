# Phase 9G: Temporal Communication Event Representation

Verdict: **GREEN**

## 1. EXECUTIVE RESULT

Two already-validated PCAP captures (phase9d_UCAP172.31.69.22, phase9f_capWIN-J6GMIG1DQE5-172.31.64.89) were re-parsed with the identical Phase 9E parsing pipeline (no network access). Representation B2 (micro-batch events, 1s buckets) reconciles EXACTLY against the raw packet stream on every checked quantity (packet count, byte count, protocol counts, port sets) in both captures: True. B2 reduces [7316, 46295] raw packets to [5547, 19927] events respectively, while the graph-snapshot representation (A) stayed only ['30.9%', '80.5%'] non-empty (matching Phase 9F's finding). NOTABLY, fewer events does NOT mean smaller storage: B2 is smaller than B1 in serialized bytes in 0/2 captures -- see the per-capture storage notes in Section 5 for why (median burst size near 1 in both captures leaves little for per-event aggregate fields to compress).

## 2. INPUT CAPTURES

- CONFIRMED: `phase9d_UCAP172.31.69.22` -- 7,316 IP packets, 551 unique hosts, 32220.9s duration. Source: reused read-only, no network access this phase.
- CONFIRMED: `phase9f_capWIN-J6GMIG1DQE5-172.31.64.89` -- 46,295 IP packets, 597 unique hosts, 32536.1s duration. Source: reused read-only, no network access this phase.

## Capture: `phase9d_UCAP172.31.69.22`

### 3. REPRESENTATION A -- GRAPH SNAPSHOTS

- CONFIRMED: total_windows = 3223, non_empty_windows = 997
- CONFIRMED: total_window_edges = 3052, unique_edges = 1094
- CONFIRMED: temporal_transitions_represented = 3222
- CONFIRMED: serialized_size_bytes = 3,122,718

### 4. REPRESENTATION B1 -- PACKET EVENTS

- CONFIRMED: event_count = 7,316 (one per IP packet, by construction)
- DERIVED: estimated_serialized_size_bytes = 1,568,433

### 5. REPRESENTATION B2 -- MICRO-BATCH EVENTS

- CONFIRMED: micro_batch_seconds = 1 (fixed, not tuned)
- OBSERVED: event_count = 5,547
- DERIVED: compression_ratio_packets_per_event = 1.3189111231296196
- OBSERVED: events_per_second = 0.17215534213374273
- OBSERVED: maximum_burst_size = 107
- OBSERVED: median/p95 inter-event time = 7.700920104980469e-05 / 38.302341401576996
- OBSERVED: maximum_inter_event_gap = 206.75334787368774
- OBSERVED: median/p95 burst size = 1.0 / 2.0
- OBSERVED: unique_communicating_pairs = 1094, unique_hosts = 551
- OBSERVED: %events previously-unseen pair/destination/source = 19.72% / 9.83% / 9.93%
- CONFIRMED: serialized_size_bytes = 2,665,585
- OBSERVED: B2 (5,547 events, 2,665,585 bytes) is LARGER than B1 (7,316 events, 1,568,433 bytes) in serialized bytes, despite having fewer events. Reducing EVENT COUNT does not automatically reduce STORAGE SIZE: each B2 event carries more fields than a B1 event (aggregate mean/std/count fields vs. B1's raw per-packet fields), and this capture's median burst size is 1.0 packets/event -- when most events represent only 1 packet, the extra aggregate fields are pure overhead with no compression benefit to offset them.

### 7. INFORMATION PRESERVATION (B2 vs raw packet stream)

- CONFIRMED: total_packets reconcile exactly = True (7,316 / 7,316)
- CONFIRMED: total_bytes reconcile exactly = True
- CONFIRMED: unique_hosts reconcile exactly = True (551 / 551)
- CONFIRMED: unique_directed_pairs reconcile exactly = True (1094 / 1094)
- CONFIRMED: protocol_counts reconcile exactly = True
- CONFIRMED: source/destination port sets reconcile exactly = True / True
- Information lost by B2:
  - Exact arrival order and individual timestamps of packets WITHIN one micro-batch bucket (1s) are not retained -- only packet_count, mean_packet_length/std, and mean_IAT/std survive per bucket.
  - TTL, IP fragmentation flags, and TCP window size (present on every raw packet, per Phase 9E's feature-availability table) are not carried into B2 event features in this phase -- they were not in the Step 4 required feature list and were not fabricated here.

### 6. REPRESENTATION C -- HYBRID

C = the full B2 micro-batch event stream, PLUS one compact per-10s-window state summary (active_hosts, active_edges, packet/byte volume, protocol distribution, max burst size), PLUS a window_index per event so any consumer can slice the event stream by window without re-deriving window boundaries. No topology detail (which specific hosts/edges) is duplicated in the window summary -- that detail lives only in the event stream, avoided as redundant storage.

- OBSERVED: window_state_summary_count = 997
- OBSERVED: window_state_summary_bytes = 197,626
- OBSERVED: naive_combined_bytes (A + B2) = 5,788,303
- OBSERVED: lean_hybrid_bytes (B2 + window summary + index) = 2,894,505
- DERIVED: size_saving_vs_naive_pct = 50.0%

## Capture: `phase9f_capWIN-J6GMIG1DQE5-172.31.64.89`

### 3. REPRESENTATION A -- GRAPH SNAPSHOTS

- CONFIRMED: total_windows = 3254, non_empty_windows = 2620
- CONFIRMED: total_window_edges = 10101, unique_edges = 828
- CONFIRMED: temporal_transitions_represented = 3253
- CONFIRMED: serialized_size_bytes = 10,340,510

### 4. REPRESENTATION B1 -- PACKET EVENTS

- CONFIRMED: event_count = 46,295 (one per IP packet, by construction)
- DERIVED: estimated_serialized_size_bytes = 9,932,499

### 5. REPRESENTATION B2 -- MICRO-BATCH EVENTS

- CONFIRMED: micro_batch_seconds = 1 (fixed, not tuned)
- OBSERVED: event_count = 19,927
- DERIVED: compression_ratio_packets_per_event = 2.3232297887288604
- OBSERVED: events_per_second = 0.6124585123435358
- OBSERVED: maximum_burst_size = 652
- OBSERVED: median/p95 inter-event time = 0.11444103717803955 / 8.837884843349457
- OBSERVED: maximum_inter_event_gap = 48.99052977561951
- OBSERVED: median/p95 burst size = 1.0 / 5.0
- OBSERVED: unique_communicating_pairs = 828, unique_hosts = 597
- OBSERVED: %events previously-unseen pair/destination/source = 4.16% / 1.19% / 2.98%
- CONFIRMED: serialized_size_bytes = 10,033,597
- OBSERVED: B2 (19,927 events, 10,033,597 bytes) is LARGER than B1 (46,295 events, 9,932,499 bytes) in serialized bytes, despite having fewer events. Reducing EVENT COUNT does not automatically reduce STORAGE SIZE: each B2 event carries more fields than a B1 event (aggregate mean/std/count fields vs. B1's raw per-packet fields), and this capture's median burst size is 1.0 packets/event -- when most events represent only 1 packet, the extra aggregate fields are pure overhead with no compression benefit to offset them.

### 7. INFORMATION PRESERVATION (B2 vs raw packet stream)

- CONFIRMED: total_packets reconcile exactly = True (46,295 / 46,295)
- CONFIRMED: total_bytes reconcile exactly = True
- CONFIRMED: unique_hosts reconcile exactly = True (597 / 597)
- CONFIRMED: unique_directed_pairs reconcile exactly = True (828 / 828)
- CONFIRMED: protocol_counts reconcile exactly = True
- CONFIRMED: source/destination port sets reconcile exactly = True / True
- Information lost by B2:
  - Exact arrival order and individual timestamps of packets WITHIN one micro-batch bucket (1s) are not retained -- only packet_count, mean_packet_length/std, and mean_IAT/std survive per bucket.
  - TTL, IP fragmentation flags, and TCP window size (present on every raw packet, per Phase 9E's feature-availability table) are not carried into B2 event features in this phase -- they were not in the Step 4 required feature list and were not fabricated here.

### 6. REPRESENTATION C -- HYBRID

C = the full B2 micro-batch event stream, PLUS one compact per-10s-window state summary (active_hosts, active_edges, packet/byte volume, protocol distribution, max burst size), PLUS a window_index per event so any consumer can slice the event stream by window without re-deriving window boundaries. No topology detail (which specific hosts/edges) is duplicated in the window summary -- that detail lives only in the event stream, avoided as redundant storage.

- OBSERVED: window_state_summary_count = 2620
- OBSERVED: window_state_summary_bytes = 523,439
- OBSERVED: naive_combined_bytes (A + B2) = 20,374,107
- OBSERVED: lean_hybrid_bytes (B2 + window summary + index) = 10,669,625
- DERIVED: size_saving_vs_naive_pct = 47.6%

## 8. COMPLEXITY / STORAGE SUMMARY

See per-capture Sections 3-6 above and the comparison table below (measured storage row).

## 9. A vs B vs C COMPARISON

| property | graph snapshot (A) | packet events (B1) | micro-batch events (B2) | hybrid (C) |
|---|---|---|---|---|
| temporal resolution | 10s (fixed window) | per-packet (microsecond, as captured) | 1s bucket + exact intra-bucket mean/std IAT | 1s events + 10s summary |
| variable topology handling | OBSERVED (Phase 9F): windows with different node/edge counts require padding/masking for fixed-size batching | Naturally variable-length; no padding concept needed | Naturally variable-length; no padding concept needed | Event stream variable-length; window summary is fixed-dim per window |
| sparsity handling | OBSERVED (Phase 9F): 55.8% windows non-empty across both captures combined -- empty windows still occupy a sequence slot | No empty slots by construction (a packet either happened or it did not appear) | No empty slots by construction | Event stream has no empty slots; window summary still has empty-window slots |
| sequence length (this run) | [3223, 3254] windows | [7316, 46295] events | [5547, 19927] events | [5547, 19927] events + [3223, 3254] window summaries |
| measured storage (serialized JSON, this run) | [3122718, 10340510] bytes | [1568433, 9932499] bytes (estimated) | [2665585, 10033597] bytes | [2894505, 10669625] bytes (lean form) |
| packet information preservation | DERIVED: aggregated per-window only; individual packet timing lost | CONFIRMED: complete, by construction (one event per packet) | OBSERVED (Step 6): totals/protocol/port-set reconcile exactly; intra-bucket ordering lost | Same as micro-batch events (hybrid does not change the event layer) |
| topology information | CONFIRMED: explicit node/edge structure per window | DERIVED: topology recoverable by replaying events, but not explicit | DERIVED: topology recoverable by replaying events, but not explicit | CONFIRMED: window summary carries active_hosts/active_edges counts explicitly; specific edge identities remain in the event layer only |
| burst information | DERIVED: only via edge packet_count aggregated over the full 10s window | Implicit only (would need to re-derive bursts from timestamps) | CONFIRMED: burst size is a first-class per-event field (max/median/p95 reported directly, 1s resolution) | Same as micro-batch events, plus a window-level max_burst_in_window summary field |
| port behavior | OBSERVED: unique_destination_ports/unique_source_ports tracked per node per window (aggregated across all edges of that node) | CONFIRMED: exact port pair retained per event | OBSERVED: port pair is part of the grouping key, so source/destination-port diversity per event is trivially 1 by construction -- port behavior is visible at the SEQUENCE level (which ports appear, how often), not as an internal per-event diversity statistic | Same as micro-batch events |
| implementation complexity | Reused as-is from Phase 9E (no new code needed) | Trivial (no aggregation logic) | Moderate (bucketing + aggregate feature computation, this phase's main new code) | Moderate-plus (micro-batch logic + a thin per-window summary + an index) -- no fusion mechanism implemented |
| suitability for future world-model input | HYPOTHESIZED: matches representation B in the project's architecture diagram (graph encoder branch) | HYPOTHESIZED: too fine-grained/long as a direct model input without further batching | HYPOTHESIZED: plausible direct sequence-model input given its bounded compression ratio and explicit burst/IAT fields | HYPOTHESIZED: matches the project's intended 'FLOW STATE + PACKET GRAPH -> MULTIMODAL NETWORK STATE' pattern most closely, at the representation-definition level only -- NOT VERIFIED as better for prediction, since no model was trained |

## 10. FUTURE WORLD MODEL IMPLICATIONS

- **candidate_A_event_encoder_only**: HYPOTHESIZED
  Supported by B2's exact reconciliation with the raw packet stream (Step 6) and its bounded compression ratio -- an event encoder would see nearly all packet-level information. Not chosen outright because it discards the explicit, cheap-to-compute window-level topology summary that Representation A already provides for free.
- **candidate_B_graph_encoder_only**: HYPOTHESIZED
  Phase 9F already showed graph richness is host-dependent (30.9%-80.5% non-empty windows) and windows stay small (max 16 edges) -- a graph-only encoder would be working with a highly variable, often-empty input, which Phase 9F itself flagged as a limitation.
- **candidate_C_event_plus_window_fused**: HYPOTHESIZED -- most consistent with measured evidence so far
  This phase's Representation C (Step 8) is a concrete, measured instance of this candidate's INPUT side (not the fusion mechanism, which is out of scope here). It is the only candidate that keeps both the exact reconciling event information (Step 6) AND an explicit per-window topology summary, at a measured storage cost shown in Section 8 of the report.
- **candidate_D_hierarchical_event_window_network**: HYPOTHESIZED
  A plausible extension of C with an additional network-wide aggregation level above the window level; this phase did not construct or measure a network-wide layer, so this candidate is less grounded in this phase's actual measurements than C.

None of these candidates were built or evaluated as models. This section reports which candidate's INPUT-side data requirements are best supported by the measurements in Sections 3-9 of the report -- not which candidate would predict better.

## 11. NOVELTY IMPLICATIONS

**EXISTING TECHNIQUE:**
- Representing network traffic as a sequence of communication events (flow records, connection logs, NetFlow/IPFIX-style records) is long-established practice in network intrusion detection -- this is not a novel idea on its own.
- Micro-batching/aggregating packets sharing a 5-tuple within a small time bucket is structurally similar to how flow exporters already aggregate packets into flow records (e.g. this project's own FlowFeatureEngine, which segments by a timeout rather than a fixed bucket) -- the aggregation CONCEPT is not new.
- Fixed-window graph snapshots of host communication are also well-established in network graph analysis literature.

**OUR POTENTIAL SYSTEM DIFFERENTIATOR:**
- NOT the event representation itself, but the specific combination targeted at this project's stated architecture: FLOW STATE + PACKET GRAPH -> MULTIMODAL NETWORK STATE -> WORLD MODEL -> FUTURE TRAJECTORY, with the packet-derived branch specifically designed (Representation C) to carry both exact reconciling event detail AND cheap topology summaries at a measured, bounded storage cost.
- The deterministic, per-phase-consistent anonymization scheme (same SHA-256-based mapping reused unmodified since Phase 9E) means events, graph snapshots, and any future hybrid representation all reference the SAME anonymized host identity across representations -- a coherence property a naive combination of off-the-shelf flow exporter output and a separately-built graph tool would not automatically have.

**NOT YET VERIFIED:**
- Whether any of these representations improve multi-step attack trajectory forecasting over the existing flow-level World Model (Phase 6B) -- no model was trained here.
- Whether uncertainty quantification or counterfactual intervention are supportable by these representations -- not evaluated.
- Whether the observed host-dependent variability (Phase 9F) and the trivial per-event port diversity (Step 7 caveat) cause any specific practical difficulty for a real encoder -- only representation-level structure was measured, not encoder behavior.
- Generalization beyond these two single-host, single-day captures.

## 12. LIMITATIONS

- Only the same two single-host, single-day captures already used in Phases 9D-9F were examined; no new PCAP was downloaded per hard constraint #10, so generalization beyond these two hosts is NOT VERIFIED.
- No model of any kind was trained or evaluated; all 'suitability for future world-model input' and Step 10 candidate assessments are HYPOTHESIZED, not measured predictive results.
- Source/destination-port diversity is trivially 1 for every B2 event by construction (ports are part of the grouping key) -- this is reported explicitly rather than silently included as if informative at the per-event level.
- The hybrid representation (C) is defined and size-measured only; no fusion mechanism (neural or otherwise) was implemented, per hard constraint #6.
- TTL, IP fragmentation, and TCP window size are not carried into B2 event features (not in the required Step 4 feature list); this is a scoping choice, not evidence they are unavailable -- Phase 9E already classified them DIRECT and available.
- B2's per-event compression ratio (packets-per-event) was modest in both captures ([1.32, 2.32]x) because median burst size is 1.0 packet/event in both -- most 1-second buckets never accumulate more than one packet for a given (src, dst, protocol, port pair) 5-tuple. As a direct consequence, B2's serialized size is LARGER than B1's raw per-packet events in 2/2 captures -- reducing event count does not automatically reduce storage when per-event overhead exceeds the compression gained.

## 13. RECOMMENDED NEXT EXPERIMENT

If pursued further (not started here): implement candidate C's fusion mechanism as a small, non-neural baseline first (e.g. concatenating a window's event-derived statistics with its window-summary vector) and compare information content against the flow-level World Model's existing state representation on a held-out capture, before committing to any neural encoder architecture.

## FINAL VERDICT: GREEN

STOP AFTER PHASE 9G.
