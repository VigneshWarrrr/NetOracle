# Phase 9F: Cross-Host Temporal Graph Generalization

Verdict: **YELLOW**

## 1. SECOND MEMBER SELECTION

- CONFIRMED: filename `pcap/capWIN-J6GMIG1DQE5-172.31.64.89`
- CONFIRMED: compressed_size = 6,139,533 bytes (5.86 MiB)
- CONFIRMED: uncompressed_size = 10,894,363 bytes
- CONFIRMED: compression_method = 8
- CONFIRMED: local_header_offset = 38,025,205,144
- Selected from Phase 9C's frozen discovery_report.json (never re-parsed from the archive). Selection rule: among all 9 candidate members with compressed_size in the requested [5,242,880, 7,340,032] byte band (excluding the Phase 9D member pcap/UCAP172.31.69.22), the SMALLEST was chosen -- a deterministic, content-blind tie-break that minimizes bytes downloaded while staying inside the requested band. No filename, hostname, or presumed attack/benign association was used in this decision; the only input to selection was the compressed_size field.

## 2. DOWNLOAD VALIDATION

- CONFIRMED: total bytes downloaded this phase: 0
- CONFIRMED: filename match: True
- CONFIRMED: compressed size match: True

## 3. CAPTURE VALIDATION

- CONFIRMED: detected format: pcap (microsecond resolution, little-endian byte order)
- CONFIRMED: uncompressed size match: True
- CONFIRMED: CRC-32 match: True
- CONFIRMED: first 32 bytes: `d4c3b2a1020004000000000000000000ffff000001000000692b845aaba80300`

## 4. SECOND CAPTURE STATISTICS

- OBSERVED: total_packets_in_capture = 49250
- OBSERVED: total_ip_packets = 46295
- OBSERVED: tcp_packet_count = 44557
- OBSERVED: udp_packet_count = 800
- OBSERVED: icmp_packet_count = 840
- OBSERVED: other_packet_count = 98
- OBSERVED: unique_source_ips = 593
- OBSERVED: unique_destination_ips = 237
- OBSERVED: unique_hosts_overall = 597
- OBSERVED: total_bytes = 9896715
- OBSERVED: first_timestamp = 1518611305.239787
- OBSERVED: last_timestamp = 1518643841.321168
- OBSERVED: capture_duration_seconds = 32536.081380844116

## 5. SECOND TEMPORAL GRAPH STATISTICS

- CONFIRMED: total_windows = 3254, non_empty_windows = 2620
- CONFIRMED: total_nodes_observed = 597
- CONFIRMED: total_unique_directed_source_destination_pairs = 828
- CONFIRMED: packet_accounting_check matches = True

Per-window distributions (min / median / mean / p95 / max):

| metric | min | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| node_count | 0.0 | 2.0 | 2.504302397049785 | 5.0 | 9.0 |
| edge_count | 0.0 | 2.0 | 3.104179471419791 | 8.0 | 16.0 |
| packet_count | 0.0 | 10.0 | 14.227105101413645 | 44.0 | 984.0 |
| byte_count | 0.0 | 1038.0 | 3041.399815611555 | 7451.75 | 2237873.0 |
| density | 0.16666666666666666 | 0.6666666666666666 | 0.6726984126984127 | 1.0 | 1.0 |
| average_degree | 1.0 | 2.0 | 2.3121567914697687 | 3.2 | 3.5555555555555554 |
| mean_in_degree | 0.5 | 1.0 | 1.1560783957348844 | 1.6 | 1.7777777777777777 |
| mean_out_degree | 0.5 | 1.0 | 1.1560783957348844 | 1.6 | 1.7777777777777777 |
| max_in_degree | 1.0 | 2.0 | 2.0595419847328245 | 5.0 | 8.0 |
| max_out_degree | 1.0 | 1.0 | 1.869083969465649 | 4.0 | 8.0 |
| max_degree | 1.0 | 3.0 | 3.8553435114503816 | 9.0 | 16.0 |
| isolated_node_count | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| weakly_connected_component_count | 0.0 | 1.0 | 0.8051628764597418 | 1.0 | 1.0 |

- OBSERVED: longest_consecutive_non_empty_window_run = 71
- OBSERVED: mean node-set Jaccard (consecutive) = 0.3214501662929954
- OBSERVED: mean edge-set Jaccard (consecutive) = 0.21081104010662968

## 6. DIRECT COMPARISON WITH PHASE 9E

| metric | Phase 9D/9E capture | second capture | ratio (second/first) |
|---|---:|---:|---:|
| packet_count (IP packets) | 7316 | 46295 | 6.327911427009295 |
| duration_seconds | 32220.89963197708 | 32536.081380844116 | 1.0097819040581424 |
| unique_hosts | 551 | 597 | 1.0834845735027223 |
| unique_directed_pairs | 1094 | 828 | 0.7568555758683729 |
| median_nodes_per_window | 0.0 | 2.0 | None |
| p95_nodes_per_window | 3.0 | 5.0 | 1.6666666666666667 |
| max_nodes_per_window | 5.0 | 9.0 | 1.8 |
| median_edges_per_window | 0.0 | 2.0 | None |
| p95_edges_per_window | 4.0 | 8.0 | 2.0 |
| max_edges_per_window | 8.0 | 16.0 | 2.0 |
| non_empty_window_fraction | 0.30933912503878375 | 0.8051628764597418 | 2.602848496318704 |
| longest_non_empty_run | 8 | 71 | 8.875 |
| mean_edge_jaccard_consecutive | 0.03034647930608763 | 0.21081104010662968 | 6.946803877322933 |
| mean_node_jaccard_consecutive | 0.08542431660546716 | 0.3214501662929954 | 3.762982006371972 |
| sparsity_fraction | 0.9963900346477479 | 0.9976729284003912 | 1.0012875417336915 |

Classification: **B** -- Evidence is mixed, not forced into a single bucket: richness-metric ratios range from 0.76x to 8.88x (second/first). Temporal-continuity metrics grew substantially (non_empty_window_fraction 2.60x, longest_non_empty_run 8.88x, edge/node Jaccard 6.95x / 3.76x), while per-window structural density stayed modest (max_nodes_per_window 1.80x, max_edges_per_window 2.00x) and unique_directed_pairs actually decreased (0.76x) despite ~6x more packets -- the pipeline generalizes correctly across hosts, and temporal richness improves meaningfully, but per-window population remains variable rather than uniformly tiny or uniformly rich.

## 7. GRAPH-MODEL SUITABILITY

- **q1_enough_structure_for_meaningful_gnn**: MARGINAL
  OBSERVED: median nodes/window in the second capture is 2.0, and max edges/window is 16.0 -- individual windows carry very few edges for message passing to aggregate over, in both captures (classification: B).
  HYPOTHESIZED: a GNN could still exploit structure aggregated across MULTIPLE windows or across hosts (not attempted here), even if any single window's graph is small.
- **q2_snapshots_sufficiently_populated**: PARTIALLY, AND INCONSISTENTLY BETWEEN THE TWO CAPTURES
  OBSERVED: the non-empty-window fraction differs sharply between the two captures rather than agreeing: 30.9% of 10-second windows are non-empty in the first capture (a minority) versus 80.5% in the second capture (a majority). Whether most snapshots carry any structure at all is host-dependent in this small two-capture sample, not a fixed property of the representation.
- **q3_sufficiently_long_temporal_sequences**: LIMITED
  OBSERVED: longest consecutive non-empty window run in the second capture is 71 windows (710s); sequences this short bound how much temporal context a sequence model could condition on before hitting a gap.
- **q4_would_graph_model_add_info_beyond_aggregate_stats**: NOT VERIFIED
  OBSERVED: this phase computed aggregate statistics (the same min/median/mean/p95/max distributions used throughout) and did not compare their predictive content against any graph-model output, because no model was built here.
  HYPOTHESIZED: given how small and sparse individual window graphs are (mean node-set Jaccard 0.3214501662929954 in the second capture, meaning topology itself changes substantially), a graph model's main candidate advantage would be capturing WHICH specific hosts are connected, not raw volume -- something scalar aggregate statistics cannot represent by construction. This is a hypothesis, not a measured result.
- **q5_which_representation_fits_the_observed_data**: HYPOTHESIZED: an event/edge-sequence representation (a stream of individual (timestamp, src, dst, features) edge events) looks like a better structural fit than dense per-window graph snapshots, given both captures show most windows empty and populated windows carry only a handful of edges -- a sequence representation avoids paying a fixed per-window cost for windows with little or no content.
  OBSERVED: a global/static graph would discard all temporal-change information that Phase 9E and this phase both measured as substantial (low consecutive-window Jaccard in both captures).

## 8. NOVELTY IMPLICATION

Decision: **CONDITIONAL**

Graph construction generalizes correctly across two different hosts (same pipeline, same conventions, both produce internally consistent, exact statistics). Evidence is mixed rather than uniformly weak or uniformly strong: temporal-continuity metrics (non-empty-window fraction, longest non-empty run, consecutive-window Jaccard) improved substantially in the second, larger capture, while per-window structural density (max nodes/edges per window, unique directed pairs) stayed modest or flat. This pattern -- richer WHEN traffic occurs, but not necessarily richer WHERE/HOW MANY hosts per window -- is exactly the kind of result that should not be forced toward a full GNN commitment yet. An event/edge-sequence representation (per Step 8, Q5) should be tried and evaluated first.

This decision is NOT based on 'GNN sounds novel' -- it is based only on the measured sparsity, window population, and temporal-run-length numbers from these two captures.

## 9. LIMITATIONS

- Only two endpoint captures have now been examined out of 449 members in the archive; two data points establish that the pipeline generalizes MECHANICALLY (same code runs correctly on a different host) but is a very small basis for claiming statistical generalization of graph richness across the archive.
- No attack labels exist for either capture; nothing here reflects attack detectability.
- The second capture is from the same day (Wednesday-14-02-2018) and the same AWS environment as the first; cross-day generalization is NOT VERIFIED.
- Graph-model suitability answers in Section 7 are grounded in measured sparsity/window-population/run-length numbers, but no model was trained or evaluated -- the Q4 predictive-value question is explicitly NOT VERIFIED.
- Raw per-packet records for the second capture are not persisted beyond the single decompressed .pcap file under C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\data\phase9f (not committed as a tracked result); only the JSON/Markdown reports are placed under results/.

## 10. RECOMMENDED NEXT EXPERIMENT

If pursued further (not started here): before any GNN work, prototype the event/edge-sequence representation identified in Step 8 Q5 (a chronological stream of (timestamp, src, dst, edge_features) events rather than fixed per-window graph snapshots) against a third capture, and compare its ability to represent the same information more compactly than the sparse per-window snapshot approach used in Phase 9E/9F.

## FINAL VERDICT: YELLOW

STOP AFTER PHASE 9F.
