# Phase 9E: Temporal Packet Graph Construction & Feasibility

## 1. EXECUTIVE RESULT

Verdict: **GREEN**

A temporal, windowed, directed communication graph was constructed from the single validated PCAP member `pcap/UCAP172.31.69.22` (7316 IP packets across 3223 10-second windows, 997 of them non-empty). CONFIRMED: graph construction is exact and internally consistent (packet-accounting check: True). OBSERVED: the graph changes meaningfully over time (see section 9). NOT VERIFIED: predictive value, cross-host/cross-day generalization, or any attack relevance.

## 2. INPUT PCAP

- Path: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\data\phase9d\UCAP172.31.69.22.pcap`
- Source: Phase 9D validated artifact (already retrieved/decompressed; no network access in this phase)
- Capture span: 1518611531.157735 to 1518643752.057367 (32220.900s)
- Cross-check against Phase 9D persisted aggregates: True

## 3. PARSING RESULTS

- Total packets in capture: 10417
- IP packets (parsed into graph): 7316
- Status counts: {'ok': 7316, 'non_ip': 3101}

## 4. TEMPORAL WINDOW STRUCTURE

- Window size: 10s (CONFIRMED reused project convention)
- Total windows: 3223

## 5. GRAPH STATISTICS

- total_windows: 3223
- non_empty_windows: 997
- empty_windows: 2226
- total_nodes_observed: 551
- total_unique_directed_source_destination_pairs: 1094
- unique_edges_across_entire_capture: 1094
- unique_edges_note: Identical to total_unique_directed_source_destination_pairs above -- both are the count of the set of distinct (src, dst) directed pairs observed anywhere in the capture. Reported under both labels for direct correspondence with the Phase 9E specification.
- total_edges_across_windows: 3052
- total_edges_across_windows_note: Counts a given (src, dst) pair once per window it appears in (i.e. window-local edge instances), unlike the unique-pair count above which counts each pair once regardless of how many windows it appears in.
- total_packets_represented: 7316
- total_bytes_represented: 963452

Per-window distributions (min / median / mean / p95 / max):

| metric | windows w/ value | min | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| node_count | 3223 | 0.0 | 0.0 | 0.7942910331988831 | 3.0 | 5.0 |
| edge_count | 3223 | 0.0 | 0.0 | 0.9469438411417933 | 4.0 | 8.0 |
| packet_count | 3223 | 0.0 | 0.0 | 2.269934843313683 | 11.0 | 218.0 |
| byte_count | 3223 | 0.0 | 0.0 | 298.93018926466027 | 1029.0 | 217503.0 |
| density | 997 | 0.35 | 1.0 | 0.8132062855232364 | 1.0 | 1.0 |
| average_degree | 997 | 1.0 | 2.0 | 2.278468739551989 | 3.0 | 3.2 |
| mean_in_degree | 997 | 0.5 | 1.0 | 1.1392343697759946 | 1.5 | 1.6 |
| mean_out_degree | 997 | 0.5 | 1.0 | 1.1392343697759946 | 1.5 | 1.6 |
| max_in_degree | 997 | 1.0 | 1.0 | 1.530591775325978 | 3.0 | 4.0 |
| max_out_degree | 997 | 1.0 | 1.0 | 1.5576730190571715 | 3.0 | 4.0 |
| max_degree | 997 | 1.0 | 2.0 | 3.061183550651956 | 6.0 | 8.0 |
| isolated_node_count | 3223 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| weakly_connected_component_count | 3223 | 0.0 | 0.0 | 0.30933912503878375 | 1.0 | 1.0 |

## 6. NODE FEATURES

Per (window, anonymized host): packet_count, bytes_sent, bytes_received, unique_destinations, unique_sources, unique_destination_ports, unique_source_ports, protocol_counts (TCP/UDP/ICMP/OTHER), first_seen_offset, last_seen_offset. Topology (graph edges) is kept strictly separate from these observational node features.

## 7. EDGE FEATURES

Per (window, src->dst): packet_count, total_bytes, mean/min/max/std packet_length, first/last_packet_offset, edge_duration, packets_per_second, bytes_per_second, mean/std inter-arrival time, TCP/UDP/ICMP/OTHER packet counts, source/destination port diversity, TCP SYN/ACK/FIN/RST counts. Rate and IAT fields are `null` when mathematically undefined (single packet, zero-duration edge) rather than fabricated as zero.

## 8. PACKET FEATURE AVAILABILITY

| feature | available | coverage | classification |
|---|---|---:|---|
| timestamp | True | 100.0% | DIRECT |
| packet length | True | 100.0% | DIRECT |
| source/destination IP | True | 70.2% | DIRECT |
| protocol | True | 70.2% | DIRECT |
| ports | True | 67.1% | DIRECT |
| TTL | True | 70.2% | DIRECT |
| TCP flags | True | 41.1% | DIRECT |
| TCP window | True | 41.1% | DIRECT |
| payload length | True | 70.2% | DIRECT |
| fragmentation | True | 70.2% | DIRECT |
| inter-arrival time (IAT) | True | 56.8% | DIRECT |
| retransmission indicators | False | 0.0% | NOT IMPLEMENTED |
| connection indicators (e.g. completed handshake) | False | 0.0% | NOT IMPLEMENTED |
| port-scan indicators | False | 0.0% | NOT IMPLEMENTED |

## 9. TEMPORAL DYNAMICS

- Longest consecutive non-empty window run: 8
- Mean node-set Jaccard (consecutive windows): 0.08542431660546716
- Mean edge-set Jaccard (consecutive windows): 0.03034647930608763
- Mean node churn per consecutive pair: 1.2780881440099316
- Mean edge churn per consecutive pair: 1.7566728739913098
- Low mean Jaccard similarity between consecutive windows' node/edge sets indicates the graph's membership changes substantially window-to-window (a genuinely temporal graph); a mean near 1.0 would indicate a near-static graph merely repeated across time.

## 10. REPRESENTATION FEASIBILITY

- Representation A (edge-list snapshots): max nodes/window = 5, max edges/window = 8, node_feature_dim = 13, edge_feature_dim = 23
- Representation B (fixed-dim state): feasible = True, estimated_dim = 12
- Sparsity: 0.9963900346477479
- Serialized artifact bytes: 4647750
- No representation is selected as superior here, per Phase 9E scope; this is a feasibility characterization only.

## 11. CSV RELATIONSHIP

- csv_exists_for_this_day: True
- endpoint_identity_present_in_csv: False
- protocol_column_present: True
- timestamp_column_present: True
- row_level_correspondence_established: False
- No flow-row correspondence was established.

## 12. MULTIMODAL ARCHITECTURE IMPLICATIONS

- **topology_change**: OBSERVED: Mean consecutive-window node-set Jaccard similarity is 0.08542431660546716, and edge-set Jaccard is 0.03034647930608763 (over 1634 defined window pairs) -- the graph's membership is not static across windows in this single capture.
  HYPOTHESIZED: A world model could consume per-window topology-change signals (churn, Jaccard) as an auxiliary input alongside flow state, on the hypothesis that topology instability precedes or accompanies anomalous activity.
- **communication_expansion**: OBSERVED: 551 distinct hosts and 1094 unique directed pairs were observed from a single endpoint's capture over ~8.95 hours.
  HYPOTHESIZED: Sudden growth in unique_destinations for a node (already a DIRECT node feature) is a plausible predictive signal for future models; not evaluated for predictive value here.
- **host_centrality_changes**: OBSERVED: Per-window degree distributions (min/median/mean/p95/max) are computed in graph_statistics.per_window_distributions and show variation across windows.
  HYPOTHESIZED: Centrality shifts (e.g. a normally low-degree host suddenly gaining many peers) could be a future world-model feature.
- **port_behavior**: OBSERVED: Per-edge source/destination port diversity is computed exactly per window; not aggregated into a capture-wide claim here.
  HYPOTHESIZED: Port-diversity spikes could feed a future (not-yet-built) heuristic or learned scan-detection signal.
- **protocol_behavior**: OBSERVED: TCP/UDP/ICMP packet counts are tracked per edge and per node window; see graph_statistics and node/edge feature schemas.
  HYPOTHESIZED: Protocol-mix shift per window is a natural complementary signal to the flow-level protocol features already used elsewhere in this project.
- **packet_volume_changes**: OBSERVED: Per-window packet_count and byte_count distributions are reported exactly (min/median/mean/p95/max).
  HYPOTHESIZED: Volume bursts relative to a host's own baseline could contribute to a future anomaly signal.
- **burst_iat_behavior**: OBSERVED: Mean/std inter-arrival time is computed exactly per edge (where >=2 packets exist on that edge in that window).
  HYPOTHESIZED: IAT compression (packets arriving unusually close together) is a classic burst indicator that could feed a future model.
- **flow_vs_packet_complementary_information**: OBSERVED: This capture carries fields the existing flow-level CSV pipeline does not retain post-aggregation: per-packet TTL, TCP window, IP fragmentation flags, and exact per-packet timestamps rather than flow-level start/end times.
  HYPOTHESIZED: These packet-only fields are the concrete candidate contribution of the PACKET GRAPH branch in the architecture diagram, complementary to (not a replacement for) FLOW STATE.

No predictive value is claimed for any signal above; all items are either directly observed structural facts about this one capture or explicitly labeled hypotheses for future work.

## 13. NOVELTY IMPLICATIONS

This phase does NOT claim novelty merely from using a graph representation. It assesses only whether the packet modality is technically CAPABLE of supporting the intended future components.

Overall: The packet modality provides a credible, technically grounded STRUCTURAL foundation (exact windowed graph with real topology change) for the future multimodal concept, but this single-endpoint, unlabeled experiment verifies structure only -- not predictive value, not generalization, and not the downstream components (trajectory/uncertainty/counterfactual) themselves.

## 14. LIMITATIONS

- This entire experiment is scoped to ONE endpoint's capture (pcap/UCAP172.31.69.22); no claim is made about the other 448 archive members or the dataset as a whole.
- No attack label exists for any window in this capture; nothing here should be read as evidence of attack or benign traffic.
- IPv6 is not implemented (none observed in this capture; a non-IPv4 packet is classified as ipv6_not_implemented or non_ip, not dropped silently).
- Retransmission indicators, connection-completion indicators, and port-scan indicators are NOT IMPLEMENTED (would require heuristic, threshold-based judgment, out of scope for this phase).
- isolated_node_count is empirically 0.0 (max across windows) confirming the by-construction expectation that every node in a window's graph has degree >= 1.
- Anonymization is deterministic pseudonymization for artifact hygiene, not cryptographically strong privacy protection (see anonymization_validation.privacy_caveat in the JSON report).
- Raw per-packet records are not persisted to disk (only the anonymized, window-aggregated graph artifact is written), per the 'do not store raw PCAP-derived data unnecessarily' output constraint.
- Packet accounting check: True (sum of per-edge-window packet counts equals total IP packets parsed).

## 15. RECOMMENDED NEXT EXPERIMENT

If pursued further (not started here): parse a second, larger member (e.g. one of the capWIN-J6GMIG1DQE5-* hosts) with this same pipeline and compare its graph-statistics distributions against this host's, to determine whether the temporal/topological patterns observed here (non-trivial churn, sparse but multi-host structure) generalize across hosts before any model architecture is chosen.

## FINAL VERDICT: GREEN

STOP AFTER PHASE 9E.
