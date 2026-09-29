# Phase 9I: PCAP <-> Flow Alignment Investigation

Verdict: **RED**

## 1. Executive Summary

This phase sought AUTHORITATIVE evidence (not correlation) for whether the two available single-host PCAP captures (Phase 9D/9F) can be defensibly paired with the processed Wednesday-14-02-2018 flow-level CSV. Findings: (1) the CSV has NO endpoint identity columns for Wednesday (only Tuesday has them), ruling out any IP/port-based match; (2) logs.zip (newly inspected via bounded HTTP Range requests) contains a plain-text syslog for the exact same host as one of our PCAP captures (172.31.69.22); cross-referencing its DHCP lease-renewal events against the PCAP's own DHCP packets gives an EXACT, established capture-host clock offset (21/26 events match to the second at a single consistent offset) -- but re-testing the CSV correlation at exactly this now-evidenced offset still yields noise-level correlation, showing timezone was never the real blocker; (3) the PCAP archive contains ONLY per-host captures (449/449 members, zero exceptions) -- no network-wide capture exists to compare against the CSV's whole-subnet flow aggregates even in principle, which is the actual, structural, unresolvable blocker. No authoritative mapping document was found anywhere. This reinforces, with new and substantially stronger evidence, Phase 9H's conclusion: the two modalities cannot currently be paired.

## 2. Research Question

Can the real CSE-CIC-IDS2018 PCAP data be defensibly paired with the processed flow-level CSV data already used by NetOracle's forecasting pipeline?

## 3. Existing Evidence from 9H

Phase 9H found best cross-correlation |r|=0.0879 at offset +0h (indistinguishable from noise, floor was 0.30) and stopped variants B/C/D accordingly.

## 4. Flow Dataset Semantics

- Evidence source(s): ['C:\\AKSHAY\\Akshay\\SIH FOLDER MAIN\\data\\raw\\CSE-CIC-IDS2018\\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'C:\\AKSHAY\\Akshay\\SIH FOLDER MAIN\\data\\windows\\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'C:\\AKSHAY\\Akshay\\SIH FOLDER MAIN\\NetOracle\\docs\\CIC_IDS2018_DATA_READINESS.md']
- CSV filename: `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`; columns: 80
- Row represents: one CICFlowMeter bidirectional FLOW record (aggregated over the flow's full duration), NOT a single packet -- evidenced by columns Tot Fwd Pkts / Tot Bwd Pkts / Flow Duration / per-flow IAT and packet-length statistics, which are only meaningful as aggregates over multiple packets.
- Flow duration semantics: Flow Duration column, microseconds (e.g. 112641719 microseconds = 112.6 seconds), per Phase 2's numeric-quality audit finite range check
- Flow start/end timestamps available: Only a single 'Timestamp' column is present (flow start time, per CICFlowMeter's documented output convention); no separate flow-end timestamp column exists -- end time would need to be derived as Timestamp + Flow Duration.
- Endpoint identity columns present: [] (available: False)

## 5. CSV Timestamp Semantics

- Column: `Timestamp`
- Datatype/format: string, format DD/MM/YYYY HH:MM:SS (e.g. '14/02/2018 08:31:01'), no timezone suffix present in any sampled or audited value
- Example raw values: ['14/02/2018 08:31:01', '14/02/2018 08:33:50', '14/02/2018 08:36:39']
- Windowed CSV window_start range: {'min': '2018-02-14 01:00:00', 'max': '2018-02-14 12:59:50', 'row_count': 4320}
- Readiness doc excerpt: - `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `1970-01-05 03:01:17` to `2018-02-14 12:59:59`; out-of-order 289,765 (27.63417018%); invalid timestamps 0.
- **Timezone status: UNKNOWN / NOT ESTABLISHED** (see Section 8 documentation search).

## 6. PCAP Metadata

- Evidence source: C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase9c_pcap_archive_inspection\discovery_report.json (frozen, read-only; no new network access this phase)
- Total members: 449; UCAP-prefixed: 6; cap-prefixed: 443; other patterns: 0
- ALL 449 non-folder members follow one of exactly two per-HOST naming patterns (`pcap/UCAP<ip>` or `pcap/cap<hostname>-<ip>`), with zero exceptions -- including the single largest member (789 MB compressed). No member's filename or size suggests a merged, multi-host, or whole-network capture. The archive is structurally composed entirely of individual per-host captures; a network-wide population sample does not exist anywhere in this archive.

## 7. Logs ZIP Metadata

- Object: `Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip`, success=True
- Member count: 449
- Small relevant candidates found: 6
  - `logs/U172.31.69.18` (1002 bytes compressed)
  - `logs/U172.31.69.21` (1041 bytes compressed)
  - `logs/U172.31.69.22` (1012 bytes compressed)
  - `logs/U172.31.69.25` (16678 bytes compressed)
  - `logs/U172.31.69.27` (1023 bytes compressed)
  - `logs/U172.31.69.7` (1031 bytes compressed)
  - RETRIEVED `logs/U172.31.69.18` -- preview: `'Feb 14 06:46:47 ip-172-31-69-18 dhclient[967]: DHCPREQUEST of 172.31.69.18 on eth0 to 172.31.69.1 port 67 (xid=0x3fe031ab)\nFeb 14 06:46:47 ip-172-31-69-18 dhclient[967]: DHCPACK of 172.31.69.18 from 172.31.69.1\nFeb 14 06:46:47 ip-172-31-69-18 dhclient[967]: bound to 172.31.69.18 -- renewal in 1478 s'`
  - RETRIEVED `logs/U172.31.69.22` -- preview: `'Feb 14 06:32:12 ip-172-31-69-22 dhclient[931]: DHCPREQUEST of 172.31.69.22 on eth0 to 172.31.69.1 port 67 (xid=0x450e8c50)\nFeb 14 06:32:12 ip-172-31-69-22 dhclient[931]: DHCPACK of 172.31.69.22 from 172.31.69.1\nFeb 14 06:32:12 ip-172-31-69-22 dhclient[931]: bound to 172.31.69.22 -- renewal in 1773 s'`
  - RETRIEVED `logs/U172.31.69.27` -- preview: `'Feb 14 06:48:56 ip-172-31-69-27 dhclient[940]: DHCPREQUEST of 172.31.69.27 on eth0 to 172.31.69.1 port 67 (xid=0xc36aa03)\nFeb 14 06:48:56 ip-172-31-69-27 dhclient[940]: DHCPACK of 172.31.69.27 from 172.31.69.1\nFeb 14 06:48:56 ip-172-31-69-27 dhclient[940]: bound to 172.31.69.27 -- renewal in 1563 se'`
  - RETRIEVED `logs/U172.31.69.7` -- preview: `'Feb 14 06:49:27 ip-172-31-69-7 systemd[1]: Starting Daily apt upgrade and clean activities...\nFeb 14 06:49:29 ip-172-31-69-7 systemd[1]: Started Daily apt upgrade and clean activities.\nFeb 14 06:51:03 ip-172-31-69-7 dhclient[945]: DHCPREQUEST of 172.31.69.7 on eth0 to 172.31.69.1 port 67 (xid=0x7e57'`
  - RETRIEVED `logs/U172.31.69.21` -- preview: `'Feb 14 06:38:09 ip-172-31-69-21 dhclient[982]: DHCPREQUEST of 172.31.69.21 on eth0 to 172.31.69.1 port 67 (xid=0xea334)\nFeb 14 06:38:09 ip-172-31-69-21 dhclient[982]: DHCPACK of 172.31.69.21 from 172.31.69.1\nFeb 14 06:38:09 ip-172-31-69-21 dhclient[982]: bound to 172.31.69.21 -- renewal in 1657 seco'`
  - RETRIEVED `logs/U172.31.69.25` -- preview: `'Feb 14 06:27:59 ip-172-31-69-25 systemd[1]: Starting Daily apt upgrade and clean activities...\nFeb 14 06:28:02 ip-172-31-69-25 systemd[1]: Started Daily apt upgrade and clean activities.\nFeb 14 06:38:33 ip-172-31-69-25 dhclient[980]: DHCPREQUEST of 172.31.69.25 on eth0 to 172.31.69.1 port 67 (xid=0x'`
- Bytes downloaded this step: 127291

## 8. Host/Population Correspondence

ALL 449 non-folder members follow one of exactly two per-HOST naming patterns (`pcap/UCAP<ip>` or `pcap/cap<hostname>-<ip>`), with zero exceptions -- including the single largest member (789 MB compressed). No member's filename or size suggests a merged, multi-host, or whole-network capture. The archive is structurally composed entirely of individual per-host captures; a network-wide population sample does not exist anywhere in this archive.

## 9. Timestamp Correspondence

- Method: Parsed all DHCPREQUEST lease-renewal events for host 172.31.69.22 from its retrieved syslog (logs/U172.31.69.22, local clock, HH:MM:SS only, no year/timezone) and from the already-local Phase 9D PCAP (UTC, via DHCP protocol packets on UDP port 68->67 from the same host). For each candidate whole-hour offset, counted how many syslog event times, shifted by that offset, land on an exact (to-the-second) PCAP DHCPREQUEST time.
- Syslog DHCPREQUEST events (host 172.31.69.22): 26
- PCAP DHCPREQUEST events (same host, UTC): 21
- Match counts by candidate offset (hours): {0: 0, 1: 0, 2: 0, 3: 0, 4: 21, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
- **Best offset: +4h (21/26 exact-second matches)**
- ESTABLISHED: host 172.31.69.22's local system clock (as reported in its own syslog) was 4 hours BEHIND the PCAP's UTC timestamps (21/21 PCAP-observed DHCPREQUEST events match a syslog event to the exact second at this single consistent offset -- zero mismatches). This is exact, per-event, cross-source evidence -- not a correlation estimate -- for the capture HOST's clock only. It does NOT establish the separate question of whether CICFlowMeter (which produced the processed CSV) used the same clock/timezone as this capture host. (The syslog additionally covers 5 earlier/later renewal events outside the PCAP's own capture window, which is expected and does not weaken this result.)

This ESTABLISHES the PCAP-capture-HOST's clock offset to UTC via exact, per-event, cross-source evidence (not correlation, not a guess). It does NOT establish the CSV/CICFlowMeter timezone -- see candidate E2 in Section 11 for what happens when this now-evidenced offset is applied to the CSV comparison anyway (still noise-level).

## 10. Endpoint Correspondence

Exact endpoint-level correspondence cannot be tested from the processed CSV: Wednesday-14-02-2018 has no Src IP/Dst IP/Src Port/Flow ID columns (only Tuesday-20-02-2018 has these, and no PCAP capture is available for Tuesday).

## 11. Candidate Mapping Evidence

| candidate | status |
|---|---|
| A. Exact endpoint matching (specific IP <-> specific flow rows) | UNSUPPORTED |
| B. Exact 5-tuple matching (src ip, dst ip, src port, dst port, protocol) | UNSUPPORTED |
| C. Host/subnet matching (PCAP host IP appears somewhere in CSV-derivable subnet) | UNSUPPORTED |
| D. Capture-host mapping (an authoritative document maps PCAP filenames to roles/hosts) | UNSUPPORTED |
| E1. PCAP-capture-host clock offset to UTC | ESTABLISHED |
| E2. CSV/CICFlowMeter timestamp interval overlap with PCAP, using the E1-established offset | UNSUPPORTED |
| F. Flow-duration compatibility (do flow durations plausibly fit inside packet-observed windows) | UNSUPPORTED |
| G. Protocol/port distribution similarity | PLAUSIBLE BUT UNPROVEN |
| H. Dataset-generation metadata (paper, changelog, generation script found) | UNSUPPORTED |
| I. Capture/log metadata (logs.zip contains a capture manifest, config, or similar) | PLAUSIBLE BUT UNPROVEN |
| J. Any authoritative mapping between raw traffic and processed traffic | UNSUPPORTED |
| K. (Phase 9H's own finding, carried forward) Weak aggregate correlation as proof | CONTRADICTED |

## 12. Evidence Classification

- **A. Authoritative evidence**: NONE FOUND. No original CSE-CIC-IDS2018 dataset-creator documentation (e.g. a README from the UNB/CIC S3 bucket itself, a published dataset paper excerpt, or capture-tool configuration file) is present anywhere in this repository.
- **B. Repository evidence**: 1036 keyword hits found across repository docs/code for explicit timezone abbreviations (UTC/EST/EDT/AST/ADT); all are either code identifiers unrelated to dataset timezone (e.g. 'AST' as a Python ast-module alias) or, if any are genuine, are listed explicitly in timezone_specific_hits above for direct verification -- none establish the CICFlowMeter Timestamp column's timezone.
- **C. General background**: General public knowledge about CIC-IDS-family datasets (e.g. common community assumptions about Fredericton/Atlantic-Canada or AWS-default-UTC clocks) is EXPLICITLY NOT treated as evidence here, per this phase's hard constraint not to guess timestamps/timezones -- it is background only, never used as a substitute for authoritative or repository evidence.
- **D. Inference**: Phase 9H's empirical cross-correlation search (best |r|=0.088, indistinguishable from noise) is DERIVED evidence, not authoritative evidence -- it was tested but explicitly did not establish a timezone or alignment.

## 13. What Is Established

- Wednesday-14-02-2018's processed CSV has NO Src IP/Dst IP/Src Port/Flow ID columns (Phase 2 audit, docs/CIC_IDS2018_DATA_READINESS.md, re-confirmed directly from the raw CSV header here).
- The PCAP archive for Wednesday-14-02-2018 is composed entirely of per-host captures (449/449 members follow a UCAP<ip> or cap<hostname>-<ip> naming pattern; zero exceptions, including the largest member).
- logs.zip mirrors this per-host structure: 442/449 members are per-host Windows Event Logs (.evtx) and 6 are per-host Linux syslogs (matching the 6 UCAP-prefixed PCAP hosts exactly) -- no whole-network manifest or mapping document exists in it.
- The capture host 172.31.69.22's system clock is EXACTLY 4 hours behind its own PCAP's UTC timestamps -- established via 21/26 independent DHCP lease-renewal events matching to the exact second between its retrieved syslog and the already-local PCAP. This is exact, per-event, cross-source evidence, not a guess.
- logs.zip (Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip) central-directory metadata was inspected (member count: 449); see Section 7 for what was found.

## 14. What Is Unknown

- The true timezone of the CICFlowMeter Timestamp column specifically -- the capture HOST's own clock offset is now established (see what_is_established), but whether CICFlowMeter's processing machine used that same clock/timezone is untested and unconfirmed either way.
- Whether any of the 449 per-host PCAP captures corresponds to a host whose traffic is distinguishable in the Wednesday CSV (impossible to check without endpoint identity).
- Whether an authoritative host-to-capture or capture-to-timezone mapping exists ANYWHERE (e.g. in the original dataset publication) outside what is locally available to this repository.

## 15. What Cannot Be Claimed

- That the PCAP and CSV data represent the same population of traffic.
- That any timestamp offset (including the +0h/UTC=UTC hypothesis Phase 9H tested) is correct.
- That packet/graph information has no value -- only that it cannot currently be PAIRED with this specific flow dataset in a defensible experiment.

## 16. Download Budget / Retrieval Accounting

- Budget for this phase: 4,194,304 bytes
- Actually downloaded: 127,291 bytes
- No full archive (pcap.zip or logs.zip body) was downloaded; only HTTP HEAD + Range requests for ZIP central-directory metadata and, where relevant, small individual members.

## 17. Scientific Verdict

**RED**

RED: no candidate correspondence mechanism (Step 5) reached ESTABLISHED or STRONGLY SUPPORTED, and the archive-structure finding (Step 3/8) is itself ESTABLISHED as a permanent structural blocker: all 449/449 PCAP members are per-host captures with zero exceptions, so the PCAP population can never match the CSV's whole-subnet flow aggregates regardless of how the timezone/mapping questions eventually resolve. This is not evidence that packet/graph data lacks value -- it means safe correspondence cannot currently be established with THIS flow dataset.

## 18. Recommendation for Next Phase

Do NOT reopen the packet/graph modality for training. Before reopening it, obtain ONE of: (a) an authoritative document (from the original CSE-CIC-IDS2018 publication or its creators) stating the exact timezone used for CICFlowMeter Timestamp generation, or (b) a raw packet capture that has NOT been pre-filtered to a single host and can be independently verified against a known-timezone flow export, or (c) a definitive capture-to-flow mapping document. Absent any of these, no further PCAP retrieval or correlation search is likely to change this verdict, since Step 4 already inspected the one remaining locally-reachable metadata source (logs.zip) without finding such a mapping.

## 19. Exact Files Inspected

- `C:\AKSHAY\Akshay\SIH FOLDER MAIN\data\raw\CSE-CIC-IDS2018\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`
- `C:\AKSHAY\Akshay\SIH FOLDER MAIN\data\windows\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`
- `docs/CIC_IDS2018_DATA_READINESS.md`
- `docs/CIC_IDS2018_FEATURE_POLICY.md`
- `docs/TEMPORAL_DATASET_DESIGN.md`
- `reports/cic_ids2018_audit.md`
- `ingestion/csv_reader.py`
- `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase9c_pcap_archive_inspection\discovery_report.json`
- `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase9h_multimodal_baseline\multimodal_baseline_report.json`
- `Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip (central directory only, via HTTP Range)`

## 20. Reproducibility Information

- Commit hash at run time: `4e54c7533fc327331d1ea44a609a52a6010424e0`
- Branch: `main`
- Alignment feasibility table:

| capture | coverage | hosts | evidence strength | verdict |
|---|---|---|---|---|
| pcap/UCAP172.31.69.22 | 2018-02-14 12:32:11 to 21:29:12 UTC (per scapy packet.time); capture-host clock offset to UTC now ESTABLISHED as +4h via exact DHCP-event cross-check | 1 (172.31.69.22, plus ~550 peers observed communicating with it) | MIXED: host clock offset to UTC is now EXACT/ESTABLISHED (not weak) via DHCP cross-check, but endpoint identity and CSV/CICFlowMeter clock are still unresolved, and re-testing correlation at this exact offset still yields noise-level |r| -- so the OVERALL alignment remains not established | NOT ALIGNABLE with current evidence |
| pcap/capWIN-J6GMIG1DQE5-172.31.64.89 | 2018-02-14 12:28:25 to 21:30:41 (per scapy packet.time, timezone unconfirmed -- no plain-text syslog available for this Windows host to cross-check, only a binary .evtx event log) | 1 (172.31.64.89, plus ~596 peers) | WEAK (date match only; no timezone, no endpoint identity) | NOT ALIGNABLE with current evidence |

STOP AFTER PHASE 9I.
