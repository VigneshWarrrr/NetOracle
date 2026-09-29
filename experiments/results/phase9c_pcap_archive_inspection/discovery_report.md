# Phase 9C: Remote ZIP Central-Directory Inspection

Object: `Original Network Traffic and Log data/Wednesday-14-02-2018/pcap.zip`
Success: **True**

## 1. CONFIRMED FROM REMOTE ZIP METADATA

- HTTP access: confirmed (HEAD status 200)
- Content-Length: 39,913,353,098 bytes
- Accept-Ranges: bytes
- Content-Type: application/zip
- ETag: `845cbc33e555f5906ea5c9bc520113ab-2380`
- EOCD found at file offset 39,913,353,076
- ZIP64 required: True
- Resolved central directory: offset=39,913,295,876, size=57,124 bytes, entries=450
- Central-directory entries actually parsed: 450
- Consistency check: sum(member compressed sizes) + overhead = Content-Length (difference 85,954 bytes, matches expected header/CD/EOCD overhead)

## 2. NOT VERIFIED

- Whether any member's content is actually valid PCAP data (no packet was opened or parsed)
- Whether the archive's contents correspond, at the flow or timestamp level, to the existing processed CSV for this day
- Whether the per-endpoint capture files cover the full day or only part of it

## 3. CANDIDATE PCAP MEMBERS

Total non-folder members: 449. Classification counts: {'unknown_extension': 449}

None of the 449 members carry a `.pcap`/`.pcapng` extension. All are classified `unknown_extension` and treated as candidates by CONTEXT ONLY: parent archive is named `pcap.zip`, containing folder is `pcap/`, and filenames are prefixed `cap`/`UCAP` followed by a hostname and/or IP address.

10 smallest candidate members by compressed size:

| Filename | Compressed | Uncompressed | Method | ZIP64 extra used |
|---|---:|---:|---:|---|
| `pcap/UCAP172.31.69.22` | 551,024 | 1,282,048 | 8 | True |
| `pcap/UCAP172.31.69.18` | 635,141 | 1,376,256 | 8 | True |
| `pcap/UCAP172.31.69.7` | 3,970,784 | 5,136,384 | 8 | True |
| `pcap/UCAP172.31.69.27` | 4,167,826 | 5,570,560 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.65.99` | 5,109,104 | 9,593,169 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.64.89` | 6,139,533 | 10,894,363 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.65.58` | 6,361,342 | 11,566,548 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.67.26` | 6,386,111 | 11,547,973 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.64.24` | 6,514,289 | 11,601,215 | 8 | True |
| `pcap/capWIN-J6GMIG1DQE5-172.31.64.39` | 6,571,402 | 11,975,326 | 8 | True |

## 4. SMALLEST PCAP MEMBER

`pcap/UCAP172.31.69.22` -- compressed 551,024 bytes (538.1 KiB), uncompressed 1,282,048 bytes, compression method 8 (8 = DEFLATE), local header offset 39,485,257,643.

## 5. TOTAL BYTES DOWNLOADED DURING INSPECTION

**122,716 bytes** (~119.8 KiB) against a 10,485,760-byte budget (10,363,044 bytes remaining, unused).

Request log:

| Type | Purpose | Bytes | Status |
|---|---|---:|---|
| HEAD | - | 0 | 200 |
| GET_RANGE | tail_64kib | 65,536 | 206 |
| GET_RANGE | zip64_eocd_record | 56 | 206 |
| GET_RANGE | central_directory | 57,124 | 206 |

## 7. Naming comparison with processed CSV convention (metadata only, no content claim)

- Processed CSV convention: Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv (flow-level, one file for the whole day)
- Raw archive member convention: pcap/<capture-prefix><hostname>-<ip-address> (no .pcap/.pcapng extension; one file per captured endpoint)
- The two naming conventions are structurally different (one whole-day flow-level CSV vs. many per-endpoint raw capture files) and share no common filename token beyond both being associated with the 'Wednesday-14-02-2018' day folder/filename. This is a naming-convention comparison only -- no row-level, flow-level, or timestamp-level correspondence is claimed or verified; no packet was opened to check this.

