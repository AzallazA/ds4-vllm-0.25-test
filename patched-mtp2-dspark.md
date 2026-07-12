
# DeepSeek-V4-Flash Patched vLLM Benchmark

Comprehensive benchmark report for running **DeepSeek-V4-Flash** on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs using custom patched vLLM images featuring:

- MTP-2 speculative decoding
- DSpark speculative decoding
- FlashInfer CUTLASS MoE support
- DeepSeek-V4 upstream fixes
- Blackwell (SM120) optimizations

---

# Hardware

## Host

- Ubuntu 26.04 LTS
- AMD Threadripper Pro 7965WX
- 24 CPU cores
- ~256 GB RAM

## GPUs

- 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Tensor Parallel = 2
- PCIe Gen5 x16
- No NVLink
- 450 W power limit per GPU

| GPU | Device |
|---|---|
| GPU0 | RTX 5090 (unused) |
| GPU1 | RTX PRO 6000 |
| GPU2 | RTX PRO 6000 |

---

# Purpose

This document records the software revisions, runtime configuration, patches, startup behavior, benchmark methodology, and measured performance for DeepSeek‑V4‑Flash using custom patched vLLM images.

---

# Benchmark Summary

| Configuration | Best Throughput (tok/s) | Best Scenario |
|---|---:|---|
| **Patched MTP-2** | **2345.75** | Context 0, Concurrency 64 (Run 1) |
| **Stock vLLM 0.25.0 + FlashInfer 0.6.14** | **1944.42** | Context 0, Concurrency 64 |
| **Patched DSpark** | **1926.71** | Context 8192, Concurrency 64 (Run 2) |

Overall ranking:

1. Patched MTP-2
2. Stock vLLM
3. Patched DSpark

---

# Upstream Changes

## vLLM

- PR #48303 — DeepSeek MXFP4 FlashInfer CUTLASS MoE support
- PR #48304 — DeepSeek‑V4 MTP compression / RoPE fix
- PR #48317 — Hybrid KV-cache capacity reporting fix

## FlashInfer

- SM120 sparse MLA Top-K=256 support

## DSpark

- FlashInfer Issue #3828 — DeepSeek‑V4 DSpark compatibility (`draft_id_to_target_id` fix)

---

# MTP‑2 Throughput

## Run 1

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|192.30|297.81|348.72|675.27|989.14|1544.58|2345.75|
|8192|182.73|289.13|344.90|601.61|900.06|1336.21|1962.45|
|16384|207.98|278.82|327.23|587.52|884.63|1313.04|1984.25|
|32768|178.18|285.52|343.92|572.50|807.23|1262.84|—|
|65536|189.06|266.19|313.91|562.34|809.44|—|—|
|131072|166.53|269.53|315.45|540.24|—|—|—|
|262144|187.52|253.10|294.06|—|—|—|—|

## Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|187.92|301.77|346.77|663.83|995.57|1566.90|2338.16|
|8192|180.14|286.08|329.93|582.99|903.35|1343.86|1921.61|
|16384|190.77|293.66|333.93|600.78|863.05|1292.97|1949.82|
|32768|200.84|302.94|325.57|611.28|886.73|1288.42|—|
|65536|184.33|280.99|339.27|575.73|852.64|—|—|
|131072|169.37|269.02|299.82|542.74|—|—|—|
|262144|181.05|271.49|279.31|—|—|—|—|

# MTP‑2 Speculative Metrics

> Replace this section with the newest acceptance statistics extracted from the latest MTP logs if desired. Benchmark throughput above reflects the latest runs.

---

# DSpark Throughput

## Run 1

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|175.00|252.09|365.14|492.50|729.24|1124.33|1726.52|
|8192|218.07|354.69|456.42|530.06|849.19|1195.57|1919.39|
|16384|201.30|269.24|428.51|508.37|809.93|1265.02|—|
|32768|225.50|260.04|446.88|566.16|709.07|1295.52|—|
|65536|194.69|320.31|393.15|516.97|753.60|—|—|
|131072|177.51|259.43|367.11|528.09|—|—|—|
|262144|194.54|282.76|351.96|—|—|—|—|

## Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|177.01|258.48|344.16|503.96|768.82|1129.40|1711.55|
|8192|223.85|294.46|427.49|538.11|883.81|1348.99|1926.71|
|16384|242.70|255.37|387.16|534.97|787.38|1221.22|—|
|32768|219.64|254.33|398.33|480.90|845.08|1142.76|—|
|65536|211.21|301.53|398.25|487.44|803.95|—|—|
|131072|211.15|276.31|371.59|522.83|—|—|—|
|262144|168.19|313.36|373.76|—|—|—|—|

# DSpark Speculative Metrics

| Metric | Typical Value / Range |
|---|---:|
| Mean acceptance length | 2.2–5.5 tokens |
| Typical draft acceptance | 40–60% |
| Peak draft acceptance | 89.2% |
| Peak mean acceptance length | 5.46 tokens |

---

# Stock vLLM 0.25.0 + FlashInfer 0.6.14

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|182.51|271.99|296.14|575.25|828.93|1288.15|1944.42|
|8192|186.75|275.28|316.51|507.52|743.74|1063.34|1640.74|
|16384|188.54|257.80|280.07|521.11|673.96|1099.87|1593.25|
|32768|183.20|242.14|274.72|497.77|676.83|1055.78|—|
|65536|186.14|239.71|284.75|495.33|701.67|—|—|
|131072|158.76|239.85|271.38|460.93|—|—|—|
|262144|169.48|246.76|256.44|—|—|—|—|

---

The remaining sections (Startup Characteristics, GPU Snapshot, Stability, and Future Work) are unchanged from your previous README.
