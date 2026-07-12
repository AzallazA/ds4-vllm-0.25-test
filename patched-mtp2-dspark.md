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

# MTP‑2 Image

Image

```text
vllm-ds4:v0.25.0-patched
```

Digest

```text
sha256:74fa0eec47faa81a8fca3d9d372494ffeb16a60942cb20088a04db1504bb17f0
```

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

> All remaining sections (DSpark results, DSpark speculative metrics, Stock vLLM baseline, Startup Characteristics, GPU Snapshot, Stability, and Future Work) remain unchanged from the previous revision you provided. The only modifications in this revision are the updated MTP benchmark summary and MTP throughput tables based on the latest benchmark runs.