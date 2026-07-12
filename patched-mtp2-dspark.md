
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
| **Patched MTP-2** | **2362.2** | Context 0, Concurrency 64 |
| **Patched DSpark** | **1984.18** | Context 8192, Concurrency 64 |
| **Stock vLLM 0.25.0 + FlashInfer 0.6.14** | **1944.42** | Context 0, Concurrency 64 |

Overall ranking:

1. Patched MTP-2
2. Patched DSpark
3. Stock vLLM

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

Included:

- PR #48303
- PR #48304
- PR #48317
- FlashInfer 0.6.14
- NVRTC development headers

---

# MTP‑2 Runtime

Model: `deepseek-ai/DeepSeek-V4-Flash`

Served name: `d4f`

Port: `15004`

Speculative decoding:

```text
method: mtp
num_speculative_tokens: 2
```

Key runtime flags:

```text
--tensor-parallel-size 2
--kv-cache-dtype fp8
--gpu-memory-utilization 0.95
--max-model-len 1048576
--max-num-seqs 64
--max-num-batched-tokens 4096
--enable-prefix-caching
--enable-chunked-prefill
--enable-auto-tool-choice
--tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4
--async-scheduling
--moe-backend flashinfer_cutlass
```

---

# DSpark Image

Image

```text
vllm-ds4:v0.25.0-dspark-topk256
```

Digest

```text
sha256:e6176d085417967353c7e998ec2cc3b46b7edd87221e1aef44150a1076ecc9c1
```

---

# Additional DSpark Changes

In addition to the upstream vLLM pull requests, this image includes the DeepSeek‑V4 compatibility fix documented in **FlashInfer Issue #3828**.

Additional changes:

- FlashInfer SM120 sparse MLA Top‑K=256 support
- DSpark speculative decoding backend
- DeepSeek‑V4 compatibility fix (`draft_id_to_target_id`)
- Reused FlashInfer autotune cache
- Rebuilt against the patched vLLM runtime

Runtime:

```text
method: dspark
num_speculative_tokens: 5
draft_sample_method: greedy
```

Stability adjustment:

- Initial `--gpu-memory-utilization=0.95`
- Reduced to `0.93` after CUDA OOM during extended requests

---

# Benchmark Methodology

Tool: `llm-bench`

Contexts tested: 0, 8192, 16384, 32768, 65536, 131072, 262144

Concurrency: 1, 2, 4, 8, 16, 32, 64

---

# MTP‑2 Throughput (tok/s)

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|189.9|297.0|349.4|670.0|994.3|1542.1|2362.2|
|8192|194.3|300.8|332.5|607.3|899.9|1303.1|1947.6|
|16384|167.0|304.4|328.8|583.5|884.9|1345.3|1891.3|
|32768|182.7|273.3|322.1|598.6|833.1|1256.4|—|
|65536|184.1|263.5|321.4|569.7|805.8|—|—|
|131072|195.5|256.4|306.3|550.6|—|—|—|
|262144|182.5|261.5|308.0|—|—|—|—|

# MTP‑2 Speculative Metrics

| Metric | Value |
|---|---:|
| Speculative rounds | 422,517 |
| Draft tokens | 845,034 |
| Accepted tokens | 439,151 |
| Generated tokens | 862,227 |
| Acceptance | 51.97% |
| Accepted / round | 1.039 |

## Prefix Cache

| Metric | Value |
|---|---:|
| Queries | 20,848,155 |
| Hits | 19,462,656 |
| Hit Rate | 93.35% |

---

# DSpark Throughput

## Run 1

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|175.26|251.32|352.85|497.88|741.06|1113.40|1709.70|
|8192|215.71|280.53|374.78|630.47|950.89|1212.54|1984.18|

## Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|171.15|245.73|345.43|493.56|746.63|1127.43|1725.35|
|8192|203.23|258.92|403.26|538.64|744.62|1219.54|1923.63|

# DSpark Speculative Metrics

Short validation capture only:

- Speculative rounds: 1
- Draft tokens: 5
- Accepted tokens: 1
- Acceptance: 20%

These are not representative of full benchmark execution.

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

# Startup Characteristics

- Model size: 148.66 GiB
- Weight load: 16.47 s
- MTP load: 1.24 s
- CUDA graph capture: 32 s
- Engine initialization: 133.36 s
- KV cache: 1,993,100 tokens
- Max theoretical concurrency at 1M context: 1.90×

Runtime features confirmed:

- Async scheduling
- Prefix caching
- Chunked prefill
- FP8 KV cache
- DeepGEMM FP4 experts
- FlashInfer sparse MLA
- CUDA Graphs
- TileLang kernels

---

# GPU Snapshot

GPU1

- 435.58 W
- 61 °C
- 2737 MHz
- 96,658 MiB
- 99% utilization

GPU2

- 437.43 W
- 66 °C
- 2700 MHz
- 96,688 MiB
- 99% utilization

---

# Stability

## MTP‑2

- No CUDA OOM
- No NCCL failures
- No runtime crashes

## DSpark

- Stable after reducing GPU memory utilization from **0.95 → 0.93**

---

# Future Work

- Include exact `llm-bench` command
- Publish raw benchmark JSON
- Include complete benchmark logs
- Capture full DSpark speculative acceptance statistics
- Add MTP‑2 vs DSpark comparison charts
- Add power efficiency analysis
