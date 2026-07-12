# DeepSeek-V4-Flash Patched vLLM Benchmarks

This document contains the complete benchmark results and runtime
configuration used while evaluating custom patched vLLM builds for
DeepSeek-V4-Flash on dual NVIDIA RTX PRO 6000 Blackwell GPUs.

The work focused on evaluating two speculative decoding implementations:

- MTP-2
- DSpark

Both images include DeepSeek-V4 specific optimizations together with
FlashInfer CUTLASS MXFP4 MoE support.

---

# Test Hardware

## Host

| Component | Value |
|----------|-------|
| Operating System | Ubuntu 26.04 LTS |
| CPU | AMD Threadripper Pro 7965WX |
| CPU Cores | 24 |
| Memory | ~256 GB DDR5 |

## GPUs

| Component | Value |
|----------|-------|
| GPU Count | 2 |
| GPU Model | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| VRAM | 96 GB each |
| Tensor Parallel | 2 |
| Interconnect | PCIe Gen5 (No NVLink) |
| Power Limit | 450 W |

GPU assignment:

```
GPU0 : RTX 5090 (display GPU)
GPU1 : RTX PRO 6000
GPU2 : RTX PRO 6000
```

---

# Software Versions

| Component | Version |
|----------|---------|
| CUDA | 13 |
| PyTorch | 2.11.0 |
| vLLM | 0.25.0 |
| FlashInfer | 0.6.14 |
| KV Cache | FP8 |

---

# Images

## MTP-2

Image

```
vllm-ds4:v0.25.0-patched
```

Digest

```
sha256:74fa0eec47faa81a8fca3d9d372494ffeb16a60942cb20088a04db1504bb17f0
```

---

## DSpark

Image

```
vllm-ds4:v0.25.0-dspark-topk256
```

Digest

```
sha256:e6176d085417967353c7e998ec2cc3b46b7edd87221e1aef44150a1076ecc9c1
```

---

# Included Upstream Pull Requests

Both images include:

- PR #48303 — FlashInfer CUTLASS MXFP4 MoE backend
- PR #48304 — DeepSeek-V4 MTP compression / RoPE fix
- PR #48317 — Hybrid KV-cache reporting fix

Additional packages:

- FlashInfer 0.6.14
- CUDA Tile 1.5
- NVRTC development headers

---

# Additional DSpark Changes

The DSpark image required additional work beyond the upstream pull requests.

Additional modifications include:

- FlashInfer SM120 sparse MLA top-k=256 support
- DSpark speculative decoding
- DeepSeek-V4 compatibility fix required by DSpark
- Reuse of compiled FlashInfer autotune cache

These changes are specific to the DSpark image.

---

# Patch Comparison

| Feature | MTP-2 | DSpark |
|----------|------:|-------:|
| FlashInfer CUTLASS MoE | ✓ | ✓ |
| DeepSeek RoPE Fix | ✓ | ✓ |
| Hybrid KV Reporting | ✓ | ✓ |
| MTP Speculative Decoding | ✓ | — |
| DSpark Speculative Decoding | — | ✓ |
| Sparse MLA Top-k=256 | — | ✓ |
| DeepSeek DSpark Compatibility Fix | — | ✓ |

---

# Runtime Configuration

Model

```
deepseek-ai/DeepSeek-V4-Flash
```

Served name

```
d4f
```

Tensor parallel

```
2
```

Maximum sequence length

```
1,048,576
```

KV Cache

```
FP8
```

GPU memory utilization

MTP-2

```
0.95
```

DSpark

```
0.93
```

---

# Important Runtime Flags

```
--tensor-parallel-size 2

--kv-cache-dtype fp8

--gpu-memory-utilization

--max-model-len 1048576

--max-num-seqs 64

--max-num-batched-tokens 4096

--enable-prefix-caching

--enable-chunked-prefill

--async-scheduling

--reasoning-parser deepseek_v4
```

MTP additionally uses

```
method=mtp

num_speculative_tokens=2
```

DSpark additionally uses

```
method=dspark

num_speculative_tokens=5

draft_sample_method=greedy
```

---

# Runtime Notes

Confirmed during startup:

- Asynchronous scheduling enabled
- Prefix caching enabled
- Chunked prefill enabled
- FP8 KV cache
- Breakable CUDA Graphs enabled
- DeepGEMM FP4 experts
- FlashInfer Sparse MLA backend
- TileLang kernels compiled successfully

---

# Model Loading

Checkpoint size

```
148.66 GiB
```

Model memory

```
75.64 GiB per GPU
```

Weight loading

```
16.47 seconds
```

MTP draft model loading

```
1.24 seconds
```

Total engine initialization

```
133.36 seconds
```

---

# CUDA Graphs

Graph capture time

```
32 seconds
```

CUDA graph pool

```
1.73 GiB
```

Estimated graph memory

```
1.58 GiB
```

Available KV cache

```
10.29 GiB
```

GPU KV cache

```
1,993,100 tokens
```

Maximum full-context concurrency

```
1.90x
```

---

# MTP-2 Benchmarks

## Throughput (tokens/sec)

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|--------:|----:|----:|----:|----:|----:|----:|----:|
| 0 | 189.9 | 297.0 | 349.4 | 670.0 | 994.3 | 1542.1 | 2362.2 |
| 8K | 194.3 | 300.8 | 332.5 | 607.3 | 899.9 | 1303.1 | 1947.6 |
| 16K | 167.0 | 304.4 | 328.8 | 583.5 | 884.9 | 1345.3 | 1891.3 |
| 32K | 182.7 | 273.3 | 322.1 | 598.6 | 833.1 | 1256.4 | — |
| 64K | 184.1 | 263.5 | 321.4 | 569.7 | 805.8 | — | — |
| 128K | 195.5 | 256.4 | 306.3 | 550.6 | — | — | — |
| 256K | 182.5 | 261.5 | 308.0 | — | — | — | — |

---

# MTP-2 Speculative Statistics

Captured from full benchmark metrics.

| Metric | Value |
|---------|------:|
| Speculative rounds | 422,517 |
| Draft tokens | 845,034 |
| Accepted tokens | 439,151 |
| Generated tokens | 862,227 |
| Acceptance rate | **51.97%** |
| Accepted tokens / round | **1.039** |

---

# Prefix Cache

| Metric | Value |
|---------|------:|
| Queries | 20,848,155 |
| Hits | 19,462,656 |
| Hit Rate | **93.35%** |

---

# DSpark Stability

Initial GPU memory utilization

```
0.95
```

Result

- Stable for startup
- CUDA OOM during larger benchmark runs

Final configuration

```
--gpu-memory-utilization 0.93
```

This resolved the OOM issue while maintaining nearly identical throughput.

---

# DSpark Benchmarks

## Run 1

### Context 0

|Concurrency|tok/s|
|---:|---:|
|1|175.26|
|2|251.32|
|4|352.85|
|8|497.88|
|16|741.06|
|32|1113.40|
|64|1709.70|

### Context 8192

|Concurrency|tok/s|
|---:|---:|
|1|215.71|
|2|280.53|
|4|374.78|
|8|630.47|
|16|950.89|
|32|1212.54|
|64|1984.18|

---

## Run 2

### Context 0

|Concurrency|tok/s|
|---:|---:|
|1|171.15|
|2|245.73|
|4|345.43|
|8|493.56|
|16|746.63|
|32|1127.43|
|64|1725.35|

### Context 8192

|Concurrency|tok/s|
|---:|---:|
|1|203.23|
|2|258.92|
|4|403.26|
|8|538.64|
|16|744.62|
|32|1219.54|
|64|1923.63|

---

# DSpark Speculative Metrics

A full production metrics capture was not available.

Validation capture confirmed speculative decoding functionality.

| Metric | Value |
|---------|------:|
| Speculative rounds | 1 |
| Draft tokens | 5 |
| Accepted tokens | 1 |
| Acceptance | 20% |

---

# GPU Snapshot

Power limit

```
450 W
```

GPU1

- 435.58 W
- 61°C
- 2737 MHz
- 96.7 GB VRAM
- 99% utilization

GPU2

- 437.43 W
- 66°C
- 2700 MHz
- 96.7 GB VRAM
- 99% utilization

---

# Observations

MTP-2

- Stable throughout benchmarking
- Approximately 52% draft acceptance
- Excellent prefix cache utilization
- Successfully operated at 95% GPU memory utilization

DSpark

- Required additional DeepSeek compatibility changes
- Required reduced GPU memory utilization (93%)
- Stable after tuning
- Competitive throughput at higher concurrency

---

# Remaining Work

Future updates may include:

- Complete DSpark speculative statistics
- Full benchmark JSON artifacts
- Exact llm-bench command lines
- Additional long-context benchmark runs
- MTP-2 versus DSpark comparison plots