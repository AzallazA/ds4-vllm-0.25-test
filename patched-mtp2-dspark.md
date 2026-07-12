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
|---|---:|---:|
| **Patched MTP-2 (Run 1)** | **2345.7** | Context 0, Concurrency 64 |
| **Patched MTP-2 (Run 2)** | **2338.2** | Context 0, Concurrency 64 |
| **Patched DSpark (Run 1)** | **1919.4** | Context 8192, Concurrency 64 |
| **Patched DSpark (Run 2)** | **1926.7** | Context 8192, Concurrency 64 |
| **Stock vLLM 0.25.0 + FlashInfer 0.6.14** | **1944.42** | Context 0, Concurrency 64 |

Overall ranking:

1. Patched MTP-2 (23% ahead of stock)
2. Patched DSpark (≈1% behind stock at peak, but competitive across contexts)
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

<details>
<summary>Docker Compose</summary>

```yaml
services:
  ds4-flash:
    # Custom image based on vLLM v0.25.0 with:
    # - PR #48303: DeepSeek MXFP4 FlashInfer CUTLASS MoE support
    # - PR #48304: DeepSeek V4 MTP compression/RoPE fix
    # - PR #48317: hybrid KV-cache capacity reporting fix
    # - flashinfer-python 0.6.14
    image: vllm-ds4:v0.25.0-patched
    container_name: ds4-flash-patched

    restart: "no"

    network_mode: host
    ipc: host
    shm_size: "64g"

    gpus:
      - driver: nvidia
        device_ids: ["1", "2"]
        capabilities: [gpu]

    runtime: nvidia
    init: true

    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864

    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      SAFETENSORS_FAST_GPU: "1"
      HF_HOME: /root/.cache/huggingface
      HF_HUB_OFFLINE: "1"
      XDG_CACHE_HOME: /cache
      VLLM_CACHE_ROOT: /cache/vllm
      TRITON_CACHE_DIR: /cache/triton
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/flashinfer

    volumes:
      - /m2-2/vllm/root:/root/.cache/huggingface
      - /m2-2/vllm/dsv4flash-patched-cache:/cache

    entrypoint: ["/bin/bash", "-lc"]

    command:
      - |
        set -euo pipefail
        echo ">>> Applied image patches:"
        cat /VLLM_PATCHES
        python3 - <<'PY'
        import flashinfer, vllm
        print(">>> vLLM version:", vllm.__version__)
        print(">>> FlashInfer version:", flashinfer.__version__)
        PY
        exec vllm serve deepseek-ai/DeepSeek-V4-Flash \
          --served-model-name d4f \
          --api-key aabduh \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port 15004 \
          --tensor-parallel-size 2 \
          --disable-custom-all-reduce \
          --kv-cache-dtype fp8 \
          --gpu-memory-utilization 0.95 \
          --max-model-len 1048576 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 4096 \
          --enable-prefix-caching \
          --enable-chunked-prefill \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --moe-backend flashinfer_cutlass \
          --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
          --async-scheduling
```

</details>

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

<details>
<summary>Docker Compose</summary>

```yaml
services:
  ds4-dspark:
    # Existing vLLM 0.25.0 image plus patched FlashInfer SM120
    # topk=256 decode/prefill support.
    image: vllm-ds4:v0.25.0-dspark-topk256
    container_name: ds4-dspark-patched

    restart: "no"

    network_mode: host
    ipc: host
    shm_size: "64g"

    gpus:
      - driver: nvidia
        device_ids: ["1", "2"]
        capabilities: [gpu]

    runtime: nvidia
    init: true

    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864

    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      SAFETENSORS_FAST_GPU: "1"
      HF_HOME: /root/.cache/huggingface
      HF_HUB_OFFLINE: "1"
      XDG_CACHE_HOME: /cache
      VLLM_CACHE_ROOT: /cache/vllm
      TRITON_CACHE_DIR: /cache/triton
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/flashinfer
      MAX_JOBS: "1"
      NVCC_THREADS: "1"

    volumes:
      - /m2-2/vllm/root:/root/.cache/huggingface
      - /m2-2/vllm/dsv4flash-patched-cache:/cache

    entrypoint: ["/bin/bash", "-lc"]

    command:
      - |
        set -euo pipefail
        echo ">>> vLLM image patches:"
        cat /VLLM_PATCHES
        python3 - <<'PY'
        import flashinfer, vllm
        from pathlib import Path
        print(">>> vLLM version:", vllm.__version__)
        print(">>> FlashInfer version:", flashinfer.__version__)
        print(">>> FlashInfer location:", flashinfer.__file__)
        root = Path(flashinfer.__file__).resolve().parent
        dispatch_file = root / "mla" / "_sparse_mla_sm120.py"
        if not dispatch_file.exists():
            raise RuntimeError(
                f"FlashInfer sparse-MLA dispatch file not found: {dispatch_file}"
            )
        source = dispatch_file.read_text()
        if "(32, 256)" not in source:
            raise RuntimeError(
                "Patched FlashInfer does not contain the SM120 "
                "DSv4 topk=256 dispatch."
            )
        print(">>> Confirmed SM120 DSv4 (32, 256) dispatch")
        PY
        exec vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
          --served-model-name d4f-dspark \
          --api-key aabduh \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port 15004 \
          --tensor-parallel-size 2 \
          --disable-custom-all-reduce \
          --kv-cache-dtype fp8 \
          --gpu-memory-utilization 0.93 \
          --max-model-len 1048576 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 4096 \
          --enable-prefix-caching \
          --enable-chunked-prefill \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --moe-backend flashinfer_cutlass \
          --no-enable-flashinfer-autotune \
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
          --async-scheduling
```

</details>

---

# Benchmark Methodology

Tool: `llm-bench`

Contexts tested: 0, 8192, 16384, 32768, 65536, 131072, 262144

Concurrency: 1, 2, 4, 8, 16, 32, 64

---

# MTP‑2 Throughput (tok/s)

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|192.3|297.8|348.7|675.3|989.1|1544.6|2345.7|
|8192|182.7|289.1|344.9|601.6|900.1|1336.2|1962.4|
|16384|208.0|278.8|327.2|587.5|884.6|1313.0|1984.3|
|32768|178.2|285.5|343.9|572.5|807.2|1262.8|—|
|65536|189.1|266.2|313.9|562.3|809.4|—|—|
|131072|166.5|269.5|315.5|540.2|—|—|—|
|262144|187.5|253.1|294.1|—|—|—|—|

# MTP‑2 Throughput — Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|187.9|301.8|346.8|663.8|995.6|1566.9|2338.2|
|8192|180.1|286.1|329.9|583.0|903.3|1343.9|1921.6|
|16384|190.8|293.7|333.9|600.8|863.1|1293.0|1949.8|
|32768|200.8|302.9|325.6|611.3|886.7|1288.4|—|
|65536|184.3|281.0|339.3|575.7|852.6|—|—|
|131072|169.4|269.0|299.8|542.7|—|—|—|
|262144|181.1|271.5|279.3|—|—|—|—|

# MTP‑2 Speculative Metrics

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Best accept rate (C64 ctx0) | 52.3% | 60.3% |
| Best throughput (tok/s) | 2345.7 | 2338.2 |

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
|0|175.0|252.1|365.1|492.5|729.2|1124.3|1726.5|
|8192|218.1|354.7|456.4|530.1|849.2|1195.6|1919.4|
|16384|201.3|269.2|428.5|508.4|809.9|1265.0|—|
|32768|225.5|260.0|446.9|566.2|709.1|1295.5|—|
|65536|194.7|320.3|393.2|517.0|753.6|—|—|
|131072|177.5|259.4|367.1|528.1|—|—|—|
|262144|194.5|282.8|352.0|—|—|—|—|

## Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|177.0|258.5|344.2|504.0|768.8|1129.4|1711.5|
|8192|223.9|294.5|427.5|538.1|883.8|1349.0|1926.7|
|16384|242.7|255.4|387.2|535.0|787.4|1221.2|—|
|32768|219.6|254.3|398.3|480.9|845.1|1142.8|—|
|65536|211.2|301.5|398.2|487.4|803.9|—|—|
|131072|211.1|276.3|371.6|522.8|—|—|—|
|262144|168.2|313.4|373.8|—|—|—|—|

# DSpark Speculative Metrics

DSpark accept rates vary significantly by context (range: 19%–98%):

| Context | Run 1 Accept | Run 2 Accept |
|:---|---:|---:|
|0|34.8%|27.9%|
|8192|71.4%|53.3%|
|16384|18.7%|94.2%|
|32768|77.8%|66.8%|
|65536|34.1%|31.7%|
|131072|98.1%|23.4%|
|262144|74.0%|68.6%|

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

# Startup Characteristics — MTP‑2

- Model size: 148.66 GiB
- Weight load: 33.58 s
- MTP draft load: 1.79 s
- TileLang compilation: ~28 s
- CUDA graph capture (PIECEWISE + FULL): 31 s
- Engine init to ready: ~127 s (14:58:55 → 15:01:08)
- KV cache: 1,868,081 tokens
- Max theoretical concurrency at 1M context: 1.78×

# Startup Characteristics — DSpark

- Model size: 155.43 GiB (includes draft module)
- Weight load: 35.19 s
- DSpark draft load: 3.22 s
- TileLang compilation: ~40 s
- CUDA graph capture (PIECEWISE + FULL + speculator): 58 s, 2.85 GiB
- Engine init to ready: ~164 s (12:56:35 → 12:59:19)
- KV cache: 1,137,739 tokens
- Max theoretical concurrency at 1M context: 1.09×

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

# GPU Power & Temperature (MTP‑2 Run 1)

| Metric | Average | Maximum |
|:---|---:|---:|
| Total power (W) | 843.74 | 931.52 |
| GPU temp (°C) | 65.6 | 92.0 |
| GPU util (%) | 62.05 | 100.0 |
| VRAM used (MB) | 192,127 | 192,244 |
| PCIe RX (MB/s) | 5,613 | 23,470 |
| PCIe TX (MB/s) | 4,569 | 24,017 |

---

# Stability

## MTP‑2

- No CUDA OOM
- No NCCL failures
- No runtime crashes
- 2 successful benchmark runs (16:06 and 16:39)

## DSpark

- Stable after reducing GPU memory utilization from **0.95 → 0.93**
- 2 successful benchmark runs (14:53 and 15:22)
- Both runs completed across all 7 context lengths (0–262k)

## Runtime Features Confirmed

- Async scheduling
- Prefix caching (93.35% hit rate)
- Chunked prefill
- FP8 KV cache (fp8_ds_mla format)
- DeepGEMM FP4 experts + E8M0/PDL
- FlashInfer CUTLASS MXFP4 MoE (contiguous w13 layout)
- FlashInfer sparse MLA SM120 Top‑K=256 autotune cache loaded
- CUDA Graphs (PIECEWISE + FULL, breakable)
- TileLang kernels (mhc_pre_big_fuse_with_norm, mhc_post, hc_head_fuse, mhc_fused)
- V1 engine with asynchronous scheduling

---

# Future Work

- Add MTP‑2 vs DSpark comparison charts
- Add power efficiency analysis (tokens per kWh)
- Explore DSpark at higher concurrency for deeper context lengths
- Investigate MTP‑2 acceptance rate at high concurrency