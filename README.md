# DeepSeek-V4-Flash on stock vLLM v0.25.0 (SM120 / RTX PRO 6000 Blackwell)

**Fix for:** `TypeError: trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument 'swa_topk_lens'`

Running `deepseek-ai/DeepSeek-V4-Flash` (FP8, ~160 GB) on the stock `vllm/vllm-openai:v0.25.0` image, TP=2 across 2× RTX PRO 6000 Blackwell (SM120). Works with a one-line dependency bump — no fork, no source patches.

## Background

**vLLM v0.25.0** (released 2026-07-11) is the first stock release with a realistic shot at DSv4 on SM120 consumer/workstation Blackwell. The release notes bring the pieces that were missing in v0.24.0 — where stock could not run DSv4-FP8 on this hardware at all, forcing everyone onto patched forks:

- DeepGEMM updated to enable **SM120 support**
- DSv4-specific kernel work: a `token_to_req_indices` cache (5–6× kernel speedup) and an improved DSv4 MXFP8 kernel
- The DeepSeek V4 tool/reasoning parser ported to the new Streaming Parser Engine

**Model:** the official [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) checkpoint — native FP8 weights (`scale_fmt=ue8m0`), ~160 GB across 46 safetensors shards. At TP=2 that's ~75.6 GiB per GPU, leaving roughly 10 GiB per card for fp8 KV cache at `gpu_memory_utilization=0.95` on 96 GB cards. vLLM resolves the expert dtype to fp4 and runs the MoE through its `DEEPGEMM_MXFP4` backend; attention takes the default `DEEPSEEK_SPARSE_SWA` sparse-MLA path.

That default attention path is exactly where the release-day bug lives.

## The bug

vLLM v0.25.0 pins **flashinfer 0.6.13**, but its DeepSeek-V4 attention code calls flashinfer's sparse-MLA decode kernel with a new argument, `swa_topk_lens`, that only exists in **flashinfer 0.6.14**. Classic release-day skew: the vLLM code targets the *next* flashinfer release while the image ships the old pin.

Result: every DSv4 forward pass on the default `DEEPSEEK_SPARSE_SWA` attention backend dies during startup memory profiling:

```
File ".../vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py", line 769, in _forward_decode
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
TypeError: trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument 'swa_topk_lens'
```

As of this writing, the `nightly` image (`0.25.1.dev37`) has the **same** bug — the flashinfer pin hasn't been bumped there either.

## The fix

Upgrade flashinfer to **0.6.14** inside the container at startup, intentionally overriding vLLM's `flashinfer-python==0.6.13` pin. (pip will print a "dependency conflict" warning — that's expected and harmless; overriding the pin is the whole point.)

There is one wrinkle. flashinfer ships as companion packages that must be version-matched or the import hard-fails:

| Package | Role |
|---|---|
| `flashinfer-python` | Python code / kernel launchers |
| `flashinfer-cubin` | Precompiled kernel binaries (offline copy) |
| `flashinfer-jit-cache` | Prebuilt JIT cache |

At the time of the fix, PyPI had `flashinfer-python 0.6.14` but **not** `flashinfer-cubin 0.6.14`. Upgrading only the python package trips flashinfer's version check:

```
RuntimeError: flashinfer-cubin version (0.6.13) does not match flashinfer version (0.6.14).
```

**Resolution:** if a matched cubin package isn't available, *uninstall* the stale cubin/jit-cache packages entirely. The cubin package is just an offline copy — without it, flashinfer's loader downloads version-matched cubins at runtime (and JIT-compiles anything it can't fetch). Point the cache env vars at a persistent volume and this is a one-time cost that survives restarts.

The entrypoint in the compose below handles both cases automatically:

```bash
# Plan A: matched 0.6.14 pair (works once cubin 0.6.14 lands on PyPI)
# Plan B: remove stale cubin packages -> runtime download/JIT of matched kernels
if ! pip install --no-cache-dir "flashinfer-python==0.6.14" "flashinfer-cubin==0.6.14"; then
  pip install --no-cache-dir "flashinfer-python==0.6.14"
  pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true
fi
```

Do **not** use `FLASHINFER_DISABLE_VERSION_CHECK=1` instead — that runs 0.6.14 Python against 0.6.13 kernel binaries, which is exactly the kind of skew that produces silent wrong outputs rather than clean crashes.

## Result

Boots clean on the default `DEEPSEEK_SPARSE_SWA` backend. On 2× RTX PRO 6000 Blackwell (TP=2, PCIe, fp8 KV cache, MTP-2 speculative decoding):

- Weights: 75.6 GiB per GPU, ~18 s load from NVMe
- KV cache: ~10.3 GiB → ~2.0 M tokens at 1 M `max_model_len`
- Single-stream decode: **~168 tok/s** wall-clock (1500 tokens in ~8.9 s, including TTFT)

## Expiry

This workaround is temporary by design. Once vLLM bumps its flashinfer pin to ≥ 0.6.14 (a patch release or nightly), the entire `entrypoint`/`command` pip dance can be deleted and the stock image works as-is.

## Notes on the compose

- `--disable-custom-all-reduce` is unrelated to the flashinfer bug: it's required for PCIe-connected GPUs without peer-to-peer access (no NVLink), where vLLM's custom all-reduce kernel fails with `invalid argument`.
- `HF_HOME` must be pinned explicitly because `XDG_CACHE_HOME` (set for kernel caches) would otherwise redirect HuggingFace cache resolution away from the mounted model volume.
- `device_ids`, volume paths, port, and the API key are specific to this host — adjust for yours. **Change the `--api-key`.**
- `restart: "no"` is bring-up mode; switch to `unless-stopped` once stable.

## docker-compose.yml

```yaml
services:
  ds4-flash:
    # Stock vLLM v0.25.0 on SM120 + flashinfer 0.6.14 (fixes swa_topk_lens bug).
    # cubin 0.6.14 may not be on PyPI; fallback removes the stale 0.6.13 cubin
    # packages so flashinfer downloads/JITs matching kernels at runtime instead.
    image: vllm/vllm-openai:v0.25.0
    container_name: ds4-flash-stock
    restart: "no"   # bring-up mode; restore unless-stopped once stable

    network_mode: host
    ipc: host
    shm_size: "64g"
    gpus:
      # host GPU 0 is a sidecar card — excluded. Serving GPUs at 1 and 2.
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
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NCCL_IB_DISABLE=1
      - NCCL_P2P_LEVEL=SYS
      - SAFETENSORS_FAST_GPU=1
      # HF cache pinned — XDG_CACHE_HOME below would otherwise hijack it
      - HF_HOME=/root/.cache/huggingface
      - HF_HUB_OFFLINE=1
      # persistent kernel caches. Runtime-downloaded flashinfer cubins land
      # under XDG cache too, so they persist across restarts.
      - XDG_CACHE_HOME=/cache
      - VLLM_CACHE_ROOT=/cache/vllm
      - TRITON_CACHE_DIR=/cache/triton
      - TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
      - FLASHINFER_WORKSPACE_BASE=/cache/flashinfer

    volumes:
      - /m2-2/vllm/root:/root/.cache/huggingface
      - /m2-2/vllm/dsv4flash-stock-cache:/cache

    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        set -euo pipefail
        # Plan A: matched 0.6.14 pair (works if cubin 0.6.14 has landed on PyPI).
        # Plan B: cubin 0.6.14 unavailable -> remove stale cubin/jit-cache packages
        # so the loader fetches/JITs version-matched kernels at runtime.
        if ! pip install --no-cache-dir "flashinfer-python==0.6.14" "flashinfer-cubin==0.6.14"; then
          echo ">>> flashinfer-cubin 0.6.14 not available; removing stale cubin packages"
          pip install --no-cache-dir "flashinfer-python==0.6.14"
          pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true
        fi
        python3 -c "import flashinfer; print('>>> flashinfer version:', flashinfer.__version__)"
        exec vllm serve deepseek-ai/DeepSeek-V4-Flash \
          --served-model-name d4f \
          --api-key CHANGE_ME \
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
          --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
          --async-scheduling
```

## Benchmark results

Aggregate decode throughput (tok/s) by prompt depth and concurrency, measured with `vllm bench serve` (random dataset) on the compose above — 2× RTX PRO 6000 Blackwell, TP=2, fp8 KV cache, MTP-2. Missing cells: prompt depth × concurrency exceeds the KV cache pool at 1 M `max_model_len`.

| Prompt depth \ Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 167.4 | 232.1 | 229.7 | 430.5 | 616.3 | 903.7 | 1383.1 |
| 8k | 181.1 | 236.4 | 219.4 | 391.9 | 502.8 | 738.9 | 1071.1 |
| 16k | 171.9 | 220.2 | 215.6 | 365.9 | 532.7 | 718.8 | 1119.9 |
| 32k | 166.0 | 226.8 | 230.6 | 384.0 | 471.2 | 721.9 | — |
| 64k | 153.6 | 210.4 | 211.9 | 367.5 | 506.2 | — | — |
| 128k | 154.8 | 206.9 | 195.9 | 356.2 | — | — | — |
| 256k | 162.1 | 217.0 | 191.8 | — | — | — | — |

Notable: single-stream decode holds remarkably flat with depth (~167 tok/s shallow → ~155–162 tok/s at 128–256k), and throughput scales to ~1.4k tok/s aggregate at 64 concurrent shallow streams.
