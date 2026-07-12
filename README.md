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

## Full container startup log

Complete log from a clean `docker compose up` of the compose above, for diffing against your own boot. Key lines to look for: the `>>> flashinfer version: 0.6.14` print (the fix is active), `DEEPSEEK_SPARSE_SWA backend` (default attention path, previously the crash site), `Using ['PYNCCL']` (custom all-reduce disabled), and `GPU KV cache size` at the end of init.

<details>
<summary>Click to expand full startup log</summary>

```
aabduh@threadripper-aabduh:~$ sudo docker logs ds4-flash-stock
Collecting flashinfer-python==0.6.14
  Downloading flashinfer_python-0.6.14-py3-none-any.whl.metadata (11 kB)
ERROR: Could not find a version that satisfies the requirement flashinfer-cubin==0.6.14 (from versions: 0.4.0, 0.4.1, 0.5.0, 0.5.1, 0.5.2, 0.5.3, 0.6.0, 0.6.1, 0.6.2, 0.6.3, 0.6.4, 0.6.5, 0.6.6, 0.6.7, 0.6.7.post1, 0.6.7.post2, 0.6.7.post3, 0.6.8rc1, 0.6.8, 0.6.8.post1, 0.6.9rc1, 0.6.9, 0.6.10rc1, 0.6.10, 0.6.10.post1, 0.6.11rc1, 0.6.11, 0.6.11.post1, 0.6.11.post2, 0.6.11.post3, 0.6.12rc1, 0.6.12rc2, 0.6.12rc3, 0.6.12, 0.6.13rc1, 0.6.13rc2, 0.6.13)
ERROR: No matching distribution found for flashinfer-cubin==0.6.14
>>> flashinfer-cubin 0.6.14 not available; removing stale cubin packages
Collecting flashinfer-python==0.6.14
  Downloading flashinfer_python-0.6.14-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: apache-tvm-ffi!=0.1.8,!=0.1.8.post0,<0.2,>=0.1.6 in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (0.1.9)
Requirement already satisfied: click in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (8.4.2)
Collecting cuda-tile>=1.4.0 (from flashinfer-python==0.6.14)
  Downloading cuda_tile-1.5.0-cp312-cp312-manylinux2014_x86_64.whl.metadata (7.4 kB)
Requirement already satisfied: einops in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (0.8.2)
Requirement already satisfied: ninja in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (1.13.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (2.2.6)
Requirement already satisfied: nvidia-cudnn-frontend>=1.13.0 in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (1.26.0)
Requirement already satisfied: nvidia-cutlass-dsl>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (4.5.2)
Requirement already satisfied: nvidia-ml-py in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (13.610.43)
Requirement already satisfied: packaging>=24.2 in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (26.2)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (2.34.2)
Requirement already satisfied: tabulate in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (0.10.0)
Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (2.11.0+cu130)
Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from flashinfer-python==0.6.14) (4.68.4)
Requirement already satisfied: typing-extensions>=4.5 in /usr/local/lib/python3.12/dist-packages (from apache-tvm-ffi!=0.1.8,!=0.1.8.post0,<0.2,>=0.1.6->flashinfer-python==0.6.14) (4.16.0)
Requirement already satisfied: nvidia-cutlass-dsl-libs-base==4.5.2 in /usr/local/lib/python3.12/dist-packages (from nvidia-cutlass-dsl>=4.5.0->flashinfer-python==0.6.14) (4.5.2)
Requirement already satisfied: cuda-python>=12.8 in /usr/local/lib/python3.12/dist-packages (from nvidia-cutlass-dsl-libs-base==4.5.2->nvidia-cutlass-dsl>=4.5.0->flashinfer-python==0.6.14) (13.3.1)
Requirement already satisfied: cuda-bindings~=13.3.1 in /usr/local/lib/python3.12/dist-packages (from cuda-python>=12.8->nvidia-cutlass-dsl-libs-base==4.5.2->nvidia-cutlass-dsl>=4.5.0->flashinfer-python==0.6.14) (13.3.1)
Requirement already satisfied: cuda-core~=1.0.0 in /usr/local/lib/python3.12/dist-packages (from cuda-python>=12.8->nvidia-cutlass-dsl-libs-base==4.5.2->nvidia-cutlass-dsl>=4.5.0->flashinfer-python==0.6.14) (1.0.1)
Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-python>=12.8->nvidia-cutlass-dsl-libs-base==4.5.2->nvidia-cutlass-dsl>=4.5.0->flashinfer-python==0.6.14) (1.5.6)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->flashinfer-python==0.6.14) (3.4.9)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->flashinfer-python==0.6.14) (3.18)
Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.12/dist-packages (from requests->flashinfer-python==0.6.14) (2.7.0)
Requirement already satisfied: certifi>=2023.5.7 in /usr/local/lib/python3.12/dist-packages (from requests->flashinfer-python==0.6.14) (2026.6.17)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (3.29.7)
Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (80.10.2)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (2026.6.0)
Requirement already satisfied: cuda-toolkit==13.0.2 in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.2)
Requirement already satisfied: nvidia-cudnn-cu13==9.19.0.56 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (9.19.0.56)
Requirement already satisfied: nvidia-cusparselt-cu13==0.8.0 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (0.8.0)
Requirement already satisfied: nvidia-nccl-cu13==2.28.9 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (2.28.9)
Requirement already satisfied: nvidia-nvshmem-cu13==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (3.4.5)
Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch->flashinfer-python==0.6.14) (3.6.0)
Requirement already satisfied: nvidia-cublas==13.1.0.3.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.1.0.3)
Requirement already satisfied: nvidia-cuda-runtime==13.0.96.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.96)
Requirement already satisfied: nvidia-cufft==12.0.0.61.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (12.0.0.61)
Requirement already satisfied: nvidia-cufile==1.15.1.6.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (1.15.1.6)
Requirement already satisfied: nvidia-cuda-cupti==13.0.85.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.85)
Requirement already satisfied: nvidia-curand==10.4.0.35.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (10.4.0.35)
Requirement already satisfied: nvidia-cusolver==12.0.4.66.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (12.0.4.66)
Requirement already satisfied: nvidia-cusparse==12.6.3.3.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (12.6.3.3)
Requirement already satisfied: nvidia-nvjitlink==13.0.88.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.88)
Requirement already satisfied: nvidia-cuda-nvrtc==13.0.88.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.88)
Requirement already satisfied: nvidia-nvtx==13.0.85.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->flashinfer-python==0.6.14) (13.0.85)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch->flashinfer-python==0.6.14) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch->flashinfer-python==0.6.14) (3.0.3)
Downloading flashinfer_python-0.6.14-py3-none-any.whl (14.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.6/14.6 MB 5.0 MB/s  0:00:02
Downloading cuda_tile-1.5.0-cp312-cp312-manylinux2014_x86_64.whl (324 kB)
Installing collected packages: cuda-tile, flashinfer-python
  Attempting uninstall: cuda-tile
    Found existing installation: cuda-tile 1.3.0
    Uninstalling cuda-tile-1.3.0:
      Successfully uninstalled cuda-tile-1.3.0
  Attempting uninstall: flashinfer-python
    Found existing installation: flashinfer-python 0.6.13
    Uninstalling flashinfer-python-0.6.13:
      Successfully uninstalled flashinfer-python-0.6.13

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
vllm 0.25.0 requires flashinfer-python==0.6.13, but you have flashinfer-python 0.6.14 which is incompatible.
Successfully installed cuda-tile-1.5.0 flashinfer-python-0.6.14
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Found existing installation: flashinfer-cubin 0.6.13
Uninstalling flashinfer-cubin-0.6.13:
  Successfully uninstalled flashinfer-cubin-0.6.13
Found existing installation: flashinfer-jit-cache 0.6.13+cu130
Uninstalling flashinfer-jit-cache-0.6.13+cu130:
  Successfully uninstalled flashinfer-jit-cache-0.6.13+cu130
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
>>> flashinfer version: 0.6.14
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339] 
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339]        █     █     █▄   ▄█
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.25.0
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339]   █▄█▀ █     █     █     █  model   deepseek-ai/DeepSeek-V4-Flash
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:339] 
(APIServer pid=57) INFO 07-12 02:33:24 [api_utils.py:273] non-default args: {'model_tag': 'deepseek-ai/DeepSeek-V4-Flash', 'enable_auto_tool_choice': True, 'tool_call_parser': 'deepseek_v4', 'host': '0.0.0.0', 'port': 15004, 'api_key': ['aabduh'], 'model': 'deepseek-ai/DeepSeek-V4-Flash', 'tokenizer_mode': 'deepseek_v4', 'trust_remote_code': True, 'max_model_len': 1048576, 'served_model_name': ['d4f'], 'reasoning_parser': 'deepseek_v4', 'tensor_parallel_size': 2, 'disable_custom_all_reduce': True, 'gpu_memory_utilization': 0.95, 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, 'max_num_batched_tokens': 4096, 'max_num_seqs': 64, 'enable_chunked_prefill': True, 'async_scheduling': True, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2}}
(APIServer pid=57) INFO 07-12 02:33:24 [arg_utils.py:772] HF_HUB_OFFLINE is True, replace model_id [deepseek-ai/DeepSeek-V4-Flash] to model_path [/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1]
(APIServer pid=57) WARNING 07-12 02:33:24 [envs.py:2041] Unknown vLLM environment variable detected: VLLM_BUILD_URL
(APIServer pid=57) WARNING 07-12 02:33:24 [envs.py:2041] Unknown vLLM environment variable detected: VLLM_IMAGE_TAG
(APIServer pid=57) WARNING 07-12 02:33:24 [envs.py:2041] Unknown vLLM environment variable detected: VLLM_BUILD_PIPELINE
(APIServer pid=57) WARNING 07-12 02:33:24 [envs.py:2041] Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT
(APIServer pid=57) INFO 07-12 02:33:24 [config.py:742] Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.
(APIServer pid=57) INFO 07-12 02:33:24 [model.py:619] Resolved architecture: DeepseekV4ForCausalLM
(APIServer pid=57) INFO 07-12 02:33:24 [model.py:1776] Using max model len 1048576
(APIServer pid=57) INFO 07-12 02:33:26 [cache.py:286] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor
(APIServer pid=57) INFO 07-12 02:33:26 [model.py:619] Resolved architecture: DeepSeekV4MTPModel
(APIServer pid=57) INFO 07-12 02:33:26 [model.py:1776] Using max model len 1048576
(APIServer pid=57) WARNING 07-12 02:33:26 [speculative.py:866] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer,which may result in lower acceptance rate
(APIServer pid=57) INFO 07-12 02:33:26 [scheduler.py:252] Chunked prefill is enabled with max_num_batched_tokens=4096.
(APIServer pid=57) INFO 07-12 02:33:26 [vllm.py:1042] Asynchronous scheduling is enabled.
(APIServer pid=57) INFO 07-12 02:33:26 [vllm.py:1128] Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out.
(APIServer pid=57) WARNING 07-12 02:33:26 [vllm.py:1134] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
(APIServer pid=57) WARNING 07-12 02:33:26 [vllm.py:1144] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=57) INFO 07-12 02:33:26 [kernel.py:292] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(APIServer pid=57) WARNING 07-12 02:33:26 [vllm.py:1648] max_num_scheduled_tokens is set to 4096 based on the speculative decoding settings. This may lead to suboptimal performance. Consider increasing max_num_batched_tokens to accommodate the additional draft token slots, or decrease num_speculative_tokens or max_num_seqs.
(APIServer pid=57) INFO 07-12 02:33:26 [compilation.py:312] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=395) INFO 07-12 02:33:32 [core.py:114] Initializing a V1 LLM engine (v0.25.0) with config: model='/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1', speculative_config=SpeculativeConfig(method='mtp', model='/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1', num_spec_tokens=2), tokenizer='/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1', skip_tokenizer_init=False, tokenizer_mode=deepseek_v4, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=1048576, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=deepseek_v4_fp8, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=fp8, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='deepseek_v4', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=d4f, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['+quant_fp8', 'all', '+quant_fp8'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [4096], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 384, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=395) WARNING 07-12 02:33:32 [multiproc_executor.py:1067] Reducing Torch parallelism from 24 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
(EngineCore pid=395) INFO 07-12 02:33:32 [multiproc_executor.py:140] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=192.168.0.8 (local), world_size=2, local_world_size=2
(Worker pid=514) INFO 07-12 02:33:38 [parallel_state.py:1607] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:45059 backend=nccl
(Worker pid=515) INFO 07-12 02:33:38 [parallel_state.py:1607] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:45059 backend=nccl
(Worker pid=514) INFO 07-12 02:33:38 [pynccl.py:113] vLLM is using nccl==2.28.9
(Worker pid=514) WARNING 07-12 02:33:38 [symm_mem.py:66] SymmMemCommunicator: Device capability 12.0 not supported, communicator is not available.
(Worker pid=514) INFO 07-12 02:33:38 [cuda_communicator.py:264] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=515) WARNING 07-12 02:33:38 [symm_mem.py:66] SymmMemCommunicator: Device capability 12.0 not supported, communicator is not available.
(Worker pid=514) INFO 07-12 02:33:38 [cuda_communicator.py:264] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=514) INFO 07-12 02:33:38 [parallel_state.py:1942] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=514) INFO 07-12 02:33:39 [topk_topp_sampler.py:55] Using FlashInfer for top-p & top-k sampling.
(Worker pid=514) WARNING 07-12 02:33:39 [__init__.py:204] min_p and logit_bias parameters won't work with speculative decoding.
(Worker pid=515) WARNING 07-12 02:33:39 [__init__.py:204] min_p and logit_bias parameters won't work with speculative decoding.
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [gpu_model_runner.py:5209] Starting to load model /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1...
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [quant_config.py:75] DeepSeek V4 expert_dtype resolved to 'fp4'
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [__init__.py:600] Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [deep_gemm.py:175] deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [deep_gemm.py:202] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [deep_gemm.py:120] DeepGEMM E8M0 enabled on current platform.
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [attention.py:91] Using DeepSeek's fp8_ds_mla KV cache format.
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [mxfp4.py:622] Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend.
(Worker_TP0 pid=514) INFO 07-12 02:33:39 [attention.py:694] Using FP8 indexer cache for Lightning Indexer.
(Worker_TP0 pid=514) INFO 07-12 02:33:40 [weight_utils.py:849] Filesystem type for checkpoints: EXT4. Checkpoint size: 148.66 GiB. Available RAM: 229.53 GiB.
(Worker_TP0 pid=514) INFO 07-12 02:33:40 [weight_utils.py:872] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/46 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   4% Completed | 2/46 [00:00<00:09,  4.86it/s]
Loading safetensors checkpoint shards:   7% Completed | 3/46 [00:00<00:11,  3.60it/s]
Loading safetensors checkpoint shards:   9% Completed | 4/46 [00:01<00:13,  3.17it/s]
Loading safetensors checkpoint shards:  11% Completed | 5/46 [00:01<00:13,  2.99it/s]
Loading safetensors checkpoint shards:  13% Completed | 6/46 [00:01<00:13,  2.91it/s]
Loading safetensors checkpoint shards:  15% Completed | 7/46 [00:02<00:13,  2.84it/s]
Loading safetensors checkpoint shards:  17% Completed | 8/46 [00:02<00:13,  2.80it/s]
Loading safetensors checkpoint shards:  20% Completed | 9/46 [00:03<00:13,  2.78it/s]
Loading safetensors checkpoint shards:  22% Completed | 10/46 [00:03<00:12,  2.78it/s]
Loading safetensors checkpoint shards:  24% Completed | 11/46 [00:03<00:12,  2.79it/s]
Loading safetensors checkpoint shards:  26% Completed | 12/46 [00:04<00:12,  2.78it/s]
Loading safetensors checkpoint shards:  28% Completed | 13/46 [00:04<00:11,  2.76it/s]
Loading safetensors checkpoint shards:  30% Completed | 14/46 [00:04<00:11,  2.75it/s]
Loading safetensors checkpoint shards:  33% Completed | 15/46 [00:05<00:11,  2.74it/s]
Loading safetensors checkpoint shards:  35% Completed | 16/46 [00:05<00:11,  2.68it/s]
Loading safetensors checkpoint shards:  37% Completed | 17/46 [00:05<00:11,  2.58it/s]
Loading safetensors checkpoint shards:  39% Completed | 18/46 [00:06<00:11,  2.48it/s]
Loading safetensors checkpoint shards:  41% Completed | 19/46 [00:06<00:11,  2.41it/s]
Loading safetensors checkpoint shards:  43% Completed | 20/46 [00:07<00:10,  2.39it/s]
Loading safetensors checkpoint shards:  46% Completed | 21/46 [00:07<00:10,  2.39it/s]
Loading safetensors checkpoint shards:  48% Completed | 22/46 [00:08<00:10,  2.37it/s]
Loading safetensors checkpoint shards:  50% Completed | 23/46 [00:08<00:09,  2.36it/s]
Loading safetensors checkpoint shards:  52% Completed | 24/46 [00:09<00:09,  2.32it/s]
Loading safetensors checkpoint shards:  54% Completed | 25/46 [00:09<00:09,  2.28it/s]
Loading safetensors checkpoint shards:  57% Completed | 26/46 [00:09<00:08,  2.27it/s]
Loading safetensors checkpoint shards:  59% Completed | 27/46 [00:10<00:08,  2.23it/s]
Loading safetensors checkpoint shards:  61% Completed | 28/46 [00:10<00:08,  2.22it/s]
Loading safetensors checkpoint shards:  63% Completed | 29/46 [00:11<00:07,  2.24it/s]
Loading safetensors checkpoint shards:  65% Completed | 30/46 [00:11<00:07,  2.23it/s]
Loading safetensors checkpoint shards:  67% Completed | 31/46 [00:12<00:06,  2.24it/s]
Loading safetensors checkpoint shards:  70% Completed | 32/46 [00:12<00:06,  2.23it/s]
Loading safetensors checkpoint shards:  72% Completed | 33/46 [00:13<00:05,  2.22it/s]
Loading safetensors checkpoint shards:  74% Completed | 34/46 [00:13<00:05,  2.22it/s]
Loading safetensors checkpoint shards:  76% Completed | 35/46 [00:14<00:05,  2.06it/s]
Loading safetensors checkpoint shards:  78% Completed | 36/46 [00:14<00:04,  2.14it/s]
Loading safetensors checkpoint shards:  80% Completed | 37/46 [00:14<00:04,  2.22it/s]
Loading safetensors checkpoint shards:  83% Completed | 38/46 [00:15<00:03,  2.29it/s]
Loading safetensors checkpoint shards:  85% Completed | 39/46 [00:15<00:02,  2.35it/s]
Loading safetensors checkpoint shards:  87% Completed | 40/46 [00:16<00:02,  2.35it/s]
Loading safetensors checkpoint shards:  89% Completed | 41/46 [00:16<00:02,  2.36it/s]
Loading safetensors checkpoint shards:  91% Completed | 42/46 [00:17<00:01,  2.30it/s]
Loading safetensors checkpoint shards:  93% Completed | 43/46 [00:17<00:01,  2.29it/s]
Loading safetensors checkpoint shards:  96% Completed | 44/46 [00:17<00:00,  2.29it/s]
Loading safetensors checkpoint shards: 100% Completed | 46/46 [00:18<00:00,  3.85it/s]
Loading safetensors checkpoint shards: 100% Completed | 46/46 [00:18<00:00,  2.55it/s]
(Worker_TP0 pid=514) 
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [default_loader.py:430] Loading weights took 18.05 seconds
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [mxfp4.py:1709] Using MoEPrepareAndFinalizeNoDPEPModular
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [mxfp4.py:1710] Using DeepGemmFP4Experts
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [gpu_model_runner.py:5233] Loading drafter model...
(Worker_TP1 pid=515) WARNING 07-12 02:33:58 [vllm.py:1144] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(Worker_TP1 pid=515) INFO 07-12 02:33:58 [kernel.py:292] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [vllm.py:1042] Asynchronous scheduling is enabled.
(Worker_TP0 pid=514) WARNING 07-12 02:33:58 [vllm.py:1134] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
(Worker_TP0 pid=514) WARNING 07-12 02:33:58 [vllm.py:1144] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [kernel.py:292] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(Worker_TP0 pid=514) WARNING 07-12 02:33:58 [vllm.py:1648] max_num_scheduled_tokens is set to 4096 based on the speculative decoding settings. This may lead to suboptimal performance. Consider increasing max_num_batched_tokens to accommodate the additional draft token slots, or decrease num_speculative_tokens or max_num_seqs.
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [compilation.py:312] Enabled custom fusions: norm_quant, act_quant
(Worker_TP0 pid=514) INFO 07-12 02:33:58 [weight_utils.py:849] Filesystem type for checkpoints: EXT4. Checkpoint size: 148.66 GiB. Available RAM: 229.41 GiB.
Loading safetensors checkpoint shards:   0% Completed | 0/46 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  13% Completed | 6/46 [00:00<00:00, 58.86it/s]
Loading safetensors checkpoint shards:  26% Completed | 12/46 [00:00<00:00, 52.97it/s]
Loading safetensors checkpoint shards:  39% Completed | 18/46 [00:00<00:00, 52.20it/s]
Loading safetensors checkpoint shards:  52% Completed | 24/46 [00:00<00:00, 51.71it/s]
Loading safetensors checkpoint shards:  65% Completed | 30/46 [00:00<00:00, 51.74it/s]
Loading safetensors checkpoint shards:  78% Completed | 36/46 [00:00<00:00, 51.28it/s]
Loading safetensors checkpoint shards:  91% Completed | 42/46 [00:00<00:00, 50.24it/s]
Loading safetensors checkpoint shards: 100% Completed | 46/46 [00:01<00:00, 36.22it/s]
(Worker_TP0 pid=514) 
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [mtp.py:479] MTP draft model loaded: 39 params
(Worker_TP1 pid=515) INFO 07-12 02:33:59 [llm_base_proposer.py:1466] Detected MTP model. Sharing target model embedding weights with the draft model.
(Worker_TP1 pid=515) INFO 07-12 02:33:59 [llm_base_proposer.py:1542] Detected MTP model. Sharing target model lm_head weights with the draft model.
(Worker_TP1 pid=515) INFO 07-12 02:33:59 [llm_base_proposer.py:1566] Shared target model lm_head with MTP shared_head.head.
(Worker_TP1 pid=515) INFO 07-12 02:33:59 [llm_base_proposer.py:1581] Detected MTP model with topk_indices_buffer. Sharing target model topk_indices_buffer with the draft model.
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [default_loader.py:430] Loading weights took 1.32 seconds
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [llm_base_proposer.py:1466] Detected MTP model. Sharing target model embedding weights with the draft model.
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [llm_base_proposer.py:1542] Detected MTP model. Sharing target model lm_head weights with the draft model.
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [llm_base_proposer.py:1566] Shared target model lm_head with MTP shared_head.head.
(Worker_TP0 pid=514) INFO 07-12 02:33:59 [llm_base_proposer.py:1581] Detected MTP model with topk_indices_buffer. Sharing target model topk_indices_buffer with the draft model.
(Worker_TP0 pid=514) INFO 07-12 02:34:00 [gpu_model_runner.py:5306] Model loading took 75.64 GiB memory and 20.219445 seconds
(Worker_TP0 pid=514) INFO 07-12 02:34:00 [breakable_cudagraph.py:288] Breakable CUDA graph enabled
(Worker_TP0 pid=514) INFO 07-12 02:34:00 [interface.py:620] Setting kv cache block size to 256 for DEEPSEEK_SPARSE_SWA backend.
(Worker_TP1 pid=515) INFO 07-12 02:34:00 [interface.py:620] Setting kv cache block size to 256 for DEEPSEEK_SPARSE_SWA backend.
(Worker_TP1 pid=515) 2026-07-12 02:34:02  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:34:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:34:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_post_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:34:10  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_post_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:34:12  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `hc_head_fuse_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:34:15  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `hc_head_fuse_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:34:02  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:34:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:34:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_post_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:34:10  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_post_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:34:12  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `hc_head_fuse_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:34:15  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `hc_head_fuse_tilelang`
(Worker_TP0 pid=514) INFO 07-12 02:34:56 [indexer.py:297] DSA indexer decode path: use_flattening=True (next_n=3, use_fp4_indexer_cache=False)
(Worker_TP1 pid=515) INFO 07-12 02:34:56 [gpu_model_runner.py:6534] Profiling CUDA graph memory: PIECEWISE=42 (largest=384), FULL=26 (largest=192)
(Worker_TP0 pid=514) INFO 07-12 02:34:56 [gpu_model_runner.py:6534] Profiling CUDA graph memory: PIECEWISE=42 (largest=384), FULL=26 (largest=192)
(EngineCore pid=395) INFO 07-12 02:35:01 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(Worker_TP1 pid=515) 2026-07-12 02:34:57  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:02  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:35:03  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:34:57  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:02  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:03  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:08  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) INFO 07-12 02:35:26 [gpu_model_runner.py:6639] Estimated CUDA graph memory: 1.58 GiB total
(Worker_TP1 pid=515) INFO 07-12 02:35:26 [gpu_model_runner.py:6639] Estimated CUDA graph memory: 1.58 GiB total
(Worker_TP0 pid=514) INFO 07-12 02:35:26 [gpu_worker.py:538] Available KV cache memory: 10.29 GiB
(Worker_TP0 pid=514) INFO 07-12 02:35:26 [gpu_worker.py:553] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9500 is equivalent to --gpu-memory-utilization=0.9334 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9666. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(Worker_TP1 pid=515) INFO 07-12 02:35:26 [gpu_worker.py:553] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9500 is equivalent to --gpu-memory-utilization=0.9334 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9666. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=395) INFO 07-12 02:35:26 [kv_cache_utils.py:2146] GPU KV cache size: 1,993,100 tokens
(EngineCore pid=395) INFO 07-12 02:35:26 [kv_cache_utils.py:2147] Maximum concurrency for 1,048,576 tokens per request: 1.90x
(Worker_TP0 pid=514) INFO 07-12 02:35:26 [flashinfer_sparse_mla_warmup.py:233] Warming up DeepSeek V4 sparse MLA attention for mixed tokens=16.
(Worker_TP1 pid=515) INFO 07-12 02:35:26 [flashinfer_sparse_mla_warmup.py:233] Warming up DeepSeek V4 sparse MLA attention for mixed tokens=16.
(Worker_TP0 pid=514) INFO 07-12 02:35:26 [flashinfer_sparse_mla_warmup.py:124] Autotuning FlashInfer SM120 sparse MLA DSv4 decode with cache: /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json
(Worker_TP0 pid=514) 2026-07-12 02:35:26,592 - INFO - autotuner.py:1899 - flashinfer.jit: [Autotuner]: Loaded 18 configs from /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json
(Worker_TP0 pid=514) 2026-07-12 02:35:26,593 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP1 pid=515) 2026-07-12 02:35:26  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:31  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:26  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:31  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:32,500 - INFO - autotuner.py:1009 - flashinfer.jit: [Autotuner]: Config cache hit for sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner, source=config file)
(Worker_TP0 pid=514) 2026-07-12 02:35:40,717 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP0 pid=514) 2026-07-12 02:35:40,732 - INFO - autotuner.py:1899 - flashinfer.jit: [Autotuner]: Loaded 18 configs from /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json
(Worker_TP0 pid=514) 2026-07-12 02:35:32  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_fused_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:34  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_fused_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:34  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:40  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) INFO 07-12 02:35:40 [flashinfer_sparse_mla_warmup.py:181] FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 0 from /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json.
(Worker_TP1 pid=515) 2026-07-12 02:35:40,732 - INFO - autotuner.py:1899 - flashinfer.jit: [Autotuner]: Loaded 18 configs from /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json
(Worker_TP1 pid=515) 2026-07-12 02:35:32  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_fused_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:34  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_fused_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:35:34  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:39  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) INFO 07-12 02:35:40 [flashinfer_sparse_mla_warmup.py:181] FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 1 from /cache/vllm/flashinfer_autotune_cache/0.6.14/120f/f5a6dc82f3a982a6eddfb943712f8f2d595d0a8c32401c78ac79272952259761/autotune_configs.json.
DeepGEMM warmup: 100%|██████████| 1064/1064 [00:00<00:00, 17096.06it/s]
(Worker_TP0 pid=514) 2026-07-12 02:35:40,799 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP1 pid=515) 2026-07-12 02:35:40,801 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP0 pid=514) 2026-07-12 02:35:40,947 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP1 pid=515) 2026-07-12 02:35:40,947 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP0 pid=514) INFO 07-12 02:35:40 [cutedsl_warmup.py:97] Skipping CuTeDSL warmup because no compile units were requested.
(Worker_TP1 pid=515) INFO 07-12 02:35:40 [cutedsl_warmup.py:97] Skipping CuTeDSL warmup because no compile units were requested.
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/42 [00:00<?, Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   2%|▏         | 1/42 [00:00<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   5%|▍         | 2/42 [00:00<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   7%|▋         | 3/42 [00:00<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  10%|▉         | 4/42 [00:00<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  12%|█▏        | 5/42 [00:00<00:                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  12%|█▏        | 5/42 [00:00<00:                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  12%|█▏        | 5/42 [00:06<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  14%|█▍        | 6/42 [00:06<01:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  17%|█▋        | 7/42 [00:06<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  19%|█▉        | 8/42 [00:07<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  21%|██▏       | 9/42 [00:07<00:                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  21%|██▏       | 9/42 [00:07<00:                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  21%|██▏       | 9/42 [00:12<00:Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  24%|██▍       | 10/42 [00:13<01Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  26%|██▌       | 11/42 [00:13<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  29%|██▊       | 12/42 [00:13<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  31%|███       | 13/42 [00:13<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  33%|███▎      | 14/42 [00:13<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  36%|███▌      | 15/42 [00:13<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  38%|███▊      | 16/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  40%|████      | 17/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  43%|████▎     | 18/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  45%|████▌     | 19/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  48%|████▊     | 20/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  50%|█████     | 21/42 [00:14<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  52%|█████▏    | 22/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  55%|█████▍    | 23/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  57%|█████▋    | 24/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  60%|█████▉    | 25/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  62%|██████▏   | 26/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  64%|██████▍   | 27/42 [00:15<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  67%|██████▋   | 28/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  69%|██████▉   | 29/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  71%|███████▏  | 30/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  74%|███████▍  | 31/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  76%|███████▌  | 32/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  79%|███████▊  | 33/42 [00:16<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  81%|████████  | 34/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  83%|████████▎ | 35/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  86%|████████▌ | 36/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  88%|████████▊ | 37/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  90%|█████████ | 38/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  93%|█████████▎| 39/42 [00:17<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  95%|█████████▌| 40/42 [00:18<00                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  95%|█████████▌| 40/42 [00:18<00                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  95%|█████████▌| 40/42 [00:19<00                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  95%|█████████▌| 40/42 [00:20<00                                                                                         Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  95%|█████████▌| 40/42 [00:25<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  98%|█████████▊| 41/42 [00:26<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 42/42 [00:26<00Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 42/42 [00:26<00:00,  1.59it/s]
(Worker_TP0 pid=514) 2026-07-12 02:35:42  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:47  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:48  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:35:53  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:35:59  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_fused_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:36:01  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_fused_tilelang`
(Worker_TP0 pid=514) 2026-07-12 02:36:01  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP0 pid=514) 2026-07-12 02:36:06  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
Capturing CUDA graphs (decode, FULL):  65%|██████▌   | 17/26 [00:02<00:01,  5.98it/s](Worker_TP1 pid=515) 2026-07-12 02:36:10,572 - INFO - autotuner.py:1009 - flashinfer.jit: [Autotuner]: Config cache hit for sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner, source=config file)
Capturing CUDA graphs (decode, FULL): 100%|██████████| 26/26 [00:04<00:00,  5.82it/s]
(Worker_TP0 pid=514) INFO 07-12 02:36:12 [gpu_model_runner.py:6707] Graph capturing finished in 31 secs, took 1.73 GiB
(Worker_TP0 pid=514) INFO 07-12 02:36:12 [gpu_worker.py:771] CUDA graph pool memory: 1.73 GiB (actual), 1.58 GiB (estimated), difference: 0.15 GiB (8.7%).
(Worker_TP1 pid=515) 2026-07-12 02:35:42  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:47  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:35:48  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:35:53  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:35:59  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_fused_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:36:01  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_fused_tilelang`
(Worker_TP1 pid=515) 2026-07-12 02:36:01  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang` with `out_idx=None`
(Worker_TP1 pid=515) 2026-07-12 02:36:06  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
(Worker_TP1 pid=515) INFO 07-12 02:36:12 [gpu_worker.py:771] CUDA graph pool memory: 1.73 GiB (actual), 1.58 GiB (estimated), difference: 0.15 GiB (8.7%).
(Worker_TP0 pid=514) INFO 07-12 02:36:12 [jit_monitor.py:73] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP1 pid=515) INFO 07-12 02:36:12 [jit_monitor.py:73] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(EngineCore pid=395) INFO 07-12 02:36:12 [core.py:344] init engine (profile, create kv cache, warmup model) took 132.64 s
(EngineCore pid=395) INFO 07-12 02:36:13 [vllm.py:1042] Asynchronous scheduling is enabled.
(EngineCore pid=395) WARNING 07-12 02:36:13 [vllm.py:1134] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
(EngineCore pid=395) WARNING 07-12 02:36:13 [vllm.py:1144] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=395) INFO 07-12 02:36:13 [kernel.py:292] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(EngineCore pid=395) WARNING 07-12 02:36:13 [vllm.py:1648] max_num_scheduled_tokens is set to 4096 based on the speculative decoding settings. This may lead to suboptimal performance. Consider increasing max_num_batched_tokens to accommodate the additional draft token slots, or decrease num_speculative_tokens or max_num_seqs.
(EngineCore pid=395) INFO 07-12 02:36:13 [compilation.py:312] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=57) INFO 07-12 02:36:13 [api_server.py:612] Supported tasks: ['generate']
(APIServer pid=57) INFO 07-12 02:36:13 [parser_manager.py:37] "auto" tool choice has been enabled.
(APIServer pid=57) WARNING 07-12 02:36:13 [model.py:1528] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 1.0, 'top_p': 1.0}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=57) INFO 07-12 02:36:13 [api_server.py:616] Starting vLLM server on http://0.0.0.0:15004
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:37] Available routes are:
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /generative_scoring, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/chat/completions/derender, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /v1/completions/derender, Methods: POST
(APIServer pid=57) INFO 07-12 02:36:13 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=57) INFO:     Started server process [57]
(APIServer pid=57) INFO:     Waiting for application startup.
(APIServer pid=57) INFO:     Application startup complete.
(APIServer pid=57) INFO:     172.28.0.2:50682 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:55716 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:50456 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.6:47848 - "GET /health HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     127.0.0.1:38120 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:55174 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:55128 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:33574 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.2:54230 - "GET /metrics HTTP/1.1" 200 OK
(APIServer pid=57) INFO:     172.28.0.6:37092 - "GET /health HTTP/1.1" 200 OK
aabduh@threadripper-aabduh:~$ 

```

</details>

