# DeepSeek-V4-Flash vLLM Patch Benchmark

Complete benchmark report for DeepSeek-V4-Flash using custom patched
vLLM builds with:

-   MTP-2 speculative decoding
-   DSpark speculative decoding
-   FlashInfer CUTLASS MoE optimizations
-   DeepSeek V4 specific patches

------------------------------------------------------------------------

# Hardware

Host:

-   Ubuntu 26.04 LTS
-   AMD Threadripper Pro 7965WX
-   24 cores
-   \~256 GB RAM

GPUs:

-   2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
-   Tensor Parallelism: 2
-   PCIe interconnect (no NVLink)

GPU assignment:

-   GPU 0: RTX 5090 (excluded)
-   GPU 1: RTX PRO 6000
-   GPU 2: RTX PRO 6000

------------------------------------------------------------------------

# Patch Summary

## MTP-2 Image

Image:

    vllm-ds4:v0.25.0-patched

Digest:

    vllm-ds4@sha256:74fa0eec47faa81a8fca3d9d372494ffeb16a60942cb20088a04db1504bb17f0

Included:

-   PR #48303 - DeepSeek MXFP4 FlashInfer CUTLASS MoE support
-   PR #48304 - DeepSeek V4 MTP compression / RoPE fix
-   PR #48317 - hybrid KV-cache capacity reporting fix
-   flashinfer-python 0.6.14
-   NVRTC development headers

------------------------------------------------------------------------

# MTP-2 Deployment

Model:

    deepseek-ai/DeepSeek-V4-Flash

Served name:

    d4f

Port:

    15004

Speculative decoding:

    method: mtp
    num_speculative_tokens: 2

Important runtime flags:

    --tensor-parallel-size 2
    --kv-cache-dtype fp8
    --gpu-memory-utilization 0.95
    --max-model-len 1048576
    --max-num-seqs 64
    --max-num-batched-tokens 4096
    --enable-prefix-caching
    --enable-chunked-prefill
    --moe-backend flashinfer_cutlass
    --async-scheduling

------------------------------------------------------------------------

# DSpark Deployment

Image:

    vllm-ds4:v0.25.0-dspark-topk256

Digest:

    vllm-ds4@sha256:e6176d085417967353c7e998ec2cc3b46b7edd87221e1aef44150a1076ecc9c1

Changes:

-   Added FlashInfer SM120 sparse MLA topk=256 support
-   Enabled DSpark speculative decoding
-   Reused existing compiled kernel cache

Final speculative configuration:

    method: dspark
    num_speculative_tokens: 5
    draft_sample_method: greedy

------------------------------------------------------------------------

# DSpark Stability Fix

Initial configuration:

    --gpu-memory-utilization 0.95

Result:

-   OOM during extended requests

Final configuration:

    --gpu-memory-utilization 0.93

Final benchmark power:

    GPU 1: 450W
    GPU 2: 450W

------------------------------------------------------------------------

# Benchmark Configuration

Tool:

    llm-bench

Concurrency:

    1,2,4,8,16,32,64

Context lengths:

    0
    8192
    16384
    32768
    65536
    131072
    262144

------------------------------------------------------------------------

# MTP-2 Benchmark Results

## Throughput tok/s

  Context         1       2       4       8      16       32       64
  --------- ------- ------- ------- ------- ------- -------- --------
  0           189.9   297.0   349.4   670.0   994.3   1542.1   2362.2
  8K          194.3   300.8   332.5   607.3   899.9   1303.1   1947.6
  16K         167.0   304.4   328.8   583.5   884.9   1345.3   1891.3
  32K         182.7   273.3   322.1   598.6   833.1   1256.4       \-
  64K         184.1   263.5   321.4   569.7   805.8       \-       \-
  128K        195.5   256.4   306.3   550.6      \-       \-       \-
  256K        182.5   261.5   308.0      \-      \-       \-       \-

------------------------------------------------------------------------

# MTP-2 Speculative Metrics

Captured from:

    mtp2-final-metrics.txt

  Metric                        Value
  ------------------------- ---------
  Speculative rounds          422,517
  Draft tokens                845,034
  Accepted tokens             439,151
  Generated tokens            862,227
  Draft acceptance             51.97%
  Accepted tokens / round       1.039

Prefix cache:

  Metric                  Value
  ---------------- ------------
  Prefix queries     20,848,155
  Prefix hits        19,462,656
  Hit rate               93.35%

------------------------------------------------------------------------

# DSpark Benchmark Results

Final runs:

    GPU 1: 450W
    GPU 2: 450W

## DSpark Run 1

Context 0:

    1:175.26
    2:251.32
    4:352.85
    8:497.88
    16:741.06
    32:1113.40
    64:1709.70

Context 8192:

    1:215.71
    2:280.53
    4:374.78
    8:630.47
    16:950.89
    32:1212.54
    64:1984.18

## DSpark Run 2

Context 0:

    1:171.15
    2:245.73
    4:345.43
    8:493.56
    16:746.63
    32:1127.43
    64:1725.35

Context 8192:

    1:203.23
    2:258.92
    4:403.26
    8:538.64
    16:744.62
    32:1219.54
    64:1923.63

------------------------------------------------------------------------

# DSpark Speculative Metrics

Small validation capture:

  Metric                 Value
  -------------------- -------
  Speculative rounds         1
  Draft tokens               5
  Accepted tokens            1
  Acceptance               20%
  Accepted / round         1.0

Note:

This was not a full benchmark metric capture.

------------------------------------------------------------------------

# GPU Snapshot

## DSpark Final Snapshot

Captured at 450W:

    GPU 1:
    Power draw: 435.58W
    Temperature: 61C
    SM clock: 2737 MHz
    Memory used: 96658 MiB
    Utilization: 99%

    GPU 2:
    Power draw: 437.43W
    Temperature: 66C
    SM clock: 2700 MHz
    Memory used: 96688 MiB
    Utilization: 99%

------------------------------------------------------------------------

# Runtime Notes

MTP-2 startup:

    init engine (profile, create kv cache, warmup model) took 80.04 s

Confirmed:

    Asynchronous scheduling is enabled.

No observed:

-   CUDA OOM on final MTP-2 run
-   NCCL failures
-   Runtime crashes

------------------------------------------------------------------------

# Remaining Items

For a fully reproducible release:

-   Exact llm-bench command line
-   MTP-2 GPU snapshot CSV contents
-   Full raw benchmark JSON files
-   Final MTP-2 vs DSpark comparison table
