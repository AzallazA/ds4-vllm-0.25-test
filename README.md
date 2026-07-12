# DeepSeek-V4 Flash on RTX PRO 6000 Blackwell

This repository contains reproducible benchmarks and configuration notes for running DeepSeek-V4-Flash on two RTX PRO 6000 Blackwell GPUs.

## Benchmarks

| Configuration | Status | Document |
|---------------|:------:|----------|
| Stock vLLM 0.25.0 + FlashInfer 0.6.14 | ✅ | [Stock Baseline](docs/stock-vllm-v0.25.md) |
| Patched vLLM (MTP2 + DSpark) | ✅ | [Patched MTP2 + DSpark](docs/patched-mtp2-dspark.md) |

## Comparison

The patched benchmark includes:

- MTP2 improvements
- DSpark speculative decoding
- Updated FlashInfer kernels
- Performance comparison against the stock vLLM baseline

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Tensor Parallel = 2
- 450 W power limit
- Ubuntu 26.04 LTS