# Hardware and model support

FlashTopK runs on NVIDIA Hopper (SM90) and Blackwell (SM100) GPUs, with FP8 or BF16 operands. The kernels follow the GPU's architecture; adaptive pruning's cost model (`PRUNE_COSTS` in [`flashtopk/plan.py`](../flashtopk/plan.py)) follows the GPU itself.

| GPU | architecture | FP8 | BF16 | adaptive pruning |
|---|---|:---:|:---:|---|
| H800 | SM90 (9.0) | ✅ | ✅ | 16-col |
| H20 | SM90 (9.0) | ✅ | ✅ | 16-col |
| H100 | SM90 (9.0) | ✅ | ✅ | 16-col |
| H200 | SM90 (9.0) | ✅ | ✅ | 16-col |
| other SM90 | SM90 (9.0) | ↩️ H800 | ↩️ H800 | 16-col |
| B200 | SM100 (10.0) | ✅ | ✅ | 16 / 24-col (>=256K keys) |
| GB200 | SM100 (10.0) | ↩️ B200 | ↩️ B200 | 16 / 24-col (>=256K keys) |
| GB300 | SM100 (10.3) | ✅ | ✅ | 16 / 24-col (>=256K keys) |
| B300 | SM100 (10.3) | ↩️ GB300 | ↩️ GB300 | 16 / 24-col (>=256K keys) |

* ✅ **Tuned:** tested, with its own `PRUNE_COSTS` entry.
* ↩️ **Falls back:** uses the `PRUNE_COSTS` of the GPU after the arrow.

## Models

FlashTopK takes indexers with 32 heads of dimension 128 and a ReLU per head; `topk` can be up to 4096.

| model | indexer heads × head dim | supported |
|---|---|:---:|
| [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) | 32 × 128 | ✅ |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 32 × 128 | ✅ |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 32 × 128 | ✅ |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview) | 32 × 128 | ✅ |
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | 64 × 128 | ❌ |

## How a GPU is matched

1. A word of `torch.cuda.get_device_name()` names a `PRUNE_COSTS` entry: `(device, dtype)` for 16-column pruning, and `(device, dtype, 24)` for SM100 24-column pruning (evaluated alongside 16-column pruning from 256K keys).
2. Otherwise the architecture's default: GB300's entry on SM100 (`10.3+`), B200's on SM100 (`10.0`), H800's on SM90.
3. `FLASHTOPK_PRUNE_COSTS="p,c,b"` and `FLASHTOPK_PRUNE_COSTS_24="p,c,b"` override the 16-column and 24-column cost models respectively.

`FLASHTOPK_PRUNING=off` (or `Selector(pruning="off")`) skips the decision and always scans, which makes the cost model irrelevant.
