# Model Optimization 101

Hands-on exercises for learning the core ideas of **model optimization** for edge deployment.

## The five ideas

| # | Exercise | One line | Core to focus on |
|---|----------|----------|------------------|
| 1 | [Quantization](examples/01_quantization.ipynb) | Use fewer bits to represent numbers | quantization scheme & formula, compression ratio |
| 2 | [Knowledge Distillation](examples/02_knowledge_distillation.ipynb) | A big model teaches a small one | distillation loss formula, teacher-student architecture |
| 3 | [Pruning](examples/03_pruning.ipynb) | Remove unused signals | granularity, ratio, criteria |
| 4 | [Neural Architecture Search](examples/04_nas.ipynb) | Search for an optimal architecture | search space & strategy |
| 5 | [Kernel Optimization](examples/05_kernel_optimization.ipynb) | Make the same math run faster on the GPU | tiling, grid layout, memory access |


Kernel optimization is more hardware-constrained and usually speeds up the computation without changing the math.
Meanwhile, the remaining ideas are more abstract and usually change what the model computes.
They are orthogonal in concept and may be coupled in realization. For example, a **quantized matmul kernel** is entirely possible.

## How to run

### Google Colab

1. Open the notebook for an exercise (links above).
2. Click **"Open in Colab"** (badge at the top of each notebook), or upload the `.ipynb`
   to Colab manually.
3. Set the runtime to **GPU** (`Runtime -> Change runtime type -> GPU`).
4. `Runtime -> Run all`.

### Local

To run locally instead:

```bash
pip install -r requirements.txt
jupyter lab
```

## Reading guide

The **core concept is highlighted** in every notebook.
Everything else - data loading, training loops, plotting - can be skimmed.
Every exercise **records model size, latency and accuracy before vs. after** the optimization so the trade-off can be considered.

## Applications

Deploying optimized models on a laptop:

- [apps/tinyml_chat/](apps/tinyml_chat/) - a from-scratch LLM chat app, following [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) and the [tinychat-tutorial](https://github.com/mit-han-lab/tinychat-tutorial).

![chat](https://github.com/mit-han-lab/tinychat-tutorial/blob/main/assets/figures/chat.gif)

- [apps/vla.cpp/](apps/vla.cpp/) - run **SmolVLA** with [`vla.cpp`](https://fai-modelopt-tech.github.io/vla-cpp.github.io/), a C++ VLA inference runtime built on `llama.cpp`.

![video](assets/smolvla-libero-object.gif)
## Kết quả thực nghiệm trên máy

- [Báo cáo Quantization](BAO_CAO_TOI_UU_HOA_QUANTIZATION.md)
- [Báo cáo Distillation, Pruning, NAS và Kernel](BAO_CAO_CAC_BAI_TOAN_MODELOPT_CON_LAI.md)
- [Kiểm thử robot GPU: 173/200 thành công](robot-results/BAO_CAO_KIEM_THU_ROBOT.md)
- [Danh sách 200 video](robot-results/DANH_SACH_VIDEO_GPU.md)
- [Benchmark Quadro T2000 theo template](vla-runtime/docs/benchmark/quadro-t2000-local.md)
