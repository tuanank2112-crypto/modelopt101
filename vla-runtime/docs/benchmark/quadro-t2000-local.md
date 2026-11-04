# Quadro T2000 (CUDA)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the engine only: no transport, no simulator, and no claim about task success. Scope: SmolVLA only.

## Test setup

| | |
|---|---|
| Device | NVIDIA Quadro T2000, 4096 MiB VRAM, Turing sm_75 |
| Host | Intel Core i7-10850H (6 cores, 12 threads), WSL-visible RAM 15810.6 MiB |
| Power | Power profile and AC/battery state not recorded. GPU power limit is unavailable in WSL; sampled rollout temperature 55–61°C. No power setting was changed |
| OS | Ubuntu 26.04 LTS, WSL2 kernel 6.18.33.2-microsoft-standard-WSL2 |
| Driver / toolkit | NVIDIA 580.92; CUDA 12.4; GCC 13.4.0 |
| Commit | `4a542ed` plus [local Turing patch](../../../robot-results/turing-local.patch). All CUDA measurements use this same build |
| llama.cpp | `b11223` |
| Date | 2026-11-04 15:53:45 to 15:56:28 (UTC+07) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`); Ampere-only BitVLA/FoldQuant archives use SM80, not validated on this GPU |
| Runtime flags | defaults for reported row; screen covers the eight flag sets below |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with lowest mean is reported; host memory is peak over all three. Screen: 2 warmups + 5 timed reps per flag set |

**Device memory MiB** is unavailable: WSL reports `[N/A]` for per-process `nvidia-smi --query-compute-apps` readings sampled during the processes at approximately 0.5-second intervals. No device-wide or weight-buffer value is substituted. **Host RSS MiB** is peak resident set size from `/usr/bin/time -v` (Linux getrusage), maximum across the three full runs.

## Model configuration

| Model | Checkpoint | Views | Input | Lang tokens | Action chunk | Action dim | Action head | Steps |
|---|---|--:|--:|--:|--:|--:|---|--:|
| SmolVLA | `vrfai/smolvla-libero-gguf` | 2 | 512 | 16 | 50 | 7 | flow matching | 10 |

Checkpoint padded action width is 32; LIBERO consumes 7 dimensions. SHA256: `6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8`. Default logs show tower weights resident BF16 and LM GEMM weights widened to F32 for default `mm-prec f32`. This differs from the rollout's explicit FP16 setting.

## Latency and memory

Rows report `predict()` with synthetic inputs. Process 2 has the lowest mean (544.1 ms); its min/p50/p90/vision are copied as printed. The three process means are 544.5, 544.1, 544.3 ms.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Device memory MiB | Host RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SmolVLA | 2 | 512 | 542.3 | 544.1 | 543.7 | 545.4 | 300.0 | N/A | 450.5 |

### Fastest configuration

Each flag set below was screened in one process with 2 warmups and 5 reps. A candidate replaces defaults only if its mean is over 3% lower in both screen and full re-run. Flash attention improved screen mean from 536.8 to 533.7 ms (about 0.6%), below the threshold. No candidate qualified; the default full-protocol row is retained, with no extra candidate re-run needed.

Flash attention and dtype flags may change the action chunk. The rollout success rate belongs to its FP16 configuration and must not be applied to the default benchmark row.

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Device memory MiB | Host RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SmolVLA | *(defaults)* | 542.3 | 544.1 | 543.7 | 545.4 | 300.0 | N/A | 450.5 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SmolVLA | 536.8 | 533.7 | 583.9 | 578.0 | 582.9 | 579.1 | 1197.2 | 1232.8 |

</details>

## Not run

- **CPU full benchmark**: stopped at the user's request. Initial CPU screen logs are retained as incomplete raw data and excluded from the tables; no CPU benchmark claim is made.
- **Other models**: not attempted; this task downloaded and evaluated SmolVLA only. The BitVLA/FoldQuant archives linking successfully does not establish compatibility or correctness for those policies on Turing.

## Notes

- Local Turing changes remove Ampere-only TF32-attention and skinny-BF16-GEMM candidates and retain existing fallback paths. This is a modified local build, not an unchanged upstream performance result.
- Separate LIBERO-Object evaluation: **173/200 successes (86.5%)**, 10 tasks × 20 episodes, 500-step cap, seed 42, replay 10. It used **resident FP16**, two 512-pixel cameras and the client's 48-token prompt budget. All 200 videos were checked with ffprobe; 27 failures are retained.
- [Task-success report and all videos](../../../robot-results/BAO_CAO_KIEM_THU_ROBOT.md) are separate from this synthetic latency report. Unpaired inference noise and differing dtypes prevent a controlled backend-quality comparison.
- Rollout per-environment-step timings divide across replayed actions; they are not per-predict latency and are not used in these tables.
- [Raw measurements](../../../robot-results/template-benchmark/results.json) include three completed CUDA full runs, eight CUDA screen runs and initial, incomplete CPU screen rows. Raw logs for each process are in the same directory.
- A short rollout utilization sample used 1257/4096 MiB device-wide VRAM with GPU activity 0–100%; that sample is not the memory probe for this table.

## Reproducing

```bash
# Start from upstream commit 4a542ed and apply the local patch:
git apply ../robot-results/turing-local.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON
cmake --build build --target vla-bench vla-server -j4

# Run three separate processes; collect peak RSS and per-process CUDA memory:
./build/vla-bench --ckpt models/smolvla/smolvla-libero.gguf --images 2 --size 512 --tokens 16 --warmup 3 --reps 20
# Screen each flag set above with --warmup 2 --reps 5.
# Keep synthetic timing runs separate from server/simulator and compilation workloads.
```
