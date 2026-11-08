# Weekly report, 03/11 to 07/11/2026

Reported by: LeNhuY
Scope: learning the basics of model optimization for robots, then using that to steer experiments on the vla.cpp repo (Quadro T2000 4 GB GPU, running in WSL)

## Summary

At the start of the week I only knew the names of the core topics: quantization, distillation, kernels. This week had two steps.

**Step 1: learn from the most basic concepts.**
- I did not just read theory. I ran the ACT model on my own machine, guessed the result before each measurement, then compared my guess with the real number.
- I started from the basics:
  - what p50 and p90 mean
  - how weights differ from activations
  - how a convolution works (I computed a 3×3 filter by hand)
  - why pruning creates many zeros but the model does not get faster
- I got some answers wrong, and I wrote those down so I understand the correct version.

**Step 2: use what I learned to steer the experiments on the T2000.**
- I asked the questions, chose the direction, rejected conclusions that had no evidence, and stopped proposals that were not needed.

Main results:
- 8 of 9 models now run at default settings on the T2000. Before the fix, all of them failed at defaults.
- GR00T N1.5 used to run out of VRAM. It now takes 0.63 s per prediction, and the robot still succeeds 28/30 times (original model: 27/30).
- GR00T N1.7 is 4.7 GB, bigger than the 4 GB card. It now runs at 0.30 s per prediction, 30/30 successes.
- SmolVLA on LIBERO-Object: 173/200 successful episodes (86.5%), with video.
- The report is on a branch in my fork and has been revised based on the repo owner's review.

Insights:
1. **"It runs" does not mean "it is usable".**
   - The original N1.7 gave no error, but it took 12.5 s per prediction.
   - The part that did not fit in VRAM was silently moved to system RAM.
   - You have to measure the time and compare VRAM used against the model size.
2. **The default-mode failures came from kernels, not from the models.**
   - Part of the kernels was compiled only for a newer GPU generation (sm_80). The T2000 is sm_75.
   - The error message on screen hid the real cause.
3. **Compression alone is not enough.** GR00T got fast through three changes together:
   - quantize so the model fits in memory
   - build the right kernels for a card without tensor cores
   - remove wasted work in the code
4. **Do not compress every part equally.** I measured each block of the model separately. Compressing the vision part made the output drift 3 times more, so that part stays uncompressed.
5. **Speed must come with a quality check.** I used a two-level check: numerical drift against the original model, then the robot doing the same 30 scenarios so each pair can be compared.
6. **Speeding up one part only saves that part's time.** I learned this from the "CPU 30 ms + GPU 10 ms" exercise. I saw it again when LIBERO evaluation stayed slow even after the model got fast, because the simulator took most of the time.

## Completed Work

### Part A. Learning the basic concepts

**A1. Learning path (05/11)**
- I started from where I stood: I work in the edge optimization team but only had a surface-level understanding. From there I set a learning path.
- The path follows one line: model → computation → memory → runtime → robot control loop.
- I asked for knowledge I can "see": pictures and everyday examples, not just formulas.
- The first lessons were reading only. I asked to change the format so I take part: I guess, I run the command myself, then I get the explanation.

**A2. Hands-on with the ACT model, running `vla-bench` myself in WSL (05/11)**

| My guess / answer | Actual result | What I learned |
|---|---|---|
| A 320×320 image would take 20–30 ms | 18.7 ms (224: 13.0 ms; 640×480: 48.4 ms) | Bigger images are slower, but time does not grow evenly with pixel count |
| Picked 224 and 320 for a "under 25 ms" requirement | Correct by p50, but p90 still leaves 10% of runs unknown | p50 is the middle run; p90 means 90% of runs are at or below that value. A strict limit needs the slow runs too |
| (Forgot to guess) Measuring 200 runs instead of 20 | p50 stayed at 18.7 ms | More runs show the variation better; they do not make the model faster |
| Weight memory stays the same when image size changes: correct | 109.2 MiB at every image size | Weights are the "instruction book", activations are the "scratch paper". Bigger images only need more scratch paper |
| **Wrong:** thought going from 32-bit to 16-bit halves total memory and doubles speed | Only the weight part is halved; speed must be measured again | Weight memory, total memory and speed are three separate things |
| **Partly wrong:** thought it was slow "because there is no graph yet" | The log already showed a CUDA Graph | CUDA Graphs only cut the cost of sending commands; they do not speed up image preparation |
| GPU from 10 ms to 5 ms gives a 35 ms total: correct | 30 + 5 = 35 ms | Find the slowest part first, then optimize it |
| **Wrong:** I did not know what a convolution was, and thought bigger images add weights | A 9-weight filter used at 100 or 200 positions still has 9 weights | One filter is shared across all positions: bigger images mean more computation, not more weights |
| Computed a filter by hand: 4×(−1) + 6×1 = 2, positive | Correct | This filter detects a change from dark to bright, which is an object edge |
| **Wrong:** thought many zeros automatically make it faster | The GPU still multiplies the zeros unless the kernel knows how to skip them | Pruning only helps if the runtime uses the zeros, or if whole rows are removed (structured pruning) |

**A3. Learning from my own experiment results (06/11 and 07/11)**
- On 06/11 I went through the report section by section, reading it as "someone with no background".
- When something was unclear, I kept asking until I understood:
  - "What does 'runs but is not usable' mean?"
  - "What does `CUDA_ARCHITECTURES` actually affect?"
  - "How exactly does the compression work?"
- On 07/11 I said clearly that the goal was to understand all the terms and methods, not to add more to the report.
- Result: I understand how Q8_0 compression works.
  - Weights are split into blocks of 32 numbers.
  - Each block stores 1 scale factor (the largest value divided by 127) plus 32 8-bit integers.
  - Each block takes 34 bytes instead of 64.
- I also understand why GR00T keeps the vision part uncompressed and only compresses the other 3 blocks.

**A4. ModelOpt101 exercises on MNIST (03/11)**
- Completed 5 exercises: quantization, distillation, pruning, NAS, kernels.
- Wrote 2 summary reports.
- Limitation: MNIST only, not yet applied to a robot policy.

### Part B. Steering the experiments on the T2000

The table below lists my decisions and questions, and what each one changed.

| Date | My decision / question | Why | Result |
|---|---|---|---|
| 03/11 | Asked whether the old T2000 numbers were measured or theoretical | No logs came with them | Confirmed there were no logs. Re-measured from scratch instead of reusing them |
| 03/11 | Models that do not run must have a stated reason, and must be tried | The report should not have gaps | Downloaded BitVLA, GR00T and VLA-JEPA to try them |
| 04/11 | GPU first, no switching to CPU on our own | The goal is to measure on the T2000 GPU | Patched kernels so the GPU path works. SmolVLA robot test: 173/200 on GPU |
| 04/11 | Keep the same configuration until all 200 episodes finish; optimize later | Changing mid-run would mix two configurations in one result set | One consistent set of 200 results, with full video |
| 04/11 | Asked for ways to run the failing models, and asked for more research when the first answer was "no way" | Did not accept an early conclusion | Found a replacement kernel for BitVLA and a quantization path for GR00T |
| 04/11 | Re-measure only 6 models; keep the 3 that already had real data | Avoid redundant runs | All 9 models merged into one report |
| 05/11 | Asked "based on what?" about the first explanation of why every model failed at defaults, and asked for deeper research | The first answer was a guess | Found the root cause: some kernels only had an sm_80 build, and the real error was hidden. After the fix, 8/9 models run at defaults |
| 05/11 | The report lacked proof for the checkpoints, so that had to be added | The report must be verifiable | Added SHA-256: 9/9 checkpoints match Hugging Face |
| 05/11 | Asked for a full list of the remaining work before continuing | Wanted to see the full scope | A 7-item list, done in order |
| 06/11 | Goal: cut latency as much as possible **while keeping quality** | Fast but wrong is useless for a robot | Added a two-level quality gate: numerical drift and 30 paired robot scenarios |
| 06/11 | Asked about newer optimization methods, why hashes were not saved, and how many runs the benchmark docs require | Make sure the repo's procedure is followed | Hashes saved for every build; run counts follow the docs |
| 06/11 | Set a rule: when a model "seems" unable to run on 4 GB, try several methods before concluding | GR00T had been labeled "cannot run" too early | GR00T N1.5: from out of VRAM to 0.63 s |
| 06/11 | Asked whether N1.7 can run on the T2000 | Check whether the method works on a larger model | N1.7: 0.30 s, 30/30, after finding that the original was spilling out of VRAM |
| 07/11 | After the repo owner's review: make the commit small and keep only my own signature | The old commit had 79 files and was hard to merge | Now 1 commit, 16 files, signed only by me |
| 07/11 | **Rejected the proposal to re-measure** the whole default table | The table followed the right order: measure defaults first, find failures, then run the separate experiments | The proposal was dropped. Saved a full re-run of 8 models |
| 07/11 | Compared the report with the existing RTX reports; kept the template layout without forcing the line count down | The report must match the repo standard, not be short at any cost | The GR00T part moved to its own file, `t2000-gr00t.md` |

Final numbers on the T2000, default settings:

| Model | Mean latency (ms) | Peak VRAM (MiB) |
|---|--:|--:|
| Octo-Small | 24.6 | 627 |
| ACT (640×480) | 48.7 | 213 |
| TurboVLA | 105.5 | 923 |
| VLA-Adapter | 514.3 | 2865 |
| SmolVLA | 535.0 | 2023 |
| Evo-1 | 1117.0 | 1547 |
| BitVLA | 1287.5 | 1255 |
| VLA-JEPA | 2345.8 | 3925 |
| GR00T N1.5 | out of VRAM, handled separately | |

Results of the separate GR00T experiments:

| Version | Per prediction | VRAM | Robot success (LIBERO-Object) |
|---|--:|--:|--:|
| N1.5 original (forced F16) | 4.3 s | 3.6 GB | 27/30 |
| N1.5 optimized | 0.63 s | 2.5 GB | 28/30 |
| N1.7 original | 12.5 s | spills into RAM | not usable |
| N1.7 optimized | 0.30 s | 2.9 GB | 30/30 |

30 episodes are enough to say the fast version is not worse. They are not enough to say which version is better.

## Plan

1. Rebase the `t2000-cuda-benchmark` branch onto the new `main`. There are conflicts in `docs/benchmark/README.md` and `src/models/gr00tn1d5.cpp`. Then re-check on the T2000 and send the link to the repo owner for review.
2. Evaluate the optimized GR00T on the other 3 LIBERO suites (Spatial, Goal, Long), with more episodes.
3. Re-run N1.7 with the official tokenizer (requires accepting the NVIDIA license).
4. Fix the 2 ModelOpt reports. They still say CUDA cannot run on the T2000, which is no longer true.
5. Keep learning the same way (guess, run it myself, compare):
   - Finish the pruning exercise.
   - Then learn distillation, CUDA Graphs, flash attention, tensor cores and dp4a.
   - Goal: explain every technique used on the T2000 without outside help.
6. Apply one technique I learned (for example quantization) to SmolVLA myself, and measure it on LIBERO with the two-level quality gate.
7. Proposals to discuss:
   - Run 2–3 simulators in parallel to shorten evaluation time.
   - Let the model predict the next action chunk while the robot is still executing the current one, to hide latency.
