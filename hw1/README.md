# HW1: cost model of a small CNN

Closed-form FLOPs, peak memory, latency and energy of one FP32 inference forward pass as functions
of image side S and batch B, checked against measurements on one GPU. Derivations are in
[derivation_notes.md](derivation_notes.md). `hw1_handwritten.pdf` must be completed separately
from derivation_notes.md.

## Network

Input `[B, 3, S, S]`, S divisible by 16, 100 classes. Every conv has `padding = k // 2`, `bias=False`,
and is followed by `ReLU(inplace=True)`.

    conv7x7 s2 3->32 -> maxpool3x3 s2 p1 -> conv5x5 32->64 -> conv3x3 s2 64->128
    -> conv1x1 128->256 -> conv3x3 s2 256->256 -> conv1x1 256->512
    -> GAP -> Linear 512->256 -> ReLU -> Linear 256->100

There is no BatchNorm: the assignment text mentions it as an allowed layer type, but the layer
table for this model does not contain it. Linear layers keep their default bias.
P = 1 040 324 parameters. Weights are random: only inference cost is measured, not accuracy.

## Files

| file | contents |
|---|---|
| `models.py` | `HomeworkCNN`; `python -m hw1.models` checks `[2,3,224,224] -> [2,100]` |
| `equations.py` | `flops`, `memory`, `latency`, `energy` plus `bytes_moved`, `extended_operation_count`, `memory_naive`, `memory_runtime_estimate` |
| `measure.py` | CUDA-only sweep over the 132-point grid, writes `results/measurements.csv` |
| `calibrate.py` | fits theta on calibration rows, writes `theta.json`, `metrics.json`, `figures/*.png` |
| `test_equations.py` | CPU checks: layer-by-layer FLOPs, parameter count, peak stage, broadcasting |
| `derivation_notes.md` | formulas to copy into the handwritten PDF |
| `results/` | `measurements.csv`, `theta.json`, `metrics.json`, `environment.json`, `profiler_flops.json`, `figures/*.png` from the run described below |

All equation functions take scalars or NumPy arrays and broadcast, e.g. `flops(S[:, None], B[None, :])`.
They do not run PyTorch.

## Reproduce

From the repository root, with an NVIDIA GPU (Kaggle/Colab T4 or P100, or a local card with <= 16 GB):

Local environment: `pip install -r requirements.txt` (pins a CUDA 12.8 torch build).
Kaggle/Colab: use the preinstalled torch; install only scipy/matplotlib if they are missing.

    python -m hw1.test_equations
    python -m hw1.measure --traces       # ~12 min on the RTX 5070 Ti, longer on a T4
    python -m hw1.calibrate

The committed `results/` come from the local RTX 5070 Ti described below.

`measure.py` refuses to run without CUDA (there is no CPU fallback) and refuses to overwrite an
existing `measurements.csv`. `python -m hw1.measure --smoke --traces --output hw1/results/smoke`
runs three points as a quick check.

Flags for every run: `cudnn.benchmark = False`, `cudnn.allow_tf32 = False`,
`cuda.matmul.allow_tf32 = False`, `model.eval()`, `torch.inference_mode()`, FP32.

## Environment

Filled in automatically in `results/environment.json` (GPU name, driver, torch/CUDA/cuDNN versions,
sampled grid values). Summary of the official run:

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB (17 094 475 776 bytes), driver 591.86, Windows 11.
  The assignment allows a local NVIDIA GPU with <= 16 GB.
- Python 3.11.9, torch 2.10.0+cu128, CUDA 12.8, cuDNN 9.10.2 (91002), NumPy 2.4.6.
- Run on 2026-09-30 (UTC), seed 42, warmup 10, 30 timing repeats, 3 s energy windows.

## Measurement grid

S in {32, 64, 128, 224, 256, 384, 512} plus 4 random multiples of 16 in [32, 512];
B in {1, 2, 4, ..., 256} plus 3 random non-powers of two in [1, 256]. Seed 42 gives
extra S = [80, 272, 352, 400], extra B = [29, 56, 179]. That is 11 x 12 = 132 points.

- `is_validation = False`: base S x base B (63 points), the only rows used for fitting.
- `is_validation = True`: every row with an extra S or an extra B (69 points), never used for fitting.

OOM rows are excluded from all fits.

## Methodology

- **Latency.** 10 warmup forwards, then 30 forwards each timed with its own pair of CUDA events;
  median, in seconds. The input is created before timing.
- **Memory.** After warmup and with no output tensor alive: `reset_peak_memory_stats()`, one forward,
  `max_memory_allocated()`. Parameters and the input are part of the peak. Before the forward
  `baseline_extra_bytes = memory_allocated() - param_bytes - input_bytes` is recorded as well.
- **Energy.** No pynvml. One `nvidia-smi --query-gpu=power.draw --loop-ms=100` process runs next to
  a loop of forwards (1.2 s warmup, then about 3 s measured). Power samples get timestamps, are
  integrated with the trapezoid rule, and the joules are divided by the number of forwards.
  The GPU is selected by UUID, so it is the same device torch uses (this matters on Kaggle 2xT4).
  If nvidia-smi gives no power readings, the run stops instead of writing energy values.
- **OOM.** `torch.cuda.OutOfMemoryError` is caught per point, the row is written with `oom=True` and
  empty measurements, then references are dropped, `gc.collect()`, `empty_cache()`.
- **Profiler.** `--traces` exports Chrome traces (CPU+CUDA, `with_flops`, `record_shapes`,
  `profile_memory`, forward under `record_function("FWD")`) for the smallest, a middle and the
  largest feasible configuration. The traces help diagnose the latency regimes. The regime labels themselves come from the fitted
  model's terms, and the traces are not the official timing.

## Analytical models

**FLOPs.** 1 MAC = 2 FLOPs, Conv and Linear MACs only (no bias adds, no GAP, no comparisons):

    FLOPs = 17712 B S^2 + 313344 B

It matches `torch.profiler(with_flops=True)` exactly on the traced points. That shows our convention
is the same as the profiler's for conv/addmm, not that it is the only possible definition.
`extended_operation_count` adds ReLU and MaxPool comparisons (+37 B S^2 + 256 B) and is kept
separate from FLOPs.

**Memory.** Two models, on purpose:

1. `memory(S, B) = 4P + 68 B S^2` bytes, the homework equation. It is derived from tensor lifetimes
   with no fitted parameters. A plain lifetime count gives 13 B S^2 FP32 elements at the MaxPool
   (input x + conv1 output + pool output; x is held by the caller). The profiler memory trace shows
   that PyTorch's CUDA `max_pool2d` goes through `max_pool2d_with_indices` even in inference mode and
   allocates int64 indices: 2 B S^2 elements = 16 B S^2 bytes, i.e. 4 more FP32-equivalents, freed
   right after the pool. So the peak is 17 B S^2. This detail belongs to PyTorch's implementation,
   not to CNNs in general. `memory_naive` (13 B S^2) is kept for comparison.
2. `memory_runtime_estimate(S, B, baseline_extra_bytes) = memory(S, B) + baseline_extra_bytes`,
   diagnostic only. `baseline_extra_bytes` is measured, not derived. These are persistent runtime
   allocations (8.5-10.6 MB across the grid, slightly different between configurations) that
   the architecture does not explain. It may be library workspace, but we have not shown that,
   so it stays unattributed.

cuDNN workspace is not in either model. In the traces the conv workspaces are 0.5-9 KB at S=32, B=1
and S=512, B=4, but at S=512, B=256 conv2 allocates a 212 MiB workspace (222 822 400 bytes, freed
inside the call). It does not show in `max_memory_allocated()` because at conv2 only 9 B S^2
elements are alive against 17 B S^2 at the MaxPool. Six other grid points do show a transient peak
8-95 MiB above the runtime-aware estimate (see "Where the equations break").

**Bytes moved.** `4 (P + 91 B S^2 + 2148 B)`: each tensor an operator touches is read or written once,
weights once. It is an idealized logical count. Caches, the real conv kernel traffic, fusion,
workspace and the MaxPool indices are not modeled. It is only used as a predictor for latency and energy.

**Latency.** `T = t0 + max(Bytes / BW, FLOPs / R)`, with t0, BW and R positive, fitted by
`scipy.optimize.least_squares` on log residuals, calibration rows only. t0 is one aggregate fixed
overhead. Our measurements cannot split it into N_launch x per-kernel time, so we do not try. BW and R
are effective values of this model, not hardware peaks. Its main limitation is described in
"Where the equations break".

**Energy.** `E = P0 * T(S, B) + e_f * FLOPs + e_b * Bytes`, all coefficients >= 0, relative residuals.
T is the *predicted* latency, so errors in the latency model carry over into energy. P0 is the
effective board power while forwards run, not idle power from a datasheet. Predicted latency,
FLOPs and Bytes are strongly correlated for this network (a single architecture, S and B only
scale it), so the split between the three terms is weak. If a coefficient ends at zero we keep it
and report it (`coefficients_at_zero` in theta.json).

## Results

All 132 points measured, none OOM. Fits use the 63 calibration rows; errors are reported separately
for the 69 validation rows (`results/metrics.json`, `results/theta.json`).

| quantity | calibration MedAPE | validation MedAPE | validation RMSE |
|---|---|---|---|
| FLOPs vs profiler (3 traced points) | exact match | - | 0 |
| `memory()` (analytical) | 25.3 % | 7.5 % | 16.3 MiB |
| `memory()` + measured baseline (diagnostic) | 0.0 % | 0.08 % | 11.7 MiB |
| latency | 15.9 % | 11.1 % | 2.15 ms |
| energy | 4.3 % | 6.1 % | 0.54 J |

Fitted parameters:

- Latency: t0 = 277 us, BW = 3.0e11 bytes/s, R = 1.38e13 FLOP/s. Relative std from the Jacobian:
  0.08 / 0.27 / 0.04, so BW is the least determined parameter.
- Energy: P0 = 48.4 W, e_f = 4.5e-24 J/FLOP (at the zero bound), e_b = 9.4e-10 J/byte. e_f went to
  zero because with predicted latency and bytes already in the model, FLOPs carry no extra
  information (bytes and FLOPs are proportional for this network, see below). Measured mean board
  power goes from 60 W at S=32, B=1 to 300 W at the largest points.
- MaxPool indices: at all three traced points the allocations inside `max_pool2d_with_indices` are
  exactly 8 B S^2 (output) and 16 B S^2 bytes (int64 indices), e.g. 536 870 912 and 1 073 741 824
  bytes at S=512, B=256.
- Regime labels from the latency terms: 39 configurations launch, 0 memory, 93 compute. Network
  intensity over the grid: 4.1-48.6 FLOP/byte, fitted ridge R/BW = 46.
- Largest point S=512, B=256: 73.7 ms, 21.0 J, 4365 MiB measured vs 4356 MiB from `memory()`; the
  rest is the baseline. On a 16 GB card the OOM boundary lies outside the prescribed grid, so
  `oom_boundary.png` has no OOM points. Nothing was forced.

`results/figures/`:

- `memory.png`: measured peak vs `memory()` and the runtime-aware estimate (naive 13 B S^2 dotted)
- `latency.png`, `energy.png`: predicted vs measured and per-S curves over B
- `validation_error.png`: signed error vs GFLOPs, calibration vs validation
- `flops.png`: formula vs profiler FLOPs
- `oom_boundary.png`: predicted memory surface with observed fits/OOMs
- `regimes.png`: the three latency terms over B*S^2, network arithmetic intensity against the fitted
  ridge R/BW, and which term is largest at each (S, B)

## Where the equations break

**Latency: no separate memory-bound region.** This is the main failure of the latency equation.
The fit labeled 39 configurations launch, 0 memory, 93 compute. The reason is in the formula, not
the GPU:

- FLOPs = 17712 B S^2 + 313344 B and Bytes = 4 (P + 91 B S^2 + 2148 B). Past small sizes both are
  dominated by their B S^2 term.
- So network-level intensity FLOPs / Bytes goes to 17712 / 364 ≈ 49 FLOP/byte and stays there
  (left and middle panels of `regimes.png`).
- The bytes term and the FLOPs term then grow almost in proportion. Their curves nearly coincide,
  and the max() cannot switch from one to the other as the workload grows.
- Intensity drops below the fitted ridge R/BW only at small B S^2, where weights dominate traffic.
  In that region t0 is already larger than both terms, so the memory branch is never the largest one.
- BW is therefore weakly identifiable from these measurements (relative std 0.27 vs 0.04 for R).
  The fitted ridge (46 FLOP/byte) sits right under the asymptotic intensity (48.6), and the BW value
  mostly adjusts how the two curves meet.

The model separates the launch-dominated small workloads from the rest reasonably well. It cannot
resolve a memory-bound region before the compute-dominated one. Measuring more configurations
does not help, because for moderate and large workloads the network intensity approaches the same
asymptotic value. Individual layers do differ
(conv1, maxpool and ReLU have low intensity, conv5 high), so a per-layer roofline with the same three
shared parameters would be the natural next step. That is left as optional future work and is not
part of this submission.

**Other deviations** (from `validation_error.png`, `latency.png`, `memory.png`):

- **Launch-bound region.** For B S^2 below ~1e5 most points measure 0.25-0.34 ms regardless of S
  and B, which t0 = 277 us reproduces. The exceptions are the kernel-selection steps described below
  (S=32 with B >= 16, S=64 with B=16) and B=1 at S=128 and S=224 (0.39 and 0.48 ms).
- **The transition from launch-bound to compute-bound is over-predicted.** At S=128 B=4-8, S=256 B=2
  and S=352 B=1-2 the forward still takes about t0 (0.28-0.44 ms) while the model already adds the
  work term: +35 to +46 %. The additive form t0 + max(...) has the wrong shape there. CPU-side launch
  work and GPU execution overlap, so the real behaviour is closer to max(t0, work) than to a sum.
- **Kernel-selection steps.** At S=32 the measured latency jumps from ~0.25 ms (B <= 8) to
  0.54-0.60 ms (B = 16-64) and drops back to 0.46 ms at B = 128. With `benchmark=False` cuDNN's
  heuristics pick a different algorithm for these shapes, and the smooth model under-predicts them by
  about 40 %. Energy follows: -30 % at S=32, B=16.
- **Large workloads.** Errors stay within about +-20 %, but not randomly: validation points at
  100-500 GFLOPs sit 15-20 % below the prediction, and the largest point (S=512, B=256, 73.7 ms)
  is over-predicted by 17 %. One effective R averages conv kernels of very different efficiency
  (7x7 on 3 channels vs 1x1 on 256), and logical bytes are only a lower bound on conv traffic.
- **Energy** inherits the latency error and adds its own: +22 to +29 % at B=256 for S >= 224, -30 %
  at S=384/400, B=8. Board power spans 60-300 W over the grid, while the model has one fixed
  P0 = 48 W over predicted time plus one J/byte coefficient. nvidia-smi samples at ~100 ms and the
  sensor averages internally, so each value is a mean over thousands of forwards; the
  per-configuration noise is small, the bias comes from the model.
- **Memory.** `memory()` has no baseline term, so the smallest configurations are under-predicted by up
  to 67 % (12.2 MiB measured vs 4.0 MiB at S=32, B=1) while the error vanishes at large sizes (-0.2 %
  at S=512, B=256). With the measured baseline added, 120 of 132 points agree within 1 MiB. The rest
  show transient allocations that neither model has: +33 MiB (S=64, B=256), +95 MiB (S=80, B=256),
  +85 MiB (S=128, B=256), +17 MiB (S=352, B=8), +11 MiB (S=384, B=8), +8 MiB (S=400, B=8). The
  pattern (one B, several S) points at cuDNN workspace for particular conv algorithms. This is what
  would make an OOM prediction miss near the boundary; on this card the boundary is outside the grid.
