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

All equation functions take scalars or NumPy arrays and broadcast, e.g. `flops(S[:, None], B[None, :])`.
They do not run PyTorch.

## Reproduce

From the repository root, with an NVIDIA GPU (Kaggle/Colab T4 or P100, or a local card with <= 16 GB):

Local environment: `pip install -r requirements.txt` (pins a CUDA 12.8 torch build).
Kaggle/Colab: use the preinstalled torch; install only scipy/matplotlib if they are missing.

    python -m hw1.test_equations
    python -m hw1.measure --traces       # ~15-20 min on the full grid
    python -m hw1.calibrate

`measure.py` refuses to run without CUDA (there is no CPU fallback) and refuses to overwrite an
existing `measurements.csv`. `python -m hw1.measure --smoke --traces --output hw1/results/smoke`
runs three points as a quick check.

Flags for every run: `cudnn.benchmark = False`, `cudnn.allow_tf32 = False`,
`cuda.matmul.allow_tf32 = False`, `model.eval()`, `torch.inference_mode()`, FP32.

## Environment

Filled in automatically in `results/environment.json` (GPU name, driver, torch/CUDA/cuDNN versions,
sampled grid values). Summary of the official run:

- GPU: _TODO after the run_
- torch / CUDA / cuDNN: _TODO_

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
   allocations (about 8.5-10 MB in the smoke run, slightly different between configurations) that
   the architecture does not explain. It may be library workspace, but we have not shown that,
   so it stays unattributed.

cuDNN workspace is not in either model. In the smoke traces the conv workspaces were tiny, but this
depends on the algorithm cuDNN picks. In a local test sweep, a few configurations had a transient
peak well above the runtime-aware estimate (tens of MiB); these can be looked at with `export_trace`.

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

_Not filled in yet: the full grid has to be measured on the target GPU first._

After `calibrate.py`:
`results/metrics.json` has median/mean APE and RMSE separately for calibration and validation rows,
and `results/figures/` has:

- `memory.png`: measured peak vs `memory()` and the runtime-aware estimate (naive 13 B S^2 dotted)
- `latency.png`, `energy.png`: predicted vs measured and per-S curves over B
- `validation_error.png`: signed error vs GFLOPs, calibration vs validation
- `flops.png`: formula vs profiler FLOPs
- `oom_boundary.png`: predicted memory surface with observed fits/OOMs
- `regimes.png`: the three latency terms over B*S^2, network arithmetic intensity against the fitted
  ridge R/BW, and which term is largest at each (S, B)

OOM: with `memory()` the largest grid point (S=512, B=256) needs about 4.6 GB. On a 16 GB card
it is possible that no prescribed point runs out of memory. If so, the report says the OOM
boundary lies outside the tested grid. No OOM is forced.

## Where the equations break

**Latency: no separate memory-bound region.** This is the main failure of the latency equation.
In a local test sweep (RTX 5070 Ti) the fit labeled 39 configurations launch, 0 memory, 93 compute.
The reason is in the formula, not the GPU:

- FLOPs = 17712 B S^2 + 313344 B and Bytes = 4 (P + 91 B S^2 + 2148 B). Past small sizes both are
  dominated by their B S^2 term.
- So network-level intensity FLOPs / Bytes goes to 17712 / 364 ≈ 49 FLOP/byte and stays there
  (left and middle panels of `regimes.png`).
- The bytes term and the FLOPs term then grow almost in proportion. Their curves nearly coincide,
  and the max() cannot switch from one to the other as the workload grows.
- Intensity drops below the fitted ridge R/BW only at small B S^2, where weights dominate traffic.
  In that region t0 is already larger than both terms, so the memory branch is never the largest one.
- BW is therefore weakly identifiable from these measurements (relative std ~0.26 vs ~0.04 for R in
  the test sweep). The fitted ridge sits right under the asymptotic intensity, and the BW value
  mostly adjusts how the two curves meet.

The model separates the launch-dominated small workloads from the rest reasonably well. It cannot
resolve a memory-bound region before the compute-dominated one. Measuring more configurations
does not help, because for moderate and large workloads the network intensity approaches the same
asymptotic value. Individual layers do differ
(conv1, maxpool and ReLU have low intensity, conv5 high), so a per-layer roofline with the same three
shared parameters would be the natural next step. That is left as optional future work and is not
part of this submission.

**Other deviations** (to be checked against the real run):

- **Small S and B.** Kernel launch and framework overhead are expected to dominate the forward, so latency
  should be approximately flat in S and B. Energy may have a larger relative error there, because each
  forward is short and mostly board power.
- **Large S and B.** One effective R averages conv kernels with very different efficiency (7x7 on
  3 channels vs 1x1 on 256), and logical bytes are only a lower bound on conv traffic. With
  `benchmark=False` cuDNN's heuristics can change kernels between shapes, which gives steps the
  smooth max() cannot follow.
- **Memory.** `memory()` leaves out the persistent baseline, so small configurations are under-predicted by
  several times, while large ones are dominated by 68 B S^2. OOM prediction is approximate
  because workspace, allocator fragmentation and other processes on the GPU are not modeled.
- **Energy.** nvidia-smi samples board power at about 100 ms and the sensor averages internally, so
  per-forward energy is an average over thousands of forwards and is noisier than latency.
