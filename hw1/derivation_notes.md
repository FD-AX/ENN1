# Derivation notes

Working notes for the handwritten derivation (`hw1_handwritten.pdf` is written by hand from these).
FP32 everywhere, 4 bytes per element. Input `[B, 3, S, S]`, S divisible by 16.

## 1. Output resolution

    H_out = floor((H + 2p - k) / s) + 1,   p = k // 2

- stride 1, odd k: `H_out = H`
- stride 2, even H: `H_out = floor((H - 1) / 2) + 1 = H / 2`

| layer | out channels | out side |
|---|---|---|
| conv1 7x7 s2 | 32 | S/2 |
| maxpool 3x3 s2 p1 | 32 | S/4 |
| conv2 5x5 | 64 | S/4 |
| conv3 3x3 s2 | 128 | S/8 |
| conv4 1x1 | 256 | S/8 |
| conv5 3x3 s2 | 256 | S/16 |
| conv6 1x1 | 512 | S/16 |
| GAP | 512 | 1 |

S divisible by 16 makes every stride-2 layer see an even side.

## 2. FLOPs

Convention: 1 MAC = 2 FLOPs. Only Conv and Linear MACs are counted.
Not counted: Linear bias additions (356 B), GAP additions/division, ReLU and MaxPool comparisons.
This is the same convention torch.profiler(with_flops=True) uses for conv/addmm.

Conv: each output element needs `Cin * k * k` MACs:

    F_conv = 2 * B * H_out * W_out * Cout * Cin * k^2

Linear: `F_lin = 2 * B * in * out`.

| layer | expression | coefficient |
|---|---|---|
| conv1 | 2 (S/2)^2 * 32 * 3 * 49 | 2352 B S^2 |
| conv2 | 2 (S/4)^2 * 64 * 32 * 25 | 6400 B S^2 |
| conv3 | 2 (S/8)^2 * 128 * 64 * 9 | 2304 B S^2 |
| conv4 | 2 (S/8)^2 * 256 * 128 | 1024 B S^2 |
| conv5 | 2 (S/16)^2 * 256 * 256 * 9 | 4608 B S^2 |
| conv6 | 2 (S/16)^2 * 512 * 256 | 1024 B S^2 |
| fc1 | 2 * 512 * 256 | 262144 B |
| fc2 | 2 * 256 * 100 | 51200 B |

    FLOPs(S, B) = 17712 B S^2 + 313344 B

Matches the profiler exactly at (32,1), (128,8), (224,16).

### Extended operation count (not FLOPs)

- ReLU: one comparison per output. Conv outputs are 8 + 4 + 2 + 4 + 1 + 2 = 21 B S^2, plus 256 B after fc1.
- MaxPool 3x3: 8 comparisons per output, 2 B S^2 outputs, so 16 B S^2 (padding at the border is ignored).

    Ops_ext = FLOPs + 37 B S^2 + 256 B

## 3. Parameters

| layer | count |
|---|---|
| conv1 | 3*32*49 = 4704 |
| conv2 | 32*64*25 = 51200 |
| conv3 | 64*128*9 = 73728 |
| conv4 | 128*256 = 32768 |
| conv5 | 256*256*9 = 589824 |
| conv6 | 256*512 = 131072 |
| fc1 | 512*256 + 256 = 131328 |
| fc2 | 256*100 + 100 = 25700 |

    P = 1 040 324,   4P = 4 161 296 bytes

## 4. Peak memory

Quantity: `torch.cuda.max_memory_allocated()` during one forward, input already allocated.

Assumptions:
- parameters and input x are alive the whole time (the caller holds x);
- during an operator its input and output coexist, and the input is freed right after;
- ReLU is inplace, so no new tensor;
- no cuDNN workspace and no persistent runtime state.

Live elements per operator, in units of B S^2 (x = 3):

| operator | live tensors | total |
|---|---|---|
| conv1 | x + out(8) | 11 |
| maxpool | x + conv1(8) + out(2) | **13** |
| conv2 | x + pool(2) + out(4) | 9 |
| conv3 | x + 4 + 2 | 9 |
| conv4 | x + 2 + 4 | 9 |
| conv5 | x + 4 + 1 | 8 |
| conv6 | x + 1 + 2 | 6 |
| head | O(B) | - |

Naive tensor-lifetime model: `M_naive = 4 (P + 13 B S^2)`.

In PyTorch CUDA `max_pool2d` calls `max_pool2d_with_indices` even under inference_mode,
so it also allocates int64 indices with the output's shape: 2 B S^2 elements * 8 bytes = 16 B S^2 bytes,
the same as 4 B S^2 FP32 elements. They are freed right after the pool call.
Seen in the profiler memory trace (allocations inside max_pool2d_with_indices: 8 B S^2 and 16 B S^2 bytes).
This comes from the implementation, not from the CNN itself.

    Memory(S, B) = 4P + 4 * 17 B S^2 = 4 161 296 + 68 B S^2   [bytes]

Runtime-aware estimate (diagnostic only): `Memory(S, B) + baseline_extra`, where `baseline_extra`
is measured before the forward (`memory_allocated() - params - x`). It is about 8.5-10 MB and not attributed.

## 5. Bytes moved (logical traffic)

Idealized: every tensor an operator touches is read or written once, weights read once.
Cache reuse, im2col/implicit-GEMM traffic, fusion, workspace and MaxPool indices are not modeled.
Units of B S^2 elements:

| op | read | write | sum |
|---|---|---|---|
| conv1 | 3 | 8 | 11 |
| relu1 | 8 | 8 | 16 |
| maxpool | 8 | 2 | 10 |
| conv2 + relu | 2 + 4 | 4 + 4 | 14 |
| conv3 + relu | 4 + 2 | 2 + 2 | 10 |
| conv4 + relu | 2 + 4 | 4 + 4 | 14 |
| conv5 + relu | 4 + 1 | 1 + 1 | 7 |
| conv6 + relu | 1 + 2 | 2 + 2 | 7 |
| GAP | 2 | - | 2 |

Total 91 B S^2. The head (per image): GAP write 512, fc1 512 + 256, ReLU 256 + 256, fc2 256 + 100 = 2148 B.

    Bytes(S, B) = 4 (P + 91 B S^2 + 2148 B)

(Counting the MaxPool index write would give 95 instead of 91; not used.)

## 6. Latency

    T(S, B) = t0 + max( Bytes(S, B) / BW , FLOPs(S, B) / R )

- t0: aggregate fixed overhead (launches, Python, dispatcher). Only the sum is identifiable,
  so there is no N_launch * tau split.
- BW, R: effective bandwidth [bytes/s] and throughput [FLOPs/s], not hardware peaks.
- Fit in log space (relative error) on calibration rows only.

Regimes: t0 largest -> launch-bound; Bytes/BW largest -> memory-bound; FLOPs/R largest -> compute-bound.

Limitation: for large B S^2

    FLOPs / Bytes -> 17712 / (4 * 91) ≈ 49 FLOP/byte,

which is the same for every (S, B). The two terms scale almost in proportion, so the max() does not
switch branches and BW is weakly identifiable. Intensity is lower only at small B S^2 (weights dominate),
where t0 is already the largest term.

## 7. Energy

    E(S, B) = P0 * T(S, B) + e_f * FLOPs(S, B) + e_b * Bytes(S, B),   P0, e_f, e_b >= 0

- T is the predicted latency from section 6, not a measurement.
- P0 [W]: effective board power while the forward runs (includes static power), not the idle datasheet value.
- e_f [J/FLOP], e_b [J/byte]: dynamic energy per unit work.
- T, FLOPs and Bytes are strongly correlated here, so the three coefficients are not well separated.
