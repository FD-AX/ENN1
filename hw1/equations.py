import numpy as np


PARAMETERS = 1_040_324  # Includes both Linear biases.


def _inputs(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float),
                               np.asarray(batch, dtype=float))
    if np.any(~np.isfinite(s) | ~np.isfinite(b)):
        raise ValueError("S and B must be finite")
    if np.any((s < 16) | (s % 16 != 0) | (b < 1) | (b % 1 != 0)):
        raise ValueError("S must be a positive multiple of 16; B a positive integer")
    return s, b


def flops(image_size, batch):
    # Conv/Linear MACs only, at two FLOPs per MAC (no bias, GAP, comparisons).
    s, b = _inputs(image_size, batch)
    return b * (17_712 * s**2 + 313_344)


def extended_operation_count(image_size, batch):
    # flops() plus one comparison per ReLU output and 8 per 3x3 MaxPool output.
    s, b = _inputs(image_size, batch)
    return flops(s, b) + 37 * b * s**2 + 256 * b


def memory_naive(image_size, batch):
    # Tensor lifetimes only: at pooling, input x + conv1 output + pool output = 13 BS².
    s, b = _inputs(image_size, batch)
    return 4 * (PARAMETERS + 13 * b * s**2)


def memory(image_size, batch):
    # PyTorch CUDA max_pool2d runs max_pool2d_with_indices even in inference, so
    # 2 BS² int64 indices (= 4 BS² FP32-equivalents) coexist with the 13 BS² above.
    # Seen in the profiler trace; persistent runtime allocations are not included.
    s, b = _inputs(image_size, batch)
    return 4 * PARAMETERS + 68 * b * s**2


def memory_runtime_estimate(image_size, batch, baseline_extra_bytes):
    # Diagnostic only: baseline_extra_bytes is measured, not derived from the architecture.
    return memory(image_size, batch) + np.asarray(baseline_extra_bytes, dtype=float)


def memory_mib(image_size, batch):
    return memory(image_size, batch) / 2**20


def bytes_moved(image_size, batch):
    # One read/write per activation, weights and biases read once per forward.
    # MaxPool indices are deliberately not counted (would make 91 -> 95).
    s, b = _inputs(image_size, batch)
    return 4 * (PARAMETERS + 91 * b * s**2 + 2148 * b)


def latency_terms(image_size, batch, theta):
    f = flops(image_size, batch)
    return np.stack([np.full_like(f, theta['launch_overhead']),
                     bytes_moved(image_size, batch) / theta['bandwidth'],
                     f / theta['throughput']], axis=0)


def latency(image_size, batch, theta):
    # Both FLOPs and bytes are ~proportional to B S^2, so network intensity is almost
    # constant and the max() rarely switches branch; see README.
    overhead, traffic, compute = latency_terms(image_size, batch, theta)
    return overhead + np.maximum(traffic, compute)


def energy(image_size, batch, theta_energy):
    # Static/board power acts over the predicted time; the rest scales with work.
    return (theta_energy['P0_w'] * latency(image_size, batch, theta_energy['latency'])
            + theta_energy['e_flop'] * flops(image_size, batch)
            + theta_energy['e_byte'] * bytes_moved(image_size, batch))
