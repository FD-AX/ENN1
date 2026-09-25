import numpy as np
import torch

from hw1.equations import (PARAMETERS, bytes_moved, extended_operation_count, flops, latency,
                           memory, memory_naive, memory_runtime_estimate)
from hw1.models import HomeworkCNN

# (Cin, Cout, k, stride) with output side as a fraction of S
CONVS = [(3, 32, 7, 2, 1 / 2), (32, 64, 5, 1, 1 / 4), (64, 128, 3, 2, 1 / 8),
         (128, 256, 1, 1, 1 / 8), (256, 256, 3, 2, 1 / 16), (256, 512, 1, 1, 1 / 16)]


def layer_by_layer_flops(S, B):
    total = sum(2 * B * (S * f)**2 * cout * cin * k * k for cin, cout, k, _, f in CONVS)
    return total + 2 * B * (512 * 256 + 256 * 100)


def naive_peak_elements(S, B):
    # Live tensors per operator (x is held by the caller for the whole forward).
    x = 3 * S * S * B
    conv1, pool = 8 * S * S * B, 2 * S * S * B
    c2, c3, c4, c5, c6 = [cout * (S * f)**2 * B for _, cout, _, _, f in CONVS[1:]]
    stages = [x + conv1, x + conv1 + pool, x + pool + c2, x + c2 + c3,
              x + c3 + c4, x + c4 + c5, x + c5 + c6]
    return max(stages), stages


def test_all():
    assert sum(p.numel() for p in HomeworkCNN().parameters()) == PARAMETERS
    for S in [32, 224, 512]:
        for B in [1, 7, 256]:
            assert flops(S, B) == layer_by_layer_flops(S, B)
            peak, stages = naive_peak_elements(S, B)
            assert peak == 13 * B * S * S and stages.index(peak) == 1
            assert memory_naive(S, B) == 4 * (PARAMETERS + peak)
            assert memory(S, B) == memory_naive(S, B) + 16 * B * S * S
    S = np.array([32, 64, 128])[:, None]
    B = np.array([1, 8, 32])[None, :]
    for f in [flops, extended_operation_count, memory, memory_naive, bytes_moved]:
        assert f(S, B).shape == (3, 3)
    assert memory_runtime_estimate(S, B, 1e7).shape == (3, 3)
    theta = {'launch_overhead': 1e-4, 'bandwidth': 3e11, 'throughput': 1e13}
    assert latency(S, B, theta).shape == (3, 3)
    with torch.inference_mode():
        assert HomeworkCNN().eval()(torch.randn(2, 3, 224, 224)).shape == (2, 100)
    for bad in [(30, 1), (32, 0), (32, 1.5)]:
        try:
            flops(*bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} accepted')


if __name__ == '__main__':
    test_all()
    print('analytical checks passed')
