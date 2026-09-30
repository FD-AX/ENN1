import argparse
import csv
import gc
import json
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from hw1.equations import (bytes_moved, extended_operation_count, flops, memory,
                           memory_naive, memory_runtime_estimate)
from hw1.models import HomeworkCNN


BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RESULTS = Path(__file__).resolve().parent / 'results'


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Enable a GPU in Kaggle/Colab or install CUDA PyTorch.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_grid(seed=42):
    rng = np.random.default_rng(seed)
    extra_s = sorted(map(int, rng.choice([s for s in range(32, 513, 16)
                                         if s not in BASE_S], 4, replace=False)))
    extra_b = sorted(map(int, rng.choice([b for b in range(1, 257)
                                         if b not in BASE_B], 3, replace=False)))
    grid = [(s, b, s not in BASE_S or b not in BASE_B)
            for s, b in product(sorted(BASE_S + extra_s), sorted(BASE_B + extra_b))]
    return grid, extra_s, extra_b


@torch.inference_mode()
def measure_latency(model, x, warmup=10, repeats=30):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(repeats)]
    times = []
    for start, end in pairs:
        start.record()
        model(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return float(np.median(times))


@torch.inference_mode()
def measure_memory(model, x):
    # Called after warmup with no output referenced: only parameters, x and
    # persistent runtime state are allocated when the peak counter is reset.
    torch.cuda.synchronize()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    baseline_extra = torch.cuda.memory_allocated() - parameter_bytes - x.numel() * x.element_size()
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return int(peak), int(baseline_extra)


def smi_command(gpu_id, fields):
    return ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=' + fields,
            '--format=csv,noheader,nounits']


def process_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if platform.system() == 'Windows' else {}


@torch.inference_mode()
def measure_energy(model, x, gpu_id, duration=3.0, latency_s=0.001):
    # One persistent process avoids launching nvidia-smi for every sample.
    samples, errors = [], []
    ready = threading.Event()
    command = smi_command(gpu_id, 'power.draw') + ['--loop-ms=100']
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, **process_options())
    except OSError as exc:
        raise RuntimeError('Energy measurement requires nvidia-smi with power.draw support') from exc

    def read_power():
        for line in proc.stdout:
            try:
                watts = float(line.strip())
                if not np.isfinite(watts) or watts <= 0:
                    raise ValueError('nonpositive/nonfinite power')
                samples.append((time.perf_counter(), watts))
            except ValueError:
                errors.append(line.strip())
            ready.set()

    reader = threading.Thread(target=read_power, daemon=True)
    reader.start()
    chunk = max(1, min(100, int(0.02 / max(latency_s, 1e-6))))

    def run_chunk():
        for _ in range(chunk):
            model(x)
        torch.cuda.synchronize()

    try:
        if not ready.wait(timeout=10) or errors or not samples:
            raise RuntimeError('nvidia-smi power unavailable: ' + '; '.join(errors))
        # Sustained warmup also reduces contamination from the sensor's rolling average.
        until = time.perf_counter() + 1.2
        while time.perf_counter() < until:
            run_chunk()
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < duration:
            run_chunk()
            count += chunk
        end = time.perf_counter()
        # Keep the GPU busy until a power sample brackets the integration endpoint.
        deadline = end + 5
        while samples[-1][0] < end and time.perf_counter() < deadline:
            run_chunk()
        data = np.asarray(samples, dtype=float)
        inside = (data[:, 0] > start) & (data[:, 0] < end)
        if errors or data[-1, 0] < end or np.count_nonzero(inside) < 3:
            raise RuntimeError('Insufficient valid power samples: ' + '; '.join(errors))
        t = np.r_[start, data[inside, 0], end]
        p = np.interp(t, data[:, 0], data[:, 1])
        joules = float(np.trapezoid(p, t))
        details = {'forwards': count, 'duration_s': end - start,
                   'mean_power_w': joules / (end - start),
                   'samples': [[float(a - start), float(b)] for a, b in data]}
        return joules / count, details
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        reader.join(timeout=5)
        proc.stdout.close()


@torch.inference_mode()
def export_trace(model, S, B, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(B, 3, S, S, device='cuda', dtype=torch.float32)
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
                 with_flops=True, record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            with record_function('FWD'):
                model(x)
            torch.cuda.synchronize()
            prof.step()
    # kineto opens the path as a narrow string, which fails for non-ASCII
    # directories on Windows; write to a temp file and move it into place.
    with tempfile.TemporaryDirectory() as tmp:
        prof.export_chrome_trace(str(Path(tmp) / path.name))
        shutil.move(str(Path(tmp) / path.name), str(path))
    # Only supported Conv/Linear ops; biases and comparisons are excluded in both.
    measured = sum(e.flops for e in prof.key_averages())
    return {'S': S, 'B': B, 'analytical_flops': float(flops(S, B)),
            'profiler_flops': int(measured), 'trace': str(path.name),
            'maxpool_alloc_bytes': maxpool_allocations(path),
            'expected_pool_output_bytes': 8 * B * S * S,
            'expected_pool_indices_bytes': 16 * B * S * S}


def maxpool_allocations(trace_path):
    # Allocations made inside max_pool2d_with_indices: expect pool output and int64 indices.
    events = json.loads(Path(trace_path).read_text(encoding='utf-8'))['traceEvents']
    pools = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'aten::max_pool2d_with_indices']
    return [int(e['args']['Bytes']) for e in events
            if e.get('name') == '[memory]' and e['args'].get('Bytes', 0) > 0
            and any(p['ts'] <= e['ts'] <= p['ts'] + p['dur'] for p in pools)]


def environment(gpu_id, args, extra_s, extra_b):
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.check_output(smi_command(gpu_id, 'uuid,name,driver_version'),
                                  text=True, timeout=10, **process_options()).strip()
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': props.name, 'total_memory_bytes': props.total_memory,
            'nvidia_smi': smi, 'power_gpu_id': gpu_id,
            'precision': 'FP32', 'cudnn_benchmark': False, 'tf32': False,
            'seed': args.seed, 'extra_S': extra_s, 'extra_B': extra_b,
            'warmup': args.warmup, 'repeats': args.repeats,
            'energy_seconds': args.energy_seconds, 'smoke': args.smoke,
            'energy_skipped': args.skip_energy,
            'latency_method': 'median synchronized CUDA event interval per forward'}


def main():
    parser = argparse.ArgumentParser(description='CUDA-only FP32 inference measurements')
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--energy-seconds', type=float, default=3)
    parser.add_argument('--skip-energy', action='store_true', help='Explicit partial run, records blank energy')
    parser.add_argument('--smoke', action='store_true', help='Three points only; use a separate output directory')
    parser.add_argument('--traces', action='store_true')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3 or args.energy_seconds < 1:
        parser.error('warmup >= 1, repeats >= 3, energy-seconds >= 1 required')
    configure_cuda()
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    grid, extra_s, extra_b = make_grid(args.seed)
    if args.smoke:
        grid = [(32, 1, False), (128, 8, False), (224, 16, False)]
    # UUID makes nvidia-smi select the same physical GPU even with CUDA_VISIBLE_DEVICES.
    gpu_id = str(torch.cuda.get_device_properties(0).uuid)
    if not gpu_id.startswith('GPU-'):
        gpu_id = 'GPU-' + gpu_id
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'measurements.csv'
    if csv_path.exists():
        raise FileExistsError(f'{csv_path} exists; choose a new --output directory')
    info = environment(gpu_id, args, extra_s, extra_b)
    (args.output / 'environment.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"GPU: {info['gpu']}; extra S={extra_s}, extra B={extra_b}; {len(grid)} points", flush=True)
    model = HomeworkCNN().cuda().eval()
    rows, power_records = [], []
    fields = ['S', 'B', 'latency_s', 'memory_bytes', 'energy_j', 'oom', 'is_validation',
              'predicted_flops', 'extended_ops', 'predicted_memory_bytes',
              'predicted_memory_naive_bytes', 'baseline_extra_bytes',
              'predicted_runtime_memory_bytes', 'predicted_bytes_moved', 'free_memory_before_bytes', 'oom_stage', 'energy_status',
              'energy_forwards', 'energy_duration_s', 'mean_power_w']
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (s, b, validation) in enumerate(grid, 1):
            gc.collect()
            torch.cuda.empty_cache()
            row = dict.fromkeys(fields, '')
            row.update(S=s, B=b, oom=False, is_validation=validation,
                       predicted_flops=float(flops(s, b)), extended_ops=float(extended_operation_count(s, b)),
                       predicted_memory_bytes=float(memory(s, b)),
                       predicted_memory_naive_bytes=float(memory_naive(s, b)),
                       predicted_bytes_moved=float(bytes_moved(s, b)),
                       free_memory_before_bytes=torch.cuda.mem_get_info()[0],
                       energy_status='skipped' if args.skip_energy else 'pending')
            x, stage = None, 'input'
            energy_error = None
            try:
                x = torch.randn(b, 3, s, s, device='cuda', dtype=torch.float32)
                stage = 'latency'
                row['latency_s'] = measure_latency(model, x, args.warmup, args.repeats)
                stage = 'memory'
                row['memory_bytes'], row['baseline_extra_bytes'] = measure_memory(model, x)
                row['predicted_runtime_memory_bytes'] = float(
                    memory_runtime_estimate(s, b, row['baseline_extra_bytes']))
                if not args.skip_energy:
                    stage = 'energy'
                    row['energy_j'], details = measure_energy(model, x, gpu_id,
                                                             args.energy_seconds, row['latency_s'])
                    row.update(energy_status='ok', energy_forwards=details['forwards'],
                               energy_duration_s=details['duration_s'], mean_power_w=details['mean_power_w'])
                    power_records.append({'S': s, 'B': b, **details})
            except torch.cuda.OutOfMemoryError:
                row.update(oom=True, oom_stage=stage, latency_s='', memory_bytes='', energy_j='',
                           baseline_extra_bytes='', predicted_runtime_memory_bytes='',
                           energy_status='oom')
            except RuntimeError as exc:
                if stage != 'energy':
                    raise
                row['energy_status'] = 'error: ' + str(exc)
                energy_error = str(exc)
            finally:
                del x
                gc.collect()
                torch.cuda.empty_cache()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            (args.output / 'power_samples.json').write_text(json.dumps(power_records), encoding='utf-8')
            status = 'OOM' if row['oom'] else f"{row['latency_s'] * 1000:.3f} ms"
            print(f'[{i}/{len(grid)}] S={s} B={b}: {status}', flush=True)
            if energy_error:
                raise RuntimeError('Energy stage failed; partial CSV saved. ' + energy_error)
    if args.traces:
        feasible = sorted([r for r in rows if not r['oom']], key=lambda r: r['predicted_flops'])
        diagnostics = []
        if feasible:
            for row in [feasible[0], feasible[len(feasible) // 2], feasible[-1]]:
                s, b = row['S'], row['B']
                try:
                    result = export_trace(model, s, b, args.output / 'traces' / f'S{s}_B{b}.json')
                    diagnostics.append(result)
                    print('Profiler:', result, flush=True)
                except torch.cuda.OutOfMemoryError:
                    diagnostics.append({'S': s, 'B': b, 'status': 'profiler OOM'})
                    gc.collect()
                    torch.cuda.empty_cache()
        (args.output / 'profiler_flops.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
