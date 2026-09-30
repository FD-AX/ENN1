import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from hw1.equations import (PARAMETERS, bytes_moved, energy, flops, latency, latency_terms, memory,
                           memory_naive, memory_runtime_estimate)


def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ['S', 'B']:
            row[key] = int(row[key])
        for key in ['oom', 'is_validation']:
            row[key] = row[key].lower() == 'true'
        for key in ['latency_s', 'memory_bytes', 'energy_j', 'baseline_extra_bytes']:
            row[key] = float(row[key]) if row[key] else np.nan
    return rows


def metrics(real, predicted):
    ape = np.abs(predicted - real) / real * 100
    return {'median_ape_pct': float(np.median(ape)), 'mean_ape_pct': float(np.mean(ape)),
            'rmse': float(np.sqrt(np.mean((predicted - real)**2))), 'n': len(real)}


def fit_latency(s, b, measured):
    names = ['launch_overhead', 'bandwidth', 'throughput']

    def unpack(log_values):
        return dict(zip(names, np.exp(log_values)))

    def residual(log_values):
        return np.log(latency(s, b, unpack(log_values)) / measured)

    fits = []
    for bandwidth, throughput in [(1e11, 1e12), (5e11, 1e13), (1e10, 1e14), (1e12, 1e11)]:
        fit = least_squares(residual, np.log([1e-5, bandwidth, throughput]),
                            bounds=(np.log([1e-8, 1e6, 1e6]), np.log([1e-1, 1e14, 1e16])),
                            max_nfev=3000)
        fits.append(fit)
    fit = min(fits, key=lambda result: result.cost)
    if not fit.success:
        raise RuntimeError('Latency fitting failed: ' + fit.message)
    # Linearized std of log(parameter) ~ relative uncertainty. A parameter whose branch
    # never wins has a near-zero Jacobian column and a huge (or undefined) std.
    jac = fit.jac
    sigma2 = np.sum(fit.fun**2) / max(len(measured) - 3, 1)
    log_std = np.sqrt(np.clip(np.diag(np.linalg.pinv(jac.T @ jac)) * sigma2, 0, None))
    diagnostics = {'log_residual_rmse': float(np.sqrt(np.mean(fit.fun**2))),
                   'jacobian_singular_values': np.linalg.svd(jac, compute_uv=False).tolist(),
                   'jacobian_column_norms': dict(zip(names, np.linalg.norm(jac, axis=0).tolist())),
                   'relative_std': dict(zip(names, log_std.tolist())),
                   'warning': 'launch_overhead is one aggregate constant (not N_launch * tau); '
                              'values are effective, not hardware peaks.'}
    return unpack(fit.x), diagnostics


def fit_energy(s, b, measured, theta_latency):
    # Predicted (not measured) latency, so energy() stays a function of S and B only.
    columns = [latency(s, b, theta_latency), flops(s, b), bytes_moved(s, b)]
    scales = np.array([np.median(c) for c in columns])
    design = np.column_stack(columns) / scales
    # Relative residuals so small configurations are not ignored.
    fit = least_squares(lambda p: (design @ p - measured) / measured,
                        np.full(3, np.median(measured) / 3), bounds=(0, np.inf),
                        x_scale='jac', max_nfev=3000)
    if not fit.success:
        raise RuntimeError('Energy fitting failed: ' + fit.message)
    p0, e_flop, e_byte = fit.x / scales
    theta = {'latency': theta_latency, 'P0_w': float(p0), 'e_flop': float(e_flop), 'e_byte': float(e_byte)}
    names = ['P0_w', 'e_flop', 'e_byte']
    normalized = design / np.linalg.norm(design, axis=0)
    return theta, {'normalized_design_condition': float(np.linalg.cond(normalized)),
                   'log_predictor_correlation': np.corrcoef(np.log(np.column_stack(columns)).T).tolist(),
                   'predictor_order': ['predicted_latency', 'flops', 'bytes_moved'],
                   'coefficients_at_zero': [n for n, v in zip(names, fit.x) if v <= 1e-12],
                   'warning': 'Predicted latency, FLOPs and logical traffic are strongly correlated; '
                              'individual coefficients are not well identified.'}


def memory_plot(s, b, real, baseline, validation, out):
    mib = 2**20
    runtime = memory_runtime_estimate(s, b, baseline)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for pred, label, color in [(memory(s, b), 'Analytical memory() (17 BS^2)', 'C0'),
                               (runtime, 'Runtime-aware (+ measured baseline)', 'C1')]:
        for mask, marker in [(~validation, 'o'), (validation, 'x')]:
            axes[0].scatter(real[mask] / mib, pred[mask] / mib, marker=marker, s=22, alpha=.7,
                            color=color, label=label if marker == 'o' else None)
    limits = [real.min() / mib * .5, real.max() / mib * 1.5]
    axes[0].plot(limits, limits, 'k--', label='Exact agreement')
    axes[0].set(xscale='log', yscale='log', xlabel='Measured max_memory_allocated (MiB)',
                ylabel='Predicted peak (MiB)', title='o calibration grid, x validation')
    extra = np.median(baseline)
    for size, color in zip([32, 128, 256, 512], ['C0', 'C1', 'C2', 'C3']):
        mask = s == size
        if not np.any(mask):
            continue
        bs = np.unique(b[mask])
        axes[1].plot(bs, memory_naive(size, bs) / mib, color=color, linestyle=':', linewidth=1)
        axes[1].plot(bs, memory(size, bs) / mib, color=color, label=f'S={size}')
        axes[1].plot(bs, memory_runtime_estimate(size, bs, extra) / mib, color=color,
                     linestyle='--', linewidth=1)
        axes[1].scatter(b[mask], real[mask] / mib, color=color, s=18)
    axes[1].set(xscale='log', yscale='log', xlabel='Batch B (images)', ylabel='Peak memory (MiB)')
    axes[1].set_title(
        'dots: measured; dotted 13, solid 17 BS^2; dashed +baseline', fontsize=9)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'memory.png', dpi=160)
    plt.close(fig)


def comparison_plot(s, b, real, predicted, validation, ylabel, name, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for mask, marker, label in [(~validation, 'o', 'Calibration'), (validation, 'x', 'Validation')]:
        axes[0].scatter(real[mask], predicted[mask], marker=marker, s=22, label=label, alpha=.7)
    limits = [min(real.min(), predicted.min()) * .8, max(real.max(), predicted.max()) * 1.2]
    axes[0].plot(limits, limits, 'k--', label='Exact agreement')
    axes[0].set(xscale='log', yscale='log', xlabel='Measured ' + ylabel,
                ylabel='Predicted ' + ylabel)
    for size, color in zip([32, 128, 256, 512], ['C0', 'C1', 'C2', 'C3']):
        mask = s == size
        if not np.any(mask):
            continue
        order = np.argsort(b[mask])
        axes[1].plot(b[mask][order], predicted[mask][order], color=color, label=f'S={size} prediction')
        axes[1].scatter(b[mask], real[mask], color=color, s=18)
    axes[1].set(xscale='log', yscale='log', xlabel='Batch B (images)', ylabel=ylabel)
    axes[1].set_title('Dots: measurements; lines: equations')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=160)
    plt.close(fig)


def make_plots(rows, theta, folder, environment):
    out = folder / 'figures'
    out.mkdir(exist_ok=True)
    good = [r for r in rows if not r['oom']]
    s = np.array([r['S'] for r in good])
    b = np.array([r['B'] for r in good])
    validation = np.array([r['is_validation'] for r in good])
    predictions = {'memory_bytes': memory(s, b), 'latency_s': latency(s, b, theta['latency'])}
    if theta['energy'] is not None:
        predictions['energy_j'] = energy(s, b, theta['energy'])
    settings = [('memory_bytes', 'memory (MiB)', 2**20, 'memory'),
                ('latency_s', 'latency (ms)', .001, 'latency'), ('energy_j', 'energy (J)', 1, 'energy')]
    measured_latency = np.array([r['latency_s'] for r in good])
    report = {}
    baseline = np.array([r['baseline_extra_bytes'] for r in good])
    memory_real = np.array([r['memory_bytes'] for r in good])
    memory_plot(s, b, memory_real, baseline, validation, out)
    fig, error_axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4), squeeze=False)
    for ax, (key, label, scale, name) in zip(error_axes[0], settings):
        real = np.array([r[key] for r in good])
        valid = np.isfinite(real) & (real > 0)
        pred = predictions[key]
        if name != 'memory':
            comparison_plot(s[valid], b[valid], real[valid] / scale, pred[valid] / scale,
                            validation[valid], label, name, out)
        report[name] = {}
        for mask, marker, split in [(~validation & valid, 'o', 'calibration'),
                                    (validation & valid, 'x', 'validation')]:
            if np.any(mask):
                report[name][split] = metrics(real[mask], pred[mask])
                ax.scatter(flops(s[mask], b[mask]) / 1e9,
                           (pred[mask] / real[mask] - 1) * 100, marker=marker, s=20, label=split)
        ax.axhline(0, color='k', linewidth=.8)
        ax.set(xscale='log', xlabel='Arithmetic work (GFLOPs)', ylabel='Signed prediction error (%)', title=name)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'validation_error.png', dpi=160)
    plt.close(fig)

    runtime = memory_runtime_estimate(s, b, baseline)
    report['memory_runtime_estimate'] = {
        split: metrics(memory_real[mask], runtime[mask])
        for mask, split in [(~validation, 'calibration'), (validation, 'validation')] if np.any(mask)}
    report['baseline_extra_bytes'] = {'min': float(baseline.min()), 'median': float(np.median(baseline)),
                                      'max': float(baseline.max()),
                                      'note': 'Persistent runtime allocations, measured, not attributed.'}

    diagnostic_path = folder / 'profiler_flops.json'
    if diagnostic_path.exists():
        diagnostics = [d for d in json.loads(diagnostic_path.read_text()) if d.get('profiler_flops', 0) > 0]
        if diagnostics:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            batches = np.arange(1, 257)
            sizes = np.unique([r['S'] for r in rows])
            for i, size in enumerate(sizes):
                ax.plot(batches, flops(size, batches) / 1e9, color=plt.cm.viridis(i / len(sizes)), linewidth=.9)
                # Neighbouring sizes (256/272, 352/384/400) almost coincide: stagger the labels.
                ax.annotate(f'S={size}', (batches[-1], flops(size, batches[-1]) / 1e9), fontsize=6.5,
                            xytext=([3, 25, 47][i % 3], 0), textcoords='offset points', va='center')
            ax.scatter([d['B'] for d in diagnostics], [d['profiler_flops'] / 1e9 for d in diagnostics],
                       color='k', marker='x', s=70, zorder=3,
                       label='torch.profiler, ' + ', '.join(f"({d['S']}, {d['B']})" for d in diagnostics))
            ax.set(xscale='log', yscale='log', xlabel='Batch B (images)', ylabel='Conv/Linear work (GFLOPs)',
                   xlim=(0.9, 420), title='Lines: 17712 B S^2 + 313344 B for every S in the grid')
            ax.legend(fontsize=8, loc='upper left')
            fig.tight_layout()
            fig.savefig(out / 'flops.png', dpi=160, bbox_inches='tight')
            plt.close(fig)
            report['flops_profiler'] = metrics(np.array([d['profiler_flops'] for d in diagnostics]),
                                              np.array([d['analytical_flops'] for d in diagnostics]))

    all_s = np.array([r['S'] for r in rows])
    all_b = np.array([r['B'] for r in rows])
    oom = np.array([r['oom'] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ss, bb = np.meshgrid(np.arange(32, 513, 16), np.arange(1, 257))
    surface = ax.pcolormesh(ss, bb, memory(ss, bb) / 2**30, shading='auto', cmap='Blues')
    fig.colorbar(surface, ax=ax, label='Predicted peak memory (GiB)')
    ax.scatter(all_s[~oom], all_b[~oom], facecolors='none', edgecolors='k', s=24, label='Measured: fits')
    ax.scatter(all_s[oom], all_b[oom], color='red', marker='x', s=40, label='Measured: OOM')
    total = environment.get('total_memory_bytes')
    title = 'Analytical memory() and observed outcomes' + ('' if oom.any() else ' (no OOM observed)')
    if total and memory(ss, bb).min() < total < memory(ss, bb).max():
        ax.contour(ss, bb, memory(ss, bb), levels=[total], colors='red', linestyles='--')
    elif total:
        # Where memory() would reach the card: solve 4P + 68 B S^2 = total.
        reach = (total - 4 * PARAMETERS) / 68
        title += (f'\nmemory() reaches the {total / 2**30:.1f} GiB of this GPU at B*S^2 = {reach:.2g} '
                  f'(e.g. S=512, B={reach / 512**2:.0f}); grid maximum is B*S^2 = {(all_b * all_s**2).max():.2g}')
    ax.set(xlabel='Image side S (pixels)', ylabel='Batch B (images)', yscale='log', title=title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'oom_boundary.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    report['oom'] = {'observed': int(oom.sum()), 'configurations': len(rows),
                     'predicted_over_total_capacity': int(np.sum(memory(all_s, all_b) > total)) if total else None,
                     'note': 'Total-capacity comparison excludes external usage and backend temporary storage.'}
    if not oom.any():
        report['oom']['note'] += (' No prescribed grid point ran out of memory: '
                                  'the empirical OOM boundary is outside the tested grid.')

    terms = latency_terms(s, b, theta['latency'])
    regimes = np.argmax(terms, axis=0)
    work = b * s**2
    order = np.argsort(work, kind='stable')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for term, label in zip(terms, ['Launch overhead t0', 'Bytes moved / BW', 'FLOPs / R']):
        axes[0].plot(work[order], term[order] * 1000, label=label, linewidth=1.2)
    axes[0].plot(work[order], predictions['latency_s'][order] * 1000, 'k--', label='Total prediction')
    axes[0].scatter(work, measured_latency * 1000, s=10, color='k', label='Measured')
    axes[0].set(xscale='log', yscale='log', xlabel='Workload B*S^2 (pixels)', ylabel='Latency (ms)',
                title='Latency terms: bytes and FLOPs curves nearly coincide')
    axes[0].legend(fontsize=7)
    # Network-level intensity against the fitted ridge: explains why no memory region appears.
    ss, bb = np.meshgrid(np.arange(32, 513, 16), [1, 256])
    grid_work = (bb * ss**2).ravel()
    grid_order = np.argsort(grid_work)
    intensity = (flops(ss, bb) / bytes_moved(ss, bb)).ravel()
    axes[1].plot(grid_work[grid_order], intensity[grid_order], color='C4', label='FLOPs / bytes_moved')
    ridge = theta['latency']['throughput'] / theta['latency']['bandwidth']
    axes[1].axhline(ridge, color='k', linestyle='--', label=f'fitted ridge R/BW = {ridge:.0f}')
    axes[1].set(xscale='log', yscale='log', xlabel='Workload B*S^2 (pixels)',
                ylabel='Arithmetic intensity (FLOPs/byte)', title='Network intensity is almost constant')
    axes[1].legend(fontsize=8)
    for i, name in enumerate(['launch', 'memory', 'compute']):
        mask = regimes == i
        axes[2].scatter(s[mask], b[mask], label=f'{name} term largest ({mask.sum()})', s=30)
    axes[2].scatter(all_s[oom], all_b[oom], color='k', marker='x', label='Measured OOM')
    axes[2].set(xlabel='Image side S (pixels)', ylabel='Batch B (images)', yscale='log',
                title='Largest predicted term per configuration')
    axes[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'regimes.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    report['predicted_regimes'] = {name: int(np.sum(regimes == i))
                                   for i, name in enumerate(['launch', 'memory', 'compute'])}
    report['network_intensity_flops_per_byte'] = {'min': float(intensity.min()), 'max': float(intensity.max()),
                                                  'fitted_ridge': float(ridge)}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    rows = read_rows(args.results / 'measurements.csv')
    calibration = [r for r in rows if not r['oom'] and not r['is_validation']
                   and np.isfinite(r['latency_s']) and r['latency_s'] > 0]
    if len(calibration) < 4:
        raise ValueError('At least four feasible calibration points are required; smoke runs are not enough')
    s = np.array([r['S'] for r in calibration])
    b = np.array([r['B'] for r in calibration])
    theta_latency, latency_diagnostics = fit_latency(s, b, np.array([r['latency_s'] for r in calibration]))
    energy_rows = [r for r in calibration if np.isfinite(r['energy_j']) and r['energy_j'] > 0]
    theta_energy, energy_diagnostics = None, {'status': 'Insufficient energy measurements'}
    if len(energy_rows) >= 4:
        theta_energy, energy_diagnostics = fit_energy(np.array([r['S'] for r in energy_rows]),
                                                     np.array([r['B'] for r in energy_rows]),
                                                     np.array([r['energy_j'] for r in energy_rows]),
                                                     theta_latency)
    theta = {'latency': theta_latency, 'energy': theta_energy,
             'latency_diagnostics': latency_diagnostics, 'energy_diagnostics': energy_diagnostics,
             'fit_split': 'Only non-OOM, non-validation rows',
             'units': {'launch_overhead': 's', 'bandwidth': 'bytes/s', 'throughput': 'FLOPs/s',
                       'P0_w': 'W', 'e_flop': 'J/FLOP', 'e_byte': 'J/byte'}}
    (args.results / 'theta.json').write_text(json.dumps(theta, indent=2), encoding='utf-8')
    environment_path = args.results / 'environment.json'
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    report = make_plots(rows, theta, args.results, environment)
    (args.results / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
