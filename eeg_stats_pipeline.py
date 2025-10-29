"EEG coherence analysis pipeline without third-party dependencies."
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class SpectralResults:
    frequencies: List[float]
    raw_x: List[complex]
    raw_y: List[complex]
    psd_x: List[float]
    psd_y: List[float]
    cross_spectrum: List[complex]
    smoothed_psd_x: List[float]
    smoothed_psd_y: List[float]
    smoothed_cross_spectrum: List[complex]
    coherence: List[float]
    phase_shift: List[float]


def _rfft(values: Sequence[float]) -> List[complex]:
    n = len(values)
    output: List[complex] = []
    for k in range(n // 2 + 1):
        total = 0j
        for idx, sample in enumerate(values):
            angle = -2.0 * math.pi * k * idx / n
            total += sample * complex(math.cos(angle), math.sin(angle))
        output.append(total)
    return output


def calculate_raw_spectra(
    x: Sequence[float], y: Sequence[float], sampling_rate: float
) -> Tuple[List[float], List[complex], List[complex]]:
    if len(x) != len(y):
        raise ValueError("Input signals must have identical lengths.")
    frequencies = [sampling_rate * k / len(x) for k in range(len(x) // 2 + 1)]
    return frequencies, _rfft(x), _rfft(y)


def calculate_power_spectral_density(
    raw_spectrum: Sequence[complex], sampling_rate: float, n_samples: int
) -> List[float]:
    scale = 2.0 / (sampling_rate * n_samples)
    psd = [scale * (abs(value) ** 2) for value in raw_spectrum]
    psd[0] /= 2.0
    if n_samples % 2 == 0:
        psd[-1] /= 2.0
    return psd


def calculate_cross_spectrum(
    raw_x: Sequence[complex], raw_y: Sequence[complex], sampling_rate: float, n_samples: int
) -> List[complex]:
    scale = 2.0 / (sampling_rate * n_samples)
    cross = [scale * a * complex(b.real, -b.imag) for a, b in zip(raw_x, raw_y)]
    cross[0] /= 2.0
    if n_samples % 2 == 0:
        cross[-1] /= 2.0
    return cross


def _moving_average(values: Sequence[complex], window: int) -> List[complex]:
    if window <= 1:
        return list(values)
    left_pad = [values[0]] * (window // 2)
    right_pad = [values[-1]] * (window - 1 - window // 2)
    padded: List[complex] = left_pad + list(values) + right_pad
    result: List[complex] = []
    for idx in range(len(values)):
        window_slice = padded[idx : idx + window]
        total = sum(window_slice)
        result.append(total / window)
    return result


def smooth_spectrum_real(values: Sequence[float], window: int) -> List[float]:
    smoothed = _moving_average([complex(v, 0.0) for v in values], window)
    return [value.real for value in smoothed]


def smooth_spectrum_complex(values: Sequence[complex], window: int) -> List[complex]:
    return _moving_average(values, window)


def calculate_coherence(
    cross_spectrum: Sequence[complex], psd_x: Sequence[float], psd_y: Sequence[float]
) -> List[float]:
    coherence: List[float] = []
    for cross, gx, gy in zip(cross_spectrum, psd_x, psd_y):
        denominator = gx * gy
        if denominator <= 0.0:
            coherence.append(0.0)
            continue
        value = abs(cross) ** 2 / denominator
        coherence.append(max(0.0, min(1.0, value)))
    return coherence


def _unwrap(phases: Iterable[float]) -> List[float]:
    iterator = iter(phases)
    try:
        first = next(iterator)
    except StopIteration:
        return []
    unwrapped = [first]
    prev = first
    for phase in iterator:
        delta = phase - prev
        while delta > math.pi:
            phase -= 2.0 * math.pi
            delta = phase - prev
        while delta < -math.pi:
            phase += 2.0 * math.pi
            delta = phase - prev
        unwrapped.append(unwrapped[-1] + delta)
        prev = phase
    return unwrapped


def calculate_phase_shift(cross_spectrum: Sequence[complex]) -> List[float]:
    raw_phases = [cmath.phase(value) for value in cross_spectrum]
    return _unwrap(raw_phases)


def analyze_epoch(
    x: Sequence[float],
    y: Sequence[float],
    sampling_rate: float,
    smoothing_window: int = 5,
) -> SpectralResults:
    frequencies, raw_x, raw_y = calculate_raw_spectra(x, y, sampling_rate)
    n_samples = len(x)
    psd_x = calculate_power_spectral_density(raw_x, sampling_rate, n_samples)
    psd_y = calculate_power_spectral_density(raw_y, sampling_rate, n_samples)
    cross = calculate_cross_spectrum(raw_x, raw_y, sampling_rate, n_samples)

    smoothed_psd_x = smooth_spectrum_real(psd_x, smoothing_window)
    smoothed_psd_y = smooth_spectrum_real(psd_y, smoothing_window)
    smoothed_cross = smooth_spectrum_complex(cross, smoothing_window)

    coherence = calculate_coherence(smoothed_cross, smoothed_psd_x, smoothed_psd_y)
    phase_shift = calculate_phase_shift(smoothed_cross)

    return SpectralResults(
        frequencies=frequencies,
        raw_x=raw_x,
        raw_y=raw_y,
        psd_x=psd_x,
        psd_y=psd_y,
        cross_spectrum=cross,
        smoothed_psd_x=smoothed_psd_x,
        smoothed_psd_y=smoothed_psd_y,
        smoothed_cross_spectrum=smoothed_cross,
        coherence=coherence,
        phase_shift=phase_shift,
    )


def _simulate_epoch(
    duration: float = 1.0, sampling_rate: float = 128.0
) -> Tuple[List[float], List[float]]:
    n_samples = max(1, int(round(duration * sampling_rate)))
    times = [idx / sampling_rate for idx in range(n_samples)]
    signal_a = [math.sin(2 * math.pi * 10 * t) + 0.3 * math.sin(2 * math.pi * 20 * t) for t in times]
    signal_b = [
        0.8 * math.sin(2 * math.pi * 10 * t + math.radians(30)) + 0.2 * math.sin(2 * math.pi * 25 * t)
        for t in times
    ]
    noise_a = _uniform_noise(n_samples, seed=42)
    noise_b = _uniform_noise(n_samples, seed=1729)
    return [a + n for a, n in zip(signal_a, noise_a)], [b + n for b, n in zip(signal_b, noise_b)]


def _uniform_noise(n: int, seed: int) -> List[float]:
    rng = seed
    noise: List[float] = []
    for _ in range(n):
        rng = (1664525 * rng + 1013904223) % (2 ** 32)
        noise.append(((rng / (2 ** 32)) - 0.5) * 0.1)
    return noise


def main() -> None:
    sampling_rate = 128.0
    x, y = _simulate_epoch(duration=1.0, sampling_rate=sampling_rate)
    results = analyze_epoch(x, y, sampling_rate, smoothing_window=7)

    header = (
        "Frequency (Hz)",
        "|X_k|",
        "|Y_k|",
        "G_xx",
        "G_yy",
        "|G_xy|",
        "Coherence",
        "Phase Shift (deg)",
    )
    print("\t".join(header))

    for idx, freq in enumerate(results.frequencies):
        print(
            f"{freq:8.2f}\t"
            f"{abs(results.raw_x[idx]):8.4f}\t"
            f"{abs(results.raw_y[idx]):8.4f}\t"
            f"{results.smoothed_psd_x[idx]:8.4f}\t"
            f"{results.smoothed_psd_y[idx]:8.4f}\t"
            f"{abs(results.smoothed_cross_spectrum[idx]):8.4f}\t"
            f"{results.coherence[idx]:8.4f}\t"
            f"{math.degrees(results.phase_shift[idx]):8.2f}"
        )


if __name__ == "__main__":
    main()