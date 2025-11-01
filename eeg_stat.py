# step1_acquisition.py
# EEG acquisition parameters and 1-s epoching buffer @128 Hz

FS = 128                   # samples per second
EPOCH_SEC = 1              # seconds per epoch
SAMPLES_PER_EPOCH = FS * EPOCH_SEC
N_CHANNELS = 8             # set to your headset's channel count
DTYPE = "float32"          # incoming sample dtype

# Optional capture knobs (used later in preprocessing)
HPF_HZ = 0.5
LPF_HZ = 45.0
NOTCH_HZ = 60.0
REF_MODE = "average"
DEVICE_NAME = "OpenBCI Cyton"

from collections import deque
import numpy as np
import time
from typing import Optional, Iterator, Dict

class EpochBuffer:
    """Accumulates streaming samples into fixed 1-s epochs."""
    def __init__(self, n_channels: int, samples_per_epoch: int):
        self.n_channels = n_channels
        self.samples_per_epoch = samples_per_epoch
        self.buf = deque(maxlen=samples_per_epoch)
        self.t0: Optional[float] = None

    def push(self, sample: np.ndarray) -> Optional[Dict]:
        if self.t0 is None:
            self.t0 = time.time()
        # sample expected shape: (n_channels,)
        self.buf.append(sample.astype(DTYPE, copy=False))
        if len(self.buf) == self.samples_per_epoch:
            epoch = np.vstack(self.buf)  # shape: [SAMPLES_PER_EPOCH, N_CHANNELS]
            t_start, t_end = self.t0, time.time()
            self.buf.clear()
            self.t0 = None
            return {
                "data": epoch,
                "fs": FS,
                "channels": self.n_channels,
                "t_start": t_start,
                "t_end": t_end,
            }
        return None

def eeg_stream_simulator(n_channels: int = N_CHANNELS, fs: int = FS) -> Iterator[np.ndarray]:
    """Simulated stream: yields one multi-channel sample each 1/fs seconds."""
    t0 = time.time()
    k = 0
    while True:
        tvec = k / fs
        alpha = np.sin(2 * np.pi * 10 * tvec) * 20e-6  # 10 Hz, 20 µV
        noise = np.random.normal(0, 5e-6, size=n_channels)
        yield (alpha + noise).astype(DTYPE)
        k += 1
        target = t0 + k / fs
        sleep = target - time.time()
        if sleep > 0:
            time.sleep(sleep)

# Example usage: collect one epoch (replace simulator with real device read later)
if __name__ == "__main__":
    buf = EpochBuffer(N_CHANNELS, SAMPLES_PER_EPOCH)
    stream = eeg_stream_simulator()
    while True:
        sample = np.asarray(next(stream))
        packet = buf.push(sample)
        if packet is not None:
            print("Got epoch:", packet["data"].shape, "fs=", packet["fs"])
            break

# step2_raw_spectra.py
import numpy as np

def calculate_raw_spectra(epoch: np.ndarray, fs: int):
    """
    epoch: array shape [SAMPLES_PER_EPOCH, N_CHANNELS]
    returns: freqs (Hz), spectra (complex) shape [N_FREQS, N_CHANNELS]
    """
    # Detrend by removing per-channel mean
    x = epoch - epoch.mean(axis=0, keepdims=True)

    # Hann window (same length as samples), column-broadcast to channels
    win = np.hanning(x.shape[0])[:, None]
    xw = x * win

    # Amplitude normalization so a 1.0 sine has ~0.5 amplitude in rFFT bins (window-compensated)
    scale = 2.0 / win.sum()

    # Real FFT along time axis; complex spectra per channel
    Xk = np.fft.rfft(xw, axis=0) * scale

    # Frequency axis in Hz
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    return freqs, Xk

if __name__ == "__main__":
    # quick self-test with a 10 Hz tone in ch0
    fs = 128
    t = np.arange(fs) / fs
    ch0 = np.sin(2*np.pi*10*t)
    noise = 0.01*np.random.randn(fs, 7)
    epoch = np.column_stack([ch0] + [noise[:, i] for i in range(noise.shape[1])])
    freqs, Xk = calculate_raw_spectra(epoch, fs)
    pk_idx = np.argmax(np.abs(Xk[:, 0]))
    print("Peak ~", freqs[pk_idx], "Hz")

# step3_power_spectral_density.py
import numpy as np

def calculate_psd(epoch: np.ndarray, fs: int):
    """
    epoch: shape [SAMPLES_PER_EPOCH, N_CHANNELS]
    returns: freqs (Hz), PSD (V^2/Hz) shape [N_FREQS, N_CHANNELS]
    """
    x = epoch - epoch.mean(axis=0, keepdims=True)

    N = x.shape[0]
    w = np.hanning(N)[:, None]
    xw = x * w

    # Window power normalization (Welch scaling)
    U = (w**2).mean()  # = (1/N) * sum(w^2)

    X = np.fft.rfft(xw, axis=0)
    Sxx = (np.abs(X) ** 2) / (fs * N * U)  # one-segment periodogram, V^2/Hz

    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    return freqs, Sxx

if __name__ == "__main__":
    fs = 128
    t = np.arange(fs) / fs
    ch0 = 20e-6*np.sin(2*np.pi*10*t)
    noise = 5e-6*np.random.randn(fs, 7)
    epoch = np.column_stack([ch0] + [noise[:, i] for i in range(noise.shape[1])])
    f, G = calculate_psd(epoch, fs)
    print("PSD shape:", G.shape, "peak ch0 @", f[np.argmax(G[:,0])], "Hz")

#stpe4_cross_spectra
import numpy as np

def calculate_cross_spectrum(epoch_x: np.ndarray, epoch_y: np.ndarray, fs: int):
    """
    Inputs: epoch_x, epoch_y shape [SAMPLES_PER_EPOCH], same length; returns freqs (Hz), Gxy (complex V^2/Hz).
    """
    x = epoch_x - epoch_x.mean(); y = epoch_y - epoch_y.mean()
    N = x.shape[0]; w = np.hanning(N)
    U = (w**2).mean()
    X = np.fft.rfft(x * w); Y = np.fft.rfft(y * w)
    Gxy = (X * np.conj(Y)) / (fs * N * U)  # cross spectrum (complex)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, Gxy

# Example:
# ch0, ch1 = epoch[:,0], epoch[:,1]; f, Gxy = calculate_cross_spectrum(ch0, ch1, fs)



#step5_smooth_spectral_data

import numpy as np
from collections import deque
from typing import Optional, Tuple

class SpectralSmoother:
    """
    Smooths per-epoch PSDs Gx, Gy and cross-spectrum Gxy to produce Ĝx, Ĝy, Ĝxy.
    Use mode='boxcar' with window M (simple running mean) or mode='ewma' with alpha (exponential).
    """
    def __init__(self, n_freqs: int, mode: str = "boxcar", M: int = 8, alpha: float = 0.2):
        assert mode in ("boxcar", "ewma")
        self.mode, self.M, self.alpha = mode, M, alpha
        self.buf_Gx, self.buf_Gy, self.buf_Gxy = deque(maxlen=M), deque(maxlen=M), deque(maxlen=M)
        self.ewma_Gx: Optional[np.ndarray] = None
        self.ewma_Gy: Optional[np.ndarray] = None
        self.ewma_Gxy: Optional[np.ndarray] = None
        self.n_freqs = n_freqs

    def update(self, Gx: np.ndarray, Gy: np.ndarray, Gxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "boxcar":
            self.buf_Gx.append(Gx); self.buf_Gy.append(Gy); self.buf_Gxy.append(Gxy)
            Ĝx = np.mean(self.buf_Gx, axis=0)
            Ĝy = np.mean(self.buf_Gy, axis=0)
            Ĝxy = np.mean(self.buf_Gxy, axis=0)
        else:  # EWMA
            a = self.alpha
            if self.ewma_Gx is None:
                self.ewma_Gx, self.ewma_Gy, self.ewma_Gxy = Gx.copy(), Gy.copy(), Gxy.copy()
            else:
                self.ewma_Gx = (1-a)*self.ewma_Gx + a*Gx
                self.ewma_Gy = (1-a)*self.ewma_Gy + a*Gy
                self.ewma_Gxy = (1-a)*self.ewma_Gxy + a*Gxy
            Ĝx, Ĝy, Ĝxy = self.ewma_Gx, self.ewma_Gy, self.ewma_Gxy
        return Ĝx, Ĝy, Ĝxy

# Usage:
# freqs, Gx = calculate_psd(epoch[:,0], fs)  # from step 3 (per-channel) or stack per channel
# _, Gy = calculate_psd(epoch[:,1], fs)
# _, Gxy = calculate_cross_spectrum(epoch[:,0], epoch[:,1], fs)
# smoother = SpectralSmoother(n_freqs=len(freqs), mode="boxcar", M=8)  # or mode="ewma", alpha=0.2
# Gx_s, Gy_s, Gxy_s = smoother.update(Gx, Gy, Gxy)

#step6_calculate_coherence

import numpy as np

def calculate_phase_shift(Gxy_hat: np.ndarray):
    """
    Computes phase difference δ(f) between two EEG channels from the complex cross-spectrum.
    Returns phase in radians and degrees.
    """
    phase_rad = np.angle(Gxy_hat)
    phase_deg = np.degrees(phase_rad)
    return phase_rad, phase_deg

#step7_calculate_phase_shift

import numpy as np

def calculate_phase_shift(Gxy_hat: np.ndarray):
    """
    Computes phase difference δ(f) between two EEG channels from the complex cross-spectrum.
    Returns phase in radians and degrees.
    """
    phase_rad = np.angle(Gxy_hat)
    phase_deg = np.degrees(phase_rad)
    return phase_rad, phase_deg