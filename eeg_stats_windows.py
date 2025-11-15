# step1_acquisition.py
# EEG acquisition parameters and 1-s epoching buffer @128 Hz (Windows + cross-platform)

from collections import deque
import numpy as np
import time, os, glob
from typing import Optional, Iterator, Dict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ------------------ Config ------------------
FS = 128                   # samples per second
EPOCH_SEC = 1              # seconds per epoch
SAMPLES_PER_EPOCH = FS * EPOCH_SEC
N_CHANNELS = 8             # set to your headset's channel count (Cyton = up to 8 EEG)
DTYPE = "float32"          # incoming sample dtype

# Optional capture knobs (used later in preprocessing)
HPF_HZ = 0.5
LPF_HZ = 45.0
NOTCH_HZ = 60.0
REF_MODE = "average"
DEVICE_NAME = "OpenBCI Cyton"

# BrainFlow board & port (leave SERIAL_PORT="" to auto-detect)
BOARD_ID = BoardIds.CYTON_BOARD.value
SERIAL_PORT = "COM7"  # e.g., "COM5" on Windows, or leave "" to auto-detect
# --------------------------------------------


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


def resolve_serial_port(requested: str) -> str:
    """Cross-platform serial-port locator. Returns a COM device on Windows or /dev/* on Unix."""
    if requested:
        return requested

    if os.name == "nt":  # Windows
        try:
            from serial.tools import list_ports  # type: ignore
            ports = list(list_ports.comports())
            if not ports:
                raise RuntimeError("No COM ports found. Plug in the OpenBCI dongle.")
            keywords = ("openbci", "cp210", "silicon labs", "wch", "ch340", "usb-serial")
            scored = []
            for p in ports:
                desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
                score = sum(k in desc for k in keywords)
                scored.append((score, p.device))
            scored.sort(reverse=True)
            return scored[0][1]  # e.g., "COM5"
        except Exception:
            # Fallback guesses; BrainFlow will complain if incorrect
            for guess in ("COM3", "COM4", "COM5", "COM6", "COM7", "COM8"):
                return guess

    # macOS / Linux
    candidates = (
        glob.glob("/dev/cu.usbserial*") +
        glob.glob("/dev/cu.usbmodem*") +
        glob.glob("/dev/cu.SLAB_USBtoUART*") +
        glob.glob("/dev/ttyUSB*") +
        glob.glob("/dev/ttyACM*")
    )
    if not candidates:
        raise RuntimeError("No serial device found. Plug in the OpenBCI dongle.")
    return sorted(candidates)[0]


def eeg_stream_cyton(n_channels: int = N_CHANNELS, fs: int = FS) -> Iterator[np.ndarray]:
    """
    Live stream from OpenBCI Cyton via BrainFlow.
    Yields one multi-channel sample each call, in VOLTS, shape = (n_channels,).
    """
    BoardShim.enable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = resolve_serial_port(SERIAL_PORT)

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()

    try:
        # Verify board sampling rate & EEG channel indices
        fs_board = BoardShim.get_sampling_rate(BOARD_ID)
        eeg_chans = BoardShim.get_eeg_channels(BOARD_ID)
        if fs_board != fs:
            # Not fatal — we’ll just time using actual board fs
            fs = fs_board

        # Use first N_CHANNELS EEG channels in board order
        use_idx = eeg_chans[:min(n_channels, len(eeg_chans))]

        # BrainFlow returns EEG in microvolts (µV) for Cyton; convert to Volts
        UV_TO_V = 1e-6

        # Simple per-sample yield using small chunks
        # Pull slightly more than 1 sample and iterate through them
        chunk = max(1, int(fs // 16))  # ~60 Hz fetch cadence
        while True:
            data = board.get_current_board_data(chunk)  # shape: (rows, cols)
            if data.size == 0:
                # No data yet
                time.sleep(0.005)
                continue

            # data rows are board channels; columns are samples
            # Extract EEG rows, convert to Volts, then iterate per sample column
            eeg_block_uv = data[use_idx, :]            # µV
            eeg_block_v = (eeg_block_uv * UV_TO_V).astype(np.float32)  # Volts
            for c in range(eeg_block_v.shape[1]):
                sample = eeg_block_v[:, c]
                # If fewer than N_CHANNELS available, pad with zeros (rare)
                if sample.shape[0] < n_channels:
                    sample = np.pad(sample, (0, n_channels - sample.shape[0]))
                yield sample
            # Light pacing to avoid busy-loop
            time.sleep(0.0)
    finally:
        try:
            board.stop_stream()
        except:
            pass
        try:
            board.release_session()
        except:
            pass


def eeg_stream_simulator(n_channels: int = N_CHANNELS, fs: int = FS) -> Iterator[np.ndarray]:
    """Simulated stream: yields one multi-channel sample each 1/fs seconds (in Volts)."""
    t0 = time.time()
    k = 0
    while True:
        tvec = k / fs
        alpha = np.sin(2 * np.pi * 10 * tvec) * 20e-6  # 10 Hz, 20 µV → Volts
        noise = np.random.normal(0, 5e-6, size=n_channels)
        yield (alpha + noise).astype(DTYPE)
        k += 1
        target = t0 + k / fs
        sleep = target - time.time()
        if sleep > 0:
            time.sleep(sleep)


# Example usage: collect one epoch
if __name__ == "__main__":
    # Choose your source:
    USE_CYTON = True  # set False to force simulator (no hardware)

    buf = EpochBuffer(N_CHANNELS, SAMPLES_PER_EPOCH)
    if USE_CYTON:
        print("[acq] Starting Cyton stream…")
        stream = eeg_stream_cyton(N_CHANNELS, FS)
    else:
        print("[acq] Starting simulator…")
        stream = eeg_stream_simulator(N_CHANNELS, FS)

    while True:
        sample = np.asarray(next(stream))
        packet = buf.push(sample)
        if packet is not None:
            print("Got epoch:", packet["data"].shape, "fs=", packet["fs"])
            # Do whatever you want with packet["data"] here (save, process, queue, etc.)
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