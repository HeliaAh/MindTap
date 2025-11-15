# eeg_server_windows_ready.py
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from websocket_server import WebsocketServer
import numpy as np
import time, threading, sys, glob, os
import csv
from datetime import datetime

# ===== CONFIG =====
# Use .value everywhere so BoardShim gets an int (BrainFlow expects int)
BOARD_ID     = BoardIds.CYTON_BOARD.value   # set to BoardIds.CYTON_BOARD.value for real Cyton
SERIAL_PORT  = "COM7"                               # on Windows: "COM3", "COM5", etc. Leave empty to auto-detect
WS_PORT      = 8080
SAMPLE_BATCH = 50
TICK_SEC     = 0.02
USE_N_CHANS  = 8                                # average this many EEG channels
WINDOW_SIZE  = 1000                             # samples to use for PSD/band power (≈1s if fs≈1kHz)
# ==================

def resolve_serial_port(requested: str) -> str:
    """
    Cross-platform serial port resolution:
    - If 'requested' is provided, use it directly.
    - Windows: try serial.tools.list_ports, prefer ports mentioning OpenBCI/CP210x/CH340.
    - macOS/Linux: scan /dev/cu.* like before.
    """
    if requested:
        return requested

    if os.name == "nt":  # Windows
        # Try pyserial if available
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            if not ports:
                raise RuntimeError("No COM ports found. Plug in the OpenBCI dongle.")

            # Prefer likely OpenBCI/USB-serial adapters
            priority_keywords = ("openbci", "cp210", "silicon labs", "wch", "ch340", "usb-serial")
            scored = []
            for p in ports:
                desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
                score = sum(k in desc for k in priority_keywords)
                scored.append((score, p.device))
            scored.sort(reverse=True)  # highest score first
            best = scored[0][1]
            return best
        except Exception:
            # Fallback: guess a common COM port
            for guess in ("COM3", "COM4", "COM5", "COM6"):
                return guess  # return first guess; BrainFlow will error if wrong

    # macOS / Linux
    candidates = (
        glob.glob("/dev/cu.usbserial*") +
        glob.glob("/dev/cu.usbmodem*") +
        glob.glob("/dev/cu.SLAB_USBtoUART*") +
        glob.glob("/dev/ttyUSB*") +           # Linux USB serial
        glob.glob("/dev/ttyACM*")             # Linux CDC ACM
    )
    if not candidates:
        raise RuntimeError("No serial device found. Plug in the OpenBCI dongle.")
    return sorted(candidates)[0]

def compute_band_power(freqs, psd, band):
    """Return average power in a frequency band (safe, no NaN)."""
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx_band):
        return 0.0
    band_psd = psd[idx_band]
    if np.all(np.isnan(band_psd)):
        return 0.0
    return float(np.nanmean(band_psd))

def compute_psd(signal, fs):
    """
    Compute the Power Spectral Density (PSD) of a signal using FFT.
    Returns freqs (Hz) and psd (power/Hz).
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)  # remove DC

    N = len(signal)
    if N == 0:
        return np.array([]), np.array([])

    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    psd = (1.0 / (fs * N)) * np.abs(fft_vals) ** 2
    if len(psd) > 2:
        psd[1:-1] *= 2  # keep total power consistent (mirror dropped)
    return freqs, psd

def main():
    params = BrainFlowInputParams()

    # Only set serial port for boards that need it
    if BOARD_ID == BoardIds.CYTON_BOARD.value:
        port = resolve_serial_port(SERIAL_PORT)
        print(f"[eeg_server] Using serial port: {port}")
        params.serial_port = port

    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)

    csv_file = None
    try:
        board.prepare_session()
        board.start_stream()

        server = WebsocketServer(port=WS_PORT, host="0.0.0.0")
        threading.Thread(target=server.run_forever, daemon=True).start()
        print(f"[eeg_server] WebSocket running on ws://localhost:{WS_PORT}")

        fs = BoardShim.get_sampling_rate(BOARD_ID)
        eeg_chans = board.get_eeg_channels(BOARD_ID)
        chans = eeg_chans[:min(USE_N_CHANS, len(eeg_chans))]
        print("[eeg_server] EEG channels available:", eeg_chans)
        print("[eeg_server] Using EEG chans:", chans, f"(fs={fs} Hz)")

        # CSV log with full header
        csv_filename = f"eeg_log_{int(time.time())}.csv"
        csv_file = open(csv_filename, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp",
            "feature_uV_rms_avg",
            "delta(0.5-4)",
            "theta(4-8)",
            "alpha(8-12)",
            "beta(12-30)",
            "gamma(30-40)"
        ])

        from collections import deque
        signal_buffer = deque(maxlen=int(max(1, fs) * 4))  # ~4 s rolling buffer

        # Cyton conversion: raw counts -> microvolts (gain=24)
        CONVERSION_FACTOR = 0.02235

        while True:
            data = board.get_board_data(SAMPLE_BATCH)  # (rows, cols)
            if data is None or data.size == 0:
                time.sleep(TICK_SEC)
                continue

            data = data.astype(np.float64)

            # Convert only EEG rows to uV
            eeg_rows = board.get_eeg_channels(BOARD_ID)
            data[eeg_rows, :] *= CONVERSION_FACTOR

            X = data[chans, :]  # (n_ch, n_samples)
            if X.size == 0:
                time.sleep(SAMPLE_BATCH / fs if fs else TICK_SEC)
                continue

            # Average across selected channels
            avg_signal = np.mean(X, axis=0)
            signal_buffer.extend(avg_signal)

            # Default band powers
            delta = theta = alpha = beta = gamma = 0.0

            # Compute band powers if we have enough samples
            if len(signal_buffer) >= WINDOW_SIZE and fs:
                sig = np.array(list(signal_buffer)[-WINDOW_SIZE:], dtype=np.float64)

                # Detrend to remove drift
                DataFilter.detrend(sig, DetrendOperations.LINEAR.value)

                # PSD
                freqs, psd = compute_psd(sig, fs)
                if freqs.size and psd.size:
                    delta = compute_band_power(freqs, psd, (0.5, 4))
                    theta = compute_band_power(freqs, psd, (4, 8))
                    alpha = compute_band_power(freqs, psd, (8, 12))
                    beta  = compute_band_power(freqs, psd, (12, 30))
                    gamma = compute_band_power(freqs, psd, (30, 40))

            # Channel-wise filtering & RMS (1–40 Hz; 60 Hz notch)
            rms_vals = []
            for r in range(X.shape[0]):
                v = X[r].copy()
                DataFilter.detrend(v, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(v, fs, 1.0, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                # Try BrainFlow's built-in 60 Hz removal; fallback to bandstop if older version
                try:
                    DataFilter.remove_environmental_noise(v, fs, NoiseTypes.SIXTY.value)
                except AttributeError:
                    DataFilter.perform_bandstop(v, fs, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                rms_vals.append(np.sqrt(np.mean(v ** 2)))

            feature = float(np.mean(rms_vals)) if rms_vals else 0.0

            # Broadcast to all websocket clients
            server.send_message_to_all(str(feature))

            # Log row
            timestamp = datetime.now().isoformat(timespec="milliseconds")
            csv_writer.writerow([timestamp, feature, delta, theta, alpha, beta, gamma])
            csv_file.flush()

            print(
                f"[LOG] {timestamp} | {feature:.3f} | "
                f"Δ:{delta:.3f} Θ:{theta:.3f} α:{alpha:.3f} β:{beta:.3f} γ:{gamma:.3f}"
            )

            time.sleep(TICK_SEC)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
    finally:
        try:
            board.stop_stream()
        except:
            pass
        try:
            board.release_session()
        except:
            pass
        try:
            if csv_file:
                csv_file.close()
        except:
            pass

if __name__ == "__main__":
    main()
