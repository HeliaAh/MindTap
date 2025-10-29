"Guneev"
# eeg_server.py
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from websocket_server import WebsocketServer
import numpy as np
import time, threading, sys, glob
import csv
from datetime import datetime
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt

# ===== CONFIG =====
BOARD_ID     = BoardIds.CYTON_BOARD   # use SYNTHETIC_BOARD for a quick test
SERIAL_PORT  = ""                           # leave empty to auto-detect /dev/cu.*
WS_PORT      = 8080
SAMPLE_BATCH = 50
TICK_SEC     = 0.02
USE_N_CHANS  = 8                            # average this many EEG channels
# ==================

def resolve_serial_port(requested: str) -> str:
    if requested:
        if requested.startswith("/dev/tty."):
            requested = "/dev/cu." + requested.split("/dev/tty.", 1)[1]
        return requested
    candidates = (
        glob.glob("/dev/cu.usbserial*") +
        glob.glob("/dev/cu.usbmodem*") +
        glob.glob("/dev/cu.SLAB_USBtoUART*")
    )
    if not candidates:
        raise RuntimeError("No /dev/cu.* serial device found. Plug in the OpenBCI dongle.")
    return sorted(candidates)[0]

def compute_band_power(freqs, psd, band):
    """Return average power in a frequency band (safe, no NaN)."""
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx_band):          # no data in band
        return 0.0
    band_psd = psd[idx_band]
    if np.all(np.isnan(band_psd)):    # avoid NaN propagation
        return 0.0
    return float(np.nanmean(band_psd))

def compute_psd(signal, fs):
    """
    Compute the Power Spectral Density (PSD) of a signal mathematically using FFT.
    
    Parameters
    ----------
    signal : array_like
        Input time-domain signal (1D array).
    fs : float
        Sampling frequency in Hz.
    
    Returns
    -------
    freqs : np.ndarray
        Frequency bins corresponding to the PSD values.
    psd : np.ndarray
        Power Spectral Density (power per Hz).
    """

    # Ensure it's a NumPy array
    signal = np.asarray(signal, dtype=np.float64)

    # Remove DC offset (mean)
    signal = signal - np.mean(signal)

    N = len(signal)  # number of samples

    # Compute FFT (complex spectrum)
    fft_vals = np.fft.fft(signal)

    # Compute corresponding frequency bins
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Keep only the positive half of the spectrum
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    # Compute the two-sided power spectrum
    psd = (1 / (fs * N)) * np.abs(fft_vals) ** 2

    # Double non-DC components to preserve total power (since we dropped negative freqs)
    psd[1:-1] *= 2

    return freqs, psd


def main():
    params = BrainFlowInputParams()
    if BOARD_ID == BoardIds.CYTON_BOARD.value:
        port = resolve_serial_port(SERIAL_PORT)
        print(f"[eeg_server] Using serial port: {port}")
        params.serial_port = port

    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)

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

        # create CSV file and write header
        csv_filename = f"eeg_log_{int(time.time())}.csv"
        csv_file = open(csv_filename, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "feature_uV_rms_avg"])

        while True:
            data = board.get_board_data(SAMPLE_BATCH)  # (rows, cols)
            if data is None or data.size == 0:
                time.sleep(TICK_SEC); continue

            X = data[chans, :].astype(np.float64)  # (n_ch, n_samples)
            if X.size == 0:
                time.sleep(SAMPLE_BATCH / fs); continue
            
            # Average across EEG channels
            avg_signal = np.mean(X, axis=0)
            
            if 'signal_buffer' not in locals():
                from collections import deque
                signal_buffer = deque(maxlen=int(fs * 4))  # 4-second rolling window

            signal_buffer.extend(avg_signal)

            WINDOW_SIZE = 1000

            if len(signal_buffer) >= WINDOW_SIZE:  # need at  1s of data
                sig = np.array(list(signal_buffer)[-WINDOW_SIZE:], dtype=np.float64)
                # Detrend in-place (BrainFlow modifies the array directly)
                DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
                if len(signal_buffer) >= WINDOW_SIZE:  # need at least 1s of data
                    sig = np.array(list(signal_buffer)[-WINDOW_SIZE:], dtype=np.float64)
    
                # Detrend to remove drift
                DataFilter.detrend(sig, DetrendOperations.LINEAR.value)

                # Compute PSD using your FFT-based function
                freqs, psd = compute_psd(sig, fs)

                # Compute band powers
                delta = compute_band_power(freqs, psd, (0.5, 4))
                theta = compute_band_power(freqs, psd, (4, 7))
                alpha = compute_band_power(freqs, psd, (8, 12))
                beta  = compute_band_power(freqs, psd, (13, 30))
                gamma = compute_band_power(freqs, psd, (30, 40))
            else:
                delta = theta = alpha = beta = gamma = 0.0


            '''##rms_vals = []
            for r in range(X.shape[0]):
                v = X[r].copy()
                # Detrend (remove DC)
                DataFilter.detrend(v, DetrendOperations.CONSTANT.value)
                # Band-pass 1–40 Hz
                DataFilter.perform_bandpass(v, fs, 1.0, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                # Notch 60 Hz (handle old/new BrainFlow)
                try:
                    DataFilter.remove_environmental_noise(v, fs, NoiseTypes.SIXTY.value)
                except AttributeError:
                    # Fallback: band-stop 58–62 Hz
                    DataFilter.perform_bandstop(v, fs, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                # RMS amplitude
                rms_vals.append(np.sqrt(np.mean(v**2)))'''
            latest_vals = X[:, -1]                  # last sample per channel

            feature = float(np.mean(latest_vals))  # average across selected channels
            server.send_message_to_all(str(feature))


            timestamp = datetime.now().isoformat(timespec='milliseconds')
            csv_writer.writerow([timestamp, feature, delta, theta, alpha, beta, gamma])
            csv_file.flush()  # ensures it's written immediately
            print(f"[LOG] {timestamp} | {feature:.3f} | delta: {delta:.3f} | theta: {theta:.3f} | alpha: {alpha:.3f} | beta: {beta:.3f} | gamma: {gamma:.3f} ")

            time.sleep(TICK_SEC)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
    finally:
        try: board.stop_stream()
        except: pass
        try: board.release_session()
        except: pass
        try:
            csv_file.close()
        except:
            pass

if __name__ == "__main__":
    main()