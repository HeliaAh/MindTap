# eeg_snr_cyton_windows.py
import numpy as np
import time, os, glob, sys
from datetime import datetime
from scipy.signal import welch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ------------------------- Config -------------------------
BOARD_ID = BoardIds.CYTON_BOARD.value   # use CYTON
SERIAL_PORT = "COM7"                        # e.g., "COM5"; leave "" to auto-detect
BAND = (8, 12)                          # alpha band for SNR
NFFT = 1024                             # FFT length for Welch
UPDATE_SEC = 0.5                        # print/update rate
# ----------------------------------------------------------

def resolve_serial_port(requested: str) -> str:
    """Cross-platform serial port resolution. Returns a COM device (Windows) or /dev/* (Unix)."""
    if requested:
        return requested

    # Windows
    if os.name == "nt":
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            if not ports:
                raise RuntimeError("No COM ports found. Plug in the OpenBCI dongle.")
            # prefer likely OpenBCI USB-serial adapters
            keywords = ("openbci", "cp210", "silicon labs", "wch", "ch340", "usb-serial")
            scored = []
            for p in ports:
                desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
                score = sum(k in desc for k in keywords)
                scored.append((score, p.device))
            scored.sort(reverse=True)
            return scored[0][1]  # e.g., "COM5"
        except Exception:
            # Fallback guesses; BrainFlow will error if incorrect
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

def calculate_eeg_snr(psd, fs, nfft, band):
    """Compute SNR (dB) from a one-sided PSD array corresponding to Welch(freqs, psd)."""
    # Build frequency axis matching scipy.signal.welch output (one-sided, length=len(psd))
    freqs = np.linspace(0, fs / 2, len(psd))
    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / nfft
    band_mask = (freqs >= band[0]) & (freqs <= band[1])

    signal_power = np.sum(psd[band_mask]) * df
    noise_power = np.sum(psd[~band_mask]) * df
    # Guard against zeros
    if noise_power <= 0 or signal_power <= 0:
        return signal_power, noise_power, float("-inf")
    snr_db = 10 * np.log10(signal_power / noise_power)  # PSD power ratio → 10*log10
    return signal_power, noise_power, snr_db

def main():
    BoardShim.enable_board_logger()

    params = BrainFlowInputParams()
    # CYTON requires serial port
    params.serial_port = resolve_serial_port(SERIAL_PORT)
    print(f"[cyton] Using serial port: {params.serial_port}")

    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        board.start_stream()

        fs = BoardShim.get_sampling_rate(BOARD_ID)
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        if not eeg_channels:
            print("No EEG channels reported by board config.")
            return

        # Window length for live PSD (use at least 2 seconds)
        window_size = int(max(fs * 2, NFFT))  # ensure enough samples for Welch nperseg

        print(f"Starting live EEG SNR on CYTON. fs={fs} Hz, nfft={NFFT}, window={window_size} samples. Ctrl+C to stop.\n")

        while True:
            # Get most recent samples
            data = board.get_current_board_data(window_size)  # shape: (rows, cols)
            if data.size == 0:
                print("Waiting for data...")
                time.sleep(UPDATE_SEC)
                continue

            ch0 = eeg_channels[0]
            eeg_data = data[ch0, :]
            if eeg_data.size < max(256, NFFT // 2):
                # Not enough data yet for stable Welch estimate
                time.sleep(UPDATE_SEC)
                continue

            # Welch PSD (SciPy returns one-sided freqs, psd). Ensure nperseg <= len(eeg_data).
            nperseg = min(NFFT, eeg_data.size)
            noverlap = nperseg // 2
            freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap)

            # SNR
            _, _, snr_db = calculate_eeg_snr(psd, fs, nperseg, BAND)

            # Print
            timestamp = datetime.now().isoformat(timespec='milliseconds')
            print(f"{timestamp} | SNR {BAND[0]}-{BAND[1]} Hz: {snr_db:6.2f} dB")

            time.sleep(UPDATE_SEC)

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
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

if __name__ == "__main__":
    main()
