import numpy as np
import time
from datetime import datetime
from scipy.signal import welch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# -------------------------------------------------------------
# Function to calculate EEG SNR
# -------------------------------------------------------------
def calculate_eeg_snr(psd, fs, nfft, band):
    """
    Calculate SNR (Signal-to-Noise Ratio) from PSD.
    """
    freqs = np.linspace(0, fs / 2, nfft // 2 + 1)
    df = fs / nfft
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    signal_power = np.sum(psd[band_mask]) * df
    noise_power = np.sum(psd[~band_mask]) * df
    snr_db = 20 * np.log10(signal_power / noise_power)
    return signal_power, noise_power, snr_db


# -------------------------------------------------------------
# BrainFlow setup
# -------------------------------------------------------------
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)

board.prepare_session()
board.start_stream()

# -------------------------------------------------------------
# Live SNR computation parameters
# -------------------------------------------------------------
fs = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD)
nfft = 1000  # length for FFT
band = (8, 12)  # alpha band
window_size = fs * 2  # 2-second window

print("Starting live EEG SNR computation. Press Ctrl+C to stop.\n")

try:
    while True:
        # Collect latest EEG samples
        data = board.get_current_board_data(window_size)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD)
        eeg_data = data[eeg_channels[0]]  # use first EEG channel

        if eeg_data.size > 0:
            # Compute PSD for current window
            freqs, psd = welch(eeg_data, fs=fs, nperseg=nfft)

            # Compute SNR
            signal_power, noise_power, snr_db = calculate_eeg_snr(psd, fs, nfft, band)

            # Print results
            timestamp = datetime.now().isoformat(timespec='milliseconds')
            print(f"{timestamp} | SNR ({band[0]}-{band[1]} Hz): {snr_db:6.2f} dB")

        else:
            print("Waiting for data...")

        # Update every 0.5 seconds
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStreaming stopped by user.")

finally:
    board.stop_stream()
    board.release_session()
