from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from datetime import datetime
import time
import numpy as np
import pywt
from scipy.signal import find_peaks
from collections import deque
import matplotlib.pyplot as plt

BoardShim.enable_dev_board_logger()  # Optional: helps with debugging
params = BrainFlowInputParams()
board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()
board.start_stream()

try:
    # Set up live plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.set_title('EEG Signal')
    ax1.set_ylabel('Value')
    ax2.set_title('Peak Indicator (1 for peak, 0 otherwise)')
    ax2.set_ylabel('Binary')
    ax2.set_ylim(-0.1, 1.1)
    fig.suptitle('Real-time EEG Stream with Peak Detection')

    # Buffers for detection (sliding window)
    buffer_values = deque(maxlen=200)  # Window size for wavelet computation

    # Lists for plotting (accumulate, but limit view)
    plot_times = []
    plot_values = []
    plot_binaries = []

    while True:
        data = board.get_current_board_data(1)
        
        # Check if data is available
        if data.shape[1] > 0:
            timestamp = datetime.now()
            timestamp_str = timestamp.isoformat(timespec='milliseconds')
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD)
            eeg_value = data[eeg_channels[0]][0]
            
            # Append to buffer for detection
            buffer_values.append(eeg_value)
            
            # Append to plot lists
            plot_times.append(timestamp)
            plot_values.append(eeg_value)
            
            # Compute binary if buffer is sufficiently large
            if len(buffer_values) >= 31:
                signal = np.array(buffer_values)
                widths = np.arange(1, 31)
                cwtmatr, _ = pywt.cwt(signal, scales=widths, wavelet='mexh')
                abs_cwt = np.abs(cwtmatr)
                med = np.median(abs_cwt)
                max_abs = np.max(abs_cwt, axis=0)
                peak_indices, _ = find_peaks(max_abs, height=med * 3)
                
                binary = np.zeros(len(signal), dtype=int)
                binary[peak_indices] = 1
                bin_value = binary[-1]
            else:
                bin_value = 0
            
            plot_binaries.append(bin_value)
            
            # Print to console in real-time
            print(f"{timestamp_str} {eeg_value:.14f} {bin_value}")
            
            # Update plot (scrolling window of last 100 points)
            ax1.cla()
            ax2.cla()
            if len(plot_times) > 100:
                start_idx = len(plot_times) - 100
                ax1.plot(plot_times[start_idx:], plot_values[start_idx:])
                ax2.step(plot_times[start_idx:], plot_binaries[start_idx:], where='post')
                ax1.set_xlim(plot_times[start_idx], plot_times[-1])
                ax2.set_xlim(plot_times[start_idx], plot_times[-1])
            else:
                ax1.plot(plot_times, plot_values)
                ax2.step(plot_times, plot_binaries, where='post')
                ax1.set_xlim(plot_times[0], plot_times[-1])
                ax2.set_xlim(plot_times[0], plot_times[-1])
            
            ax1.set_title('EEG Signal')
            ax1.set_ylabel('Value')
            ax2.set_title('Peak Indicator (1 for peak, 0 otherwise)')
            ax2.set_ylabel('Binary')
            ax2.set_ylim(-0.1, 1.1)
            plt.draw()
            plt.pause(0.001)
        else:
            print("Waiting for data...")
        
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Streaming stopped.")
finally:
    board.stop_stream()
    board.release_session()
    plt.ioff()
    plt.show()