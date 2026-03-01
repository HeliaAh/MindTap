from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from websocket_server import WebsocketServer
import numpy as np
import time, threading, sys, glob, os
import csv
from datetime import datetime

# ===== CONFIG =====
BOARD_ID     = BoardIds.SYNTHETIC_BOARD.value      # set to BoardIds.CYTON_BOARD.value for real hardware
SERIAL_PORT  = "COM7"                              # Windows: "COM5" etc. Leave empty to auto-detect.
WS_PORT      = 8080
SAMPLE_BATCH = 50
TICK_SEC     = 0.02
USE_N_CHANS  = 8                                   # average this many EEG channels
SAMPLES_PER_WINDOW = 50                            # max samples before continuous clipping is considered railing
# ==================

class EEGRailingDetector:
    def __init__(self, num_channels=8, threshold_uv=10000.0):
        """
        Initializes the railing detector for live OpenBCI Cyton data.
        """
        self.num_channels = num_channels
        self.threshold_uv = threshold_uv

    def check_railing(self, data_window):
        """
        Checks a window of live EEG data for railing.
        """
        # np.any(..., axis=1) checks across the samples for each channel
        is_railing_array = np.any(np.abs(data_window) >= self.threshold_uv, axis=1)
        railing_status = is_railing_array.tolist()
        railing_channels = [ch + 1 for ch, railing in enumerate(railing_status) if railing]
        
        return railing_status, railing_channels


def main():
    # 1. Setup BrainFlow Input Parameters
    params = BrainFlowInputParams()
    # Only assign serial port if we are using a real board (not synthetic)
    if BOARD_ID != BoardIds.SYNTHETIC_BOARD.value and SERIAL_PORT:
        params.serial_port = SERIAL_PORT

    # 2. Initialize the Board
    try:
        board = BoardShim(BOARD_ID, params)
        board.prepare_session()
        board.start_stream()
        print(f"BrainFlow stream started on Board ID {BOARD_ID}. Monitoring for railing...")
    except Exception as e:
        print(f"Failed to start board: {e}")
        sys.exit(1)

    # 3. Get EEG channels for this specific board and subset to the ones we want
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    active_eeg_channels = eeg_channels[:USE_N_CHANS]

    # 4. Initialize the detector
    detector = EEGRailingDetector(num_channels=USE_N_CHANS, threshold_uv=10000.0)

    try:
        while True:
            # Wait for the tick duration
            time.sleep(TICK_SEC)

            # 5. Fetch the latest window of data
            # get_current_board_data gets the latest N samples without clearing the internal buffer completely
            data = board.get_current_board_data(SAMPLES_PER_WINDOW)

            # Ensure we have enough data before checking
            if data.shape[1] >= 1: 
                # Isolate just the EEG channels
                eeg_data = data[active_eeg_channels, :]
                
                # 6. Check for railing
                status, bad_channels = detector.check_railing(eeg_data)
                
                if bad_channels:
                    # Print timestamp and the channels that are railing
                    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{now}] ⚠️ RAILING DETECTED on Channels: {bad_channels}")
                
                # Note: Here is where you would normally pass 'status' or 'data' 
                # to your WebsocketServer to broadcast to the frontend UI.
                
    except KeyboardInterrupt:
        print("\nStopping stream...")
        
    finally:
        # 7. Safely close the session to free up the serial port
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Session released and cleanly shut down.")

if __name__ == "__main__":
    # Enable BrainFlow logging for easier debugging (optional)
    BoardShim.enable_dev_board_logger()
    main()