"Guneev"
# eeg_server.py
from importlib.resources import as_file
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from websocket_server import WebsocketServer
import numpy as np
import time, threading, sys, glob
import csv
from datetime import datetime

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

        ts_idx = BoardShim.get_timestamp_channel(BOARD_ID)

        # create CSV file and write header
        epoch = int(time.time())
        feature_csv_filename = f"eeg_log_{epoch}.csv"
        feature_csv_file = open(feature_csv_filename, 'w', newline='')
        feature_writer = csv.writer(feature_csv_file)
        feature_writer.writerow(["timestamp", "feature_uV_rms_avg"])

        raw_csv_filename = f"eeg_raw_{epoch}.csv"
        raw_csv_file = open(raw_csv_filename, 'w', newline='')
        raw_writer = csv.writer(raw_csv_file)
        raw_writer.writerow(["timestamp"] + [f"ch{i+1}" for i in range(len(chans))])

        while True:
            data = board.get_board_data(SAMPLE_BATCH)  # (rows, cols)
            ts = data[ts_idx, :]
            
            if data is None or data.size == 0:
                time.sleep(TICK_SEC); continue

            X = data[chans, :].astype(np.float64)  # (n_ch, n_samples)
            if X.size == 0:
                time.sleep(SAMPLE_BATCH / fs); continue

            rms_vals = []
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
                rms_vals.append(np.sqrt(np.mean(v**2)))
            ##latest_vals = X[:, -1]                  # last sample per channel

            feature = float(np.mean(rms_vals))  # average across selected channels
            server.send_message_to_all(str(feature))


            timestamp = datetime.now().isoformat(timespec='milliseconds')
            feature_writer.writerow([timestamp, feature])
            feature_csv_file.flush()
            print(f"[LOG] {timestamp} | {feature:.3f}")

            for j in range(X.shape[1]):
                ts_iso = datetime.fromtimestamp(ts[j]).isoformat(timespec='milliseconds')
                row = [ts_iso] + [f"{X[i, j]:.6f}" for i in range(X.shape[0])]
                raw_writer.writerow(row)
                raw_csv_file.flush()


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
            if feature_csv_file: feature_csv_file.close()
            if raw_csv_file: raw_csv_file.close()
        except:
            pass

if __name__ == "__main__":
    main()
