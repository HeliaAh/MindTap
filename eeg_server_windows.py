# eeg_server_windows.py
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from websocket_server import WebsocketServer
import numpy as np
import time, threading, sys, glob, os
import csv
from datetime import datetime

# ===== CONFIG =====
# Use .value so BrainFlow gets the integer ID it expects
BOARD_ID     = BoardIds.CYTON_BOARD.value      # set to BoardIds.SYNTHETIC_BOARD.value for quick tests
SERIAL_PORT  = "COM7"                               # Windows: "COM5" etc. Leave empty to auto-detect.
WS_PORT      = 8080
SAMPLE_BATCH = 50
TICK_SEC     = 0.02
USE_N_CHANS  = 8                                # average this many EEG channels
# ==================

def resolve_serial_port(requested: str) -> str:
    """
    Cross-platform serial port resolution:
    - If 'requested' provided, use it as-is (e.g., 'COM5' on Windows).
    - Windows: try pyserial list_ports; prefer OpenBCI/CP210x/CH340 style adapters.
    - macOS/Linux: scan /dev/cu.* and common tty* USB endpoints.
    """
    if requested:
        return requested

    if os.name == "nt":  # Windows
        try:
            from serial.tools import list_ports
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
            return scored[0][1]  # best match (e.g., 'COM5')
        except Exception:
            # Fallback guesses (BrainFlow will error if wrong)
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

def main():
    params = BrainFlowInputParams()
    # Only set serial port for boards that need it
    if BOARD_ID == BoardIds.CYTON_BOARD.value:
        port = resolve_serial_port(SERIAL_PORT)
        print(f"[eeg_server] Using serial port: {port}")
        params.serial_port = port

    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)

    feature_csv_file = None
    raw_csv_file = None
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

        # CSV headers
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
            data = board.get_board_data(SAMPLE_BATCH)  # shape: (rows, cols)
            if data is None or data.size == 0:
                time.sleep(TICK_SEC)
                continue

            ts = data[ts_idx, :]
            X = data[chans, :].astype(np.float64)
            if X.size == 0:
                time.sleep(SAMPLE_BATCH / fs if fs else TICK_SEC)
                continue

            # channel-wise filtering & RMS (1–40 Hz; 60 Hz notch)
            rms_vals = []
            for r in range(X.shape[0]):
                v = X[r].copy()
                DataFilter.detrend(v, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(v, fs, 1.0, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                try:
                    DataFilter.remove_environmental_noise(v, fs, NoiseTypes.SIXTY.value)
                except AttributeError:
                    DataFilter.perform_bandstop(v, fs, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
                rms_vals.append(np.sqrt(np.mean(v**2)))

            feature = float(np.mean(rms_vals)) if rms_vals else 0.0
            server.send_message_to_all(str(feature))

            timestamp = datetime.now().isoformat(timespec='milliseconds')
            feature_writer.writerow([timestamp, feature])
            feature_csv_file.flush()
            print(f"[LOG] {timestamp} | {feature:.3f}")

            # raw channel snapshot per sample with ISO ts (from Board timestamp)
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
        try:
            board.stop_stream()
        except:
            pass
        try:
            board.release_session()
        except:
            pass
        try:
            if feature_csv_file:
                feature_csv_file.close()
            if raw_csv_file:
                raw_csv_file.close()
        except:
            pass

if __name__ == "__main__":
    main()
