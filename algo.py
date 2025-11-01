from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from datetime import datetime
import time, glob, sys, os
import numpy as np
import pywt
from scipy.signal import find_peaks
from collections import deque
import matplotlib.pyplot as plt

# -------- CONFIG: choose your board here ----------
USE_SYNTHETIC = False  # set True to test without hardware
# --------------------------------------------------

def resolve_serial_port() -> str:
    """
    Find a likely OpenBCI dongle on macOS: /dev/cu.usbserial* or /dev/cu.usbmodem*.
    Raises if not found.
    """
    candidates = (
        glob.glob("/dev/cu.usbserial*")
        + glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/cu.SLAB_USBtoUART*")
        + glob.glob("/dev/cu.Bluetooth-Incoming-Port")  # harmless, will be ignored later if not real
    )
    # Filter out BT port if present
    candidates = [c for c in candidates if "Bluetooth-Incoming-Port" not in c]
    if not candidates:
        raise RuntimeError(
            "No /dev/cu.* serial device found. Plug in the OpenBCI dongle.\n"
            "Tip: run `ls /dev/cu.*` in Terminal to see available ports."
        )
    # Prefer usbserial over usbmodem if both exist
    candidates.sort(key=lambda p: (("usbserial" not in p), p))
    return candidates[0]

def main():
    BoardShim.enable_board_logger()

    if USE_SYNTHETIC:
        board_id = BoardIds.SYNTHETIC_BOARD.value
        params = BrainFlowInputParams()  # no serial port needed
        print("[brainflow] Using SYNTHETIC_BOARD for quick test.")
    else:
        board_id = BoardIds.CYTON_BOARD.value
        params = BrainFlowInputParams()
        try:
            params.serial_port = resolve_serial_port()
        except Exception as e:
            print("ERROR resolving serial port:", e)
            print("If you’re on macOS and this is a fresh setup, you may need a USB-serial driver (FTDI/CP210x), "
                  "or to grant Terminal/iTerm access to removable volumes in System Settings → Privacy & Security.")
            sys.exit(1)
        print(f"[brainflow] Using CYTON_BOARD on port: {params.serial_port}")

    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()  # use default buffer size
        print("[brainflow] Stream started.")

        # Get EEG channel indices for the ACTUAL board we chose
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        if not eeg_channels:
            raise RuntimeError("No EEG channels reported by board.")

        # ---- Live plot setup ----
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.set_title('EEG Signal')
        ax1.set_ylabel('Value (uV)')
        ax2.set_title('Peak Indicator (1 for peak, 0 otherwise)')
        ax2.set_ylabel('Binary')
        ax2.set_ylim(-0.1, 1.1)
        fig.suptitle('Real-time EEG Stream with Peak Detection')

        buffer_values = deque(maxlen=200)  # sliding window for CWT
        plot_times, plot_values, plot_binaries = [], [], []

        while True:
            # Pull the most recent sample (or few). If you want batches, use get_board_data.
            data = board.get_current_board_data(1)  # shape: (n_rows, n_cols)
            if data.shape[1] == 0:
                # No data yet; small wait
                time.sleep(0.02)
                continue

            ts = datetime.now()
            ts_str = ts.isoformat(timespec='milliseconds')

            # Take the first EEG channel from the actual board's EEG list
            ch0 = eeg_channels[2]
            eeg_value = data[ch0, -1]

            # Update buffers
            buffer_values.append(eeg_value)
            plot_times.append(ts)
            plot_values.append(eeg_value)

            # Peak detection when we have enough samples
            if len(buffer_values) >= 31:
                signal = np.array(buffer_values)
                widths = np.arange(1, 31)
                # CWT with Mexican hat
                cwtmatr, _ = pywt.cwt(signal, scales=widths, wavelet='mexh')
                abs_cwt = np.abs(cwtmatr)
                med = np.median(abs_cwt)
                max_abs = np.max(abs_cwt, axis=0)
                peak_indices, _ = find_peaks(max_abs, height=med * 3)
                binary = np.zeros(len(signal), dtype=int)
                binary[peak_indices] = 1
                bin_value = int(binary[-1])
            else:
                bin_value = 0

            plot_binaries.append(bin_value)
            print(f"{ts_str} {eeg_value:.6f} {bin_value}")

            # ---- Update plot (scroll last 100 points) ----
            ax1.cla(); ax2.cla()
            if len(plot_times) > 100:
                start = len(plot_times) - 100
                ax1.plot(plot_times[start:], plot_values[start:])
                ax2.step(plot_times[start:], plot_binaries[start:], where='post')
                ax1.set_xlim(plot_times[start], plot_times[-1])
                ax2.set_xlim(plot_times[start], plot_times[-1])
            else:
                ax1.plot(plot_times, plot_values)
                ax2.step(plot_times, plot_binaries, where='post')
                ax1.set_xlim(plot_times[0], plot_times[-1])
                ax2.set_xlim(plot_times[0], plot_times[-1])

            ax1.set_title('EEG Signal')
            ax1.set_ylabel('Value (uV)')
            ax2.set_title('Peak Indicator (1 for peak, 0 otherwise)')
            ax2.set_ylabel('Binary')
            ax2.set_ylim(-0.1, 1.1)

            plt.draw()
            plt.pause(0.001)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Streaming stopped by user.")
    except Exception as e:
        print("ERROR during streaming:", repr(e))
        print("Troubleshooting tips:")
        print(" • If using CYTON: verify the dongle is paired and not in use by another app (GUI/Hub).")
        print(" • Try power-cycling the board & dongle, then re-run.")
        print(" • On macOS, check `ls /dev/cu.*` shows a usbserial/usbmodem device.")
        print(" • If permission denied: try a different USB port/cable; grant Terminal/iTerm access in Privacy & Security.")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
        plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
