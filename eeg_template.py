# eeg_capture_template.py
# Minimal, Windows-ready template to acquire raw EEG from OpenBCI Cyton (BrainFlow)

import os, glob, time, sys
import numpy as np
from typing import Iterator, Optional, List
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ===================== CONFIG =====================
BOARD_ID        = BoardIds.CYTON_BOARD.value   # Cyton default. For simulation: BoardIds.SYNTHETIC_BOARD.value
SERIAL_PORT     = "COM7"                            # e.g., "COM5". Leave "" to auto-detect on Windows.
FETCH_HZ        = 60                            # how often we pull new samples (approx.)
USE_VOLTS       = True                          # convert EEG to Volts (BrainFlow Cyton EEG is µV)
SELECT_FIRST_N  = 8                             # how many EEG channels to read (Cyton has up to 8)
# ==================================================

def resolve_serial_port(requested: str) -> str:
    """
    Cross-platform serial-port locator. On Windows, prefers OpenBCI/USB-serial adapters.
    Returns a COM device (e.g., 'COM5') or a /dev/* path.
    """
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
            # Fallback guesses; BrainFlow will error if wrong
            for guess in ("COM3", "COM4", "COM5", "COM6", "COM7", "COM8"):
                return guess

    # macOS / Linux (still works here)
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


class EEGAcquirer:
    """
    Simple wrapper over BrainFlow for raw EEG capture.

    Usage:
        with EEGAcquirer() as eeg:
            for samples_v in eeg.stream_samples():  # (n_ch, n_new)
                # do stuff with samples_v (Volts) or samples_uv (µV) if USE_VOLTS=False
                ...
    """
    def __init__(self,
                 board_id: int = BOARD_ID,
                 serial_port: str = SERIAL_PORT,
                 select_first_n: Optional[int] = SELECT_FIRST_N,
                 convert_to_volts: bool = USE_VOLTS):
        self.board_id = board_id
        self.serial_port = serial_port
        self.select_first_n = select_first_n
        self.convert_to_volts = convert_to_volts

        self.params = BrainFlowInputParams()
        if self.board_id == BoardIds.CYTON_BOARD.value:
            self.params.serial_port = resolve_serial_port(self.serial_port)

        self.board: Optional[BoardShim] = None
        self.fs: Optional[int] = None
        self.eeg_channels: List[int] = []

    # Context manager
    def __enter__(self):
        BoardShim.enable_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.board.start_stream()
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        all_eeg = BoardShim.get_eeg_channels(self.board_id)
        if self.select_first_n is not None:
            self.eeg_channels = all_eeg[:min(self.select_first_n, len(all_eeg))]
        else:
            self.eeg_channels = all_eeg
        print(f"[eeg] fs={self.fs} Hz | EEG chans={self.eeg_channels} | port={self.params.serial_port or 'N/A'}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.board:
                self.board.stop_stream()
        finally:
            try:
                if self.board:
                    self.board.release_session()
            finally:
                self.board = None

    def stream_samples(self) -> Iterator[np.ndarray]:
        """
        Yields newly available samples as a 2D array shape (n_channels, n_new).
        - For Cyton, BrainFlow returns EEG in µV; convert to Volts if convert_to_volts=True.
        - Pull cadence ~ FETCH_HZ.
        """
        assert self.board is not None, "Call inside 'with EEGAcquirer() as eeg:'"
        uv_to_v = 1e-6

        # Choose a small chunk so latency is low, but not 1 (too chatty).
        chunk = max(1, int((self.fs or 250) / FETCH_HZ))

        while True:
            data = self.board.get_current_board_data(chunk)  # (rows, cols)
            if data.size > 0:
                block = data[self.eeg_channels, :]  # µV for Cyton
                if self.convert_to_volts:
                    block = (block * uv_to_v).astype(np.float32, copy=False)  # → Volts
                else:
                    block = block.astype(np.float32, copy=False)              # µV
                if block.shape[1] > 0:
                    yield block  # (n_channels, n_new)
            # light pacing
            time.sleep(max(0.0, 1.0 / FETCH_HZ))


# ------------------- Demo / CLI -------------------
if __name__ == "__main__":
    """
    Example: print stream health and first few values, then exit.
      python eeg_capture_template.py
    """
    try:
        with EEGAcquirer() as eeg:
            n_print = 10
            for block in eeg.stream_samples():
                n_new = block.shape[1]
                # Show first sample of first channel (in Volts by default)
                val0 = float(block[0, -1]) if n_new else float("nan")
                print(f"+{n_new:3d} samples | ch0 last = {val0:.6e} {'V' if USE_VOLTS else 'uV'}")
                n_print -= 1
                if n_print <= 0:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
