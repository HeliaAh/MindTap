import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

# --- Settings ---
CSV_PATTERN = "eeg_log_*.csv"  # latest CSV to watch
UPDATE_MS = 100                 # refresh rate (ms)
WINDOW_SEC = 20                 # how many seconds of data to show

# --- Find latest CSV ---
paths = sorted(glob.glob(CSV_PATTERN))
if not paths:
    raise FileNotFoundError("No EEG log files found.")
csv_path = paths[-1]
print(f"📄 Streaming from {csv_path}")

# --- Matplotlib setup ---
plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(12, 5))
line, = ax.plot([], [], lw=2, color="royalblue")
ax.set_xlabel("Time since start (s)")
ax.set_ylabel("Avg RMS (µV)")
ax.set_title("Live EEG RMS Stream")
ax.set_ylim(0, 2000)   # adjust upper limit if your spikes are huge

# --- Data update function ---
t0 = None

def update(_):
    global t0
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return line,

    if df.empty:
        return line,

    # Parse timestamps only once initially (faster)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if t0 is None:
        t0 = df["timestamp"].iloc[0]

    df["time_s"] = (df["timestamp"] - t0).dt.total_seconds()
    x = df["time_s"]
    y = df["feature_uV_rms_avg"]

    # Optionally smooth a bit
    y = pd.Series(y).rolling(window=3, min_periods=1, center=True).mean()

    # Limit to last WINDOW_SEC seconds
    if len(x) and x.iloc[-1] > WINDOW_SEC:
        mask = x > (x.iloc[-1] - WINDOW_SEC)
        x, y = x[mask], y[mask]

    line.set_data(x, y)
    ax.set_xlim(x.iloc[0], x.iloc[-1] if len(x) else WINDOW_SEC)
    return line,

# --- Animate ---
ani = FuncAnimation(fig, update, interval=UPDATE_MS, blit=True)
plt.tight_layout()
plt.show()
