# eeg_ica.py
import sys
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CSV = "eeg_raw_1761801533.csv"
SFREQ = 250.0  # <-- set your sampling rate

# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------
df = pd.read_csv(CSV)

# Drop timestamp column if present
if df.columns[0].lower().startswith(("time", "timestamp")):
    df = df.iloc[:, 1:]

data = df.to_numpy().T  # shape: (n_ch, n_times)
n_ch, n_times = data.shape
dur = n_times / SFREQ
print(f"Loaded: {n_ch} channels, {n_times} samples (~{dur:.1f}s)")

# Hard stops
if n_ch < 2:
    sys.exit("Need >= 2 EEG channels for ICA. Record more channels and try again.")
if dur < 60:
    sys.exit("Recording too short for stable ICA. Record at least 1–2 minutes.")

# ------------------------------------------------------------
# Channel names: your exact 8 electrodes (order as you like)
# ------------------------------------------------------------
expected8 = ['Fp1', 'Fp2', 'F3', 'F4', 'Cz', 'Pz', 'O1', 'O2']

if n_ch == 8:
    ch_names = expected8
elif n_ch == 16:
    ch_names = ['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P3','Pz','P4']
else:
    ch_names = [f"EEG{i+1}" for i in range(n_ch)]

info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# ------------------------------------------------------------
# Montage setup
# 1) By default, use standard_1020 positions for the 8 labels you gave.
# 2) Optional: fill ch_pos_user with your own measured (x, y, z) in meters to override.
# MNE head coords: +X=right, +Y=front, +Z=up.
# ------------------------------------------------------------

# --- OPTIONAL USER OVERRIDES (leave as None to use standard positions) ---
ch_pos_user = {
    'Fp1': (-0.0293,  0.0859, 0.0305),
    'Fp2': ( 0.0293,  0.0859, 0.0305),
    'F3' : (-0.0469,  0.0375, 0.0763),
    'F4' : ( 0.0469,  0.0375, 0.0763),
    'Cz' : ( 0.0000,  0.0000, 0.0900),
    'Pz' : ( 0.0000, -0.0375, 0.0763),
    'O1' : (-0.0293, -0.0859, 0.0305),
    'O2' : ( 0.0293, -0.0859, 0.0305),
}

def set_montage_standard_or_user(raw, ch_pos_user):
    """Use user (x,y,z) if fully provided; otherwise pull from MNE standard_1020."""
    # If every value is a 3-tuple, use user montage
    all_filled = all(isinstance(v, tuple) and len(v) == 3 for v in ch_pos_user.values())
    if all_filled:
        mont = mne.channels.make_dig_montage(ch_pos=ch_pos_user, coord_frame='head')
        raw.set_montage(mont, on_missing='warn', match_case=False)
        print('[montage] Using USER coordinates.')
        return

    # Otherwise, grab canonical 10–20 coordinates for just our channels
    std = mne.channels.make_standard_montage("standard_1020")
    std_pos = std.get_positions()['ch_pos']  # dict: name -> (x,y,z)
    subset = {}
    missing = []
    for name in raw.ch_names:
        key = name  # case-sensitive match; standard_1020 is typically proper-cased
        if key in std_pos:
            subset[name] = tuple(float(v) for v in std_pos[key])
        else:
            missing.append(name)

    if missing:
        print(f"[montage] WARNING: Missing from standard_1020: {missing}. "
              f"Those channels will lack 3D positions.")

    mont = mne.channels.make_dig_montage(ch_pos=subset, coord_frame='head')
    raw.set_montage(mont, on_missing='warn', match_case=False)
    print('[montage] Using standard_1020 positions for available channels.')

set_montage_standard_or_user(raw, ch_pos_user)

# Optional: average reference before ICA (common in EEG pipelines)
raw.set_eeg_reference('average', projection=False)

# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------
raw.filter(1., 40., fir_design="firwin")
raw.notch_filter([50, 60])

# ------------------------------------------------------------
# ICA
# ------------------------------------------------------------
ica = mne.preprocessing.ICA(method="fastica", random_state=97, max_iter="auto")
ica.fit(raw)

src = ica.get_sources(raw).get_data()
A = ica.get_components()
var_share = src.var(axis=1)
var_share /= var_share.sum()
pick = np.argsort(var_share)[::-1][:4]

# Topomaps of selected ICs
figs = ica.plot_components(picks=pick, inst=raw, show=False)

# ------------------------------------------------------------
# PSDs for selected ICs (0–30 Hz)
# ------------------------------------------------------------
sf = raw.info["sfreq"]
for ic in pick:
    f, Pxx = welch(src[ic], fs=sf, nperseg=int(sf*2), noverlap=int(sf))
    m = (f >= 0) & (f <= 30)
    plt.figure()
    plt.plot(f[m], 10*np.log10(Pxx[m]))
    plt.title(f"IC {ic} ({100*var_share[ic]:.2f}%)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid(True, alpha=.3)

plt.show()
