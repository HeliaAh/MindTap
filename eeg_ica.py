# eeg_ica.py
import pandas as pd, numpy as np, mne, sys
from scipy.signal import welch
import matplotlib.pyplot as plt

import numpy as np
import mne

def forehead_band_coords(n=8, spacing=0.05, R=0.09, z_front=0.05, names=None):
    """
    Make n points on a great-circle arc tilted so the center is at z_front above origin.
    spacing: chord length along the arc (meters), approx skin separation.
    Returns ch_pos dict in MNE head coords (X=right, Y=front, Z=up).
    """
    # angular step along a great circle for given chord spacing
    dphi = 2.0 * np.arcsin(spacing / (2.0 * R))
    # tilt arc upward so the front-center point has desired z
    theta = np.arcsin(np.clip(z_front / R, -1.0, 1.0))

    # indices symmetric about the front (φ=0)
    ks = np.arange(n) - (n - 1) / 2.0
    ch_pos = {}
    if names is None:
        names = [f"F{i+1}" for i in range(n)]

    for name, k in zip(names, ks):
        phi = k * dphi
        # great circle around Y-axis, then rotate about X by theta
        x = R * np.sin(phi)
        y = R * np.cos(phi) * np.cos(theta)
        z = R * np.cos(phi) * np.sin(theta)
        ch_pos[name] = (float(x), float(y), float(z))
    return ch_pos

# Example: 8 points, 5 cm spacing
ch_pos = forehead_band_coords(n=8, spacing=0.05, R=0.09, z_front=0.05,
                              names=['F1','F2','F3','F4','F5','F6','F7','F8'])

# Build montage and apply
mont = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
raw.set_montage(mont, on_missing='warn', match_case=False)

# sanity check
# raw.plot_sensors(kind='topomap'); raw.plot_sensors(kind='3d')


CSV = "eeg_raw_1761705811.csv"
SFREQ = 250.0  # <-- set your sampling rate

df = pd.read_csv(CSV)

# Drop timestamp if present
if df.columns[0].lower().startswith(("time","timestamp")):
    df = df.iloc[:, 1:]

data = df.to_numpy().T  # shape: (n_ch, n_times)
n_ch, n_times = data.shape
dur = n_times / SFREQ
print(f"Loaded: {n_ch} channels, {n_times} samples (~{dur:.1f}s)")

# --- hard stops that explain what's wrong ---
if n_ch < 2:
    sys.exit("Need >= 2 EEG channels for ICA. Record more channels and try again.")
if dur < 60:
    sys.exit("Recording too short for stable ICA. Record at least 1–2 minutes.")

# Choose standard 10–20 names for common counts
names_8  = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2']
names_16 = ['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P3','Pz','P4']
if n_ch == 8:
    ch_names = names_8
elif n_ch == 16:
    ch_names = names_16
else:
    # Fallback: generic names; montage will ignore missing positions
    ch_names = [f"EEG{i+1}" for i in range(n_ch)]

info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# Set montage (ignore any names the template doesn't know)
mont = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(mont, match_case=False, on_missing="ignore")

# Preprocess
raw.filter(1., 40., fir_design="firwin")
raw.notch_filter([50, 60])

# ICA
ica = mne.preprocessing.ICA(method="fastica", random_state=97, max_iter="auto")
ica.fit(raw)

src = ica.get_sources(raw).get_data()
A = ica.get_components()
var_share = src.var(axis=1); var_share /= var_share.sum()
pick = np.argsort(var_share)[::-1][:4]

figs = ica.plot_components(picks=pick, inst=raw, show=False)  # returns a list of figures

# PSDs 0–30 Hz
import matplotlib.pyplot as plt
from scipy.signal import welch

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
