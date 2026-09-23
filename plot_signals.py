import os
import sys
import pandas as pd
import numpy as np
from aux.bit_eda_sensor import bitalino_eda_to_us
from aux.funcs import get_signals_as_dict
from aux.signal_processing import filter_eda, acc_multi_filtering, dac_settling_time, crop_arrays, normalize
import matplotlib.pyplot as plt
from aux.sym_eda_sensor import raw_to_conductance

root_path = sys.argv[0]
root_path, _ = os.path.split(root_path)
data_path = os.path.join(root_path, 'data')
signals_path = os.path.join(data_path, "signals")
coeff_path = os.path.join(data_path, 'TF', 'fit_coeffs_4-1A__All_max.csv')
df_coeffs = pd.read_csv(coeff_path, comment='#')

FS = 1000.

time_scale = 's' # or 'm' or ms

if time_scale == 'm':
    time_factor = 1 / 60_000
elif time_scale == 's':
    time_factor = 1 / 1000
else:
    time_factor = 1

figures_path = os.path.join(root_path, "figures")
os.makedirs(figures_path, exist_ok=True)

s_fnames = [x for x in os.listdir(signals_path) if x.endswith(".csv")]

for s_filename in s_fnames:
    clean_fname = s_filename.split('.csv')[0]
    print(f"Signal {clean_fname}")

    sym_path = os.path.join(signals_path, s_filename)
    signals = get_signals_as_dict(sym_path, device_name='sympathia')
    sympathia_signals = signals['sympathia']

    # From now on, it's the standard part of the pipeline:
    dac_sym = sympathia_signals['AI4']
    eda_fingers = sympathia_signals['AI6']
    eda_wrist = sympathia_signals['AX7']

    acc_raw_axes = np.asarray([sympathia_signals["AI1"],
                               sympathia_signals["AI2"],
                               sympathia_signals["AI3"]])

    acc_filt_axes = np.column_stack(acc_multi_filtering(acc_raw_axes, fs=int(FS)))
    acc_vm = np.linalg.norm(acc_filt_axes, axis=1)

    eda_wrist, crop_idx = raw_to_conductance(eda_wrist, dac_sym, df_coeffs, crop_leading_nans=False)
    eda_fingers = bitalino_eda_to_us(eda_fingers, n_bits=12, vcc=3.3)

    eda_wrist, eda_fingers, acc_vm, dac_sym = crop_arrays(
        crop_idx, eda_wrist, eda_fingers, acc_vm, dac_sym)

    # --- second crop: DAC settling + tolerance ---
    settle_idx, transitions, intervals_s = dac_settling_time(dac_sym)
    CROP_TOL_S = 10
    crop_settle = settle_idx + int(CROP_TOL_S * FS)

    eda_wrist, eda_fingers, acc_vm, dac_sym = crop_arrays(
        crop_settle, eda_wrist, eda_fingers, acc_vm, dac_sym)

    # filter AFTER cropping, so the settling transient isn't smeared into the kept data
    eda_wrist = filter_eda(eda_wrist)
    eda_fingers = filter_eda(eda_fingers)

    time_arr = np.arange(len(eda_wrist)) * time_factor

    saturation_type = 'Sensor'  # ADC or Sensor
    if saturation_type == 'Sensor':
        low_lim, up_lim = 300_000, 7_000_000
    else:
        low_lim, up_lim = 5_000, 8_360_000

    fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f"Subject {s_filename}")

    axes[0].plot(time_arr, normalize(eda_wrist), label="EDA Wrist")
    axes[0].plot(time_arr, normalize(eda_fingers), label="EDA Fingers")
    axes[0].set_ylabel("Norm. Amplitude [0-1]")
    # axes[0].set_ylabel("Conductance [$\\mu$S]")
    axes[0].set_title("EDA signals")

    axes[1].plot(time_arr, acc_vm,
                 linewidth=0.8, color="#d62728")
    axes[1].set_ylabel("[a.u.]")
    axes[1].set_title("ACC VM")

    axes[2].plot(time_arr, dac_sym)
    axes[2].set_ylabel("[0-255]")
    axes[2].set_xlabel(f"Time [{time_scale}]")

    fig.savefig(os.path.join(figures_path, f"{clean_fname}_plot.png"), dpi=150)
    fig.tight_layout()
    plt.show()