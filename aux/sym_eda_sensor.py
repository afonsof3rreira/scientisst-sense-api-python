import numpy as np


def g_to_raw_mm(G, G0, K, Rmax):
    """Shifted Michaelis-Menten forward model: conductance → raw ADC counts.

    Raw = Rmax * max(G - G0, 0) / (max(G - G0, 0) + K)

    Parameters
    ----------
    G : float or np.ndarray
        True conductance [µS].
    G0 : float
        Onset conductance — dead zone threshold [µS].
    K : float
        Half-saturation conductance after onset [µS].
    Rmax : float
        Saturation level [ADC counts].

    Returns
    -------
    float or np.ndarray
        Predicted raw ADC counts.
    """""
    dG = np.maximum(G - G0, 0.0)
    return Rmax * dG / (dG + K)


def get_inverse_mm(raw, G0, K, Rmax):
    """Shifted Michaelis-Menten inverse model: raw ADC counts → conductance.

    G = G0 + K * Raw / (Rmax - Raw)

    Parameters
    ----------
    raw : float or np.ndarray
        Raw ADC counts.
    G0 : float
        Onset conductance [µS].
    K : float
        Half-saturation conductance after onset [µS].
    Rmax : float
        Saturation level [ADC counts].

    Returns
    -------
    float or np.ndarray
        Predicted true conductance [µS].
    """""
    return G0 + K * raw / (Rmax - raw)


def get_error_at_raw_mm(raw, K, Rmax, rmse):
    """Compute propagated conductance error σ_G at a given raw ADC value.

    Derived from error propagation through the inverse of the Michaelis-Menten model:
    σ_G = dG/dRaw * RMSE = K * Rmax * RMSE / (Rmax - Raw)²

    Parameters
    ----------
    raw : float or np.ndarray
        Raw ADC counts. Must be strictly less than Rmax.
    K : float
        Half-saturation conductance after onset [µS].
    Rmax : float
        Saturation level [ADC counts].
    rmse : float
        Root mean squared error of the MM fit [ADC counts].

    Returns
    -------
    float or np.ndarray
        Propagated conductance error σ_G [µS].
    """""
    return K * Rmax * rmse / (Rmax - raw) ** 2


def get_phys_max(df):
    phys_max_map = {}

    for dac, grp in df.groupby('DAC'):
        grp = grp.sort_values('True_Conductance')
        G_arr = grp['True_Conductance'].to_numpy()
        phys_max_map[dac] = G_arr[-1]

    return phys_max_map


def get_curve_onsets(df, get_next_sample=True):
    onset_map = {}
    for dac, grp in df.groupby('DAC'):
        grp = grp.sort_values('True_Conductance')
        G_arr = grp['True_Conductance'].to_numpy()
        raw_arr = grp['Mean_Raw'].fillna(0).to_numpy()
        idx = first_raise_point_idx(G_arr, raw_arr, get_next_sample=get_next_sample)
        onset_map[dac] = G_arr[idx]

    return onset_map


def first_raise_point_idx(G, raw, get_next_sample=True):
    """Find the index of the first point where the curve starts rising.

    Uses the slope (finite differences) to detect the onset: the first
    point where dRaw/dG exceeds 1% of the peak slope.

    Parameters
    ----------
    G : np.ndarray
        Conductance values [µS], sorted ascending.
    raw : np.ndarray
        Corresponding raw ADC values.
    get_next_sample : bool, optional
        If True, returns idx+1 (first rising point). If False, returns idx
        (last flat point before the rise). Default is True.

    Returns
    -------
    int
        Index into G and raw. Returns 0 if no rising region is detected.
    """""
    dydx = np.diff(raw) / np.diff(G)
    threshold = 0.01 * np.max(dydx)
    rising = np.where(dydx > threshold)[0]
    if len(rising) == 0:
        return 0
    idx = rising[0]
    if get_next_sample:
        idx = min(idx + 1, len(G) - 1)
    return idx


def sigma_at_raw(raw, K, Rmax, rmse):
    """σ_G = K * Rmax * RMSE / (Rmax - Raw)²"""
    return K * Rmax * rmse / (Rmax - raw) ** 2


def raw_at_sigma(sigma, K, Rmax, rmse):
    """Raw = Rmax - sqrt(K * Rmax * RMSE / sigma)"""
    return Rmax - np.sqrt(K * Rmax * rmse / sigma)


def raw_to_conductance(eda_signal, dac_signal, df_coeffs, crop_leading_nans=True):
    """Convert a raw EDA signal to conductance [µS] using per-DAC MM coefficients.

    Segments the signal by DAC transitions and applies the shifted MM inverse
    function to each segment using the corresponding calibration coefficients.

    Parameters
    ----------
    eda_signal : np.ndarray
        Raw ADC values, shape (N,).
    dac_signal : np.ndarray
        DAC integer values, same shape as eda_signal.
    df_coeffs : pd.DataFrame
        Coefficient table with columns DAC, G0, K, Rmax.
    crop_leading_nans : bool, default True
        If True, discard everything up to and including the last NaN, so the
        returned signal is NaN-free. Note this removes ALL NaNs, including any
        occurring mid-recording, not just a leading block.

    Returns
    -------
    G_signal : np.ndarray
        Conductance signal in µS. Length N if crop_leading_nans is False,
        otherwise N - crop_idx.
    crop_idx : int
        Index of the first returned sample in the original signal. 0 when no
        NaNs were present or cropping is disabled. Use to trim companion
        signals: dac_sym = dac_sym[crop_idx:].
    """
    coeff_map = {
        int(row['DAC']): (row['G0'], row['K'], row['Rmax'])
        for _, row in df_coeffs.iterrows()
    }

    # build nan array
    G_signal = np.full_like(eda_signal, np.nan, dtype=float)

    # identify ranges
    transitions = np.where(np.diff(dac_signal) != 0)[0] + 1
    starts = np.concatenate([[0], transitions])
    ends = np.concatenate([transitions, [len(dac_signal)]])

    # iterate through the DAC ranges and apply conversion there
    for start, end in zip(starts, ends):
        dac_val = int(dac_signal[start])

        # no coefficients for this DAC -> leave segment = NaN
        if dac_val not in coeff_map:
            continue

        G0, K, Rmax = coeff_map[dac_val]
        raw_chunk = eda_signal[start:end].astype(float)
        G_signal[start:end] = G0 + K * raw_chunk / (Rmax - raw_chunk)

    # if enabled, crop from last nan idx onwards
    nan_idx = np.where(np.isnan(G_signal))[0]

    # if there is at least one NaN, chose the last position
    if nan_idx.size:
        crop_idx = int(nan_idx[-1]) + 1
    else:
        crop_idx = 0
    # then crop the signal from that onwards
    if crop_leading_nans:
            G_signal = G_signal[crop_idx:]

    return G_signal, crop_idx


def sensitivity_mm(G, G0, K, Rmax):
    """dRAW/dG of the shifted MM model [ADC / µS]."""
    denom = (G - G0 + K)
    return Rmax * K / denom ** 2


def raw_to_sigma(eda_signal, dac_signal, df_coeffs):
    coeff_map = {
        int(row['DAC']): (row['K'], row['Rmax'], row['RMSE'])
        for _, row in df_coeffs.iterrows()
    }
    sigma_signal = np.full_like(eda_signal, np.nan, dtype=float)
    transitions = np.where(np.diff(dac_signal) != 0)[0] + 1
    starts = np.concatenate([[0], transitions])
    ends = np.concatenate([transitions, [len(dac_signal)]])
    for start, end in zip(starts, ends):
        dac_val = int(dac_signal[start])
        if dac_val not in coeff_map:
            continue
        K, Rmax, rmse = coeff_map[dac_val]
        sigma_signal[start:end] = get_error_at_raw_mm(
            eda_signal[start:end].astype(float), K, Rmax, rmse
        )
    return sigma_signal
