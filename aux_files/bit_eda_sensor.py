import numpy as np

def bitalino_eda_to_us(adc, n_bits=10, vcc=3.3):
    """BITalino EDA transfer function: raw ADC -> microsiemens. Range [0, 25] uS."""
    adc = np.asarray(adc, dtype=float)
    return (adc / 2 ** n_bits) * vcc / 0.132

def bitalino_eda_to_siemens(adc, n_bits=10, vcc=3.3):
    return bitalino_eda_to_us(adc, n_bits, vcc) * 1e-6