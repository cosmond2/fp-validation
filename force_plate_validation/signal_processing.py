"""Signal processing functions: filtering, FFT analysis."""
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt


def butterworth_filter(signal, fs=1000.0, cutoff_hz=20.0, order=4, btype="low"):
    """Apply zero-phase Butterworth filter to signal or DataFrame."""
    if isinstance(signal, pd.DataFrame):
        filtered_df = signal.copy()
        for col in filtered_df.columns:
            filtered_df[col] = butterworth_filter(
                filtered_df[col].values, fs=fs, cutoff_hz=cutoff_hz,
                order=order, btype=btype
            )
        return filtered_df
    
    if isinstance(signal, pd.Series):
        return pd.Series(
            butterworth_filter(signal.values, fs=fs, cutoff_hz=cutoff_hz,
                             order=order, btype=btype),
            index=signal.index, name=signal.name
        )
    
    signal_array = np.asarray(signal, dtype=float)
    if signal_array.ndim == 0 or signal_array.size < 3:
        return signal_array.copy()
    
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal_array.copy()
    
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        return signal_array.copy()
    
    sos = butter(order, cutoff_hz / nyquist, btype=btype, output="sos")
    return sosfiltfilt(sos, signal_array)


def compute_fft_noise_signature(signal, fs=1000.0, high_freq_hz=20.0, line_band_hz=10.0):
    """Return line-noise and high-frequency power ratios for a signal."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1.0 / fs)[:n // 2]
    power = np.abs(yf[:n // 2]) ** 2 / n
    
    total_power = np.sum(power)
    if total_power == 0:
        return {
            "line_noise_share_pct": np.nan,
            "high_freq_share_pct": np.nan,
            "line_noise_power": np.nan,
            "high_freq_power": np.nan,
        }
    
    line_mask = (xf >= 50.0 - line_band_hz) & (xf <= 50.0 + line_band_hz)
    high_mask = xf > high_freq_hz
    
    line_power = np.sum(power[line_mask]) if np.any(line_mask) else 0.0
    high_freq_power = np.sum(power[high_mask]) if np.any(high_mask) else 0.0
    
    return {
        "line_noise_share_pct": 100.0 * line_power / total_power,
        "high_freq_share_pct": 100.0 * high_freq_power / total_power,
        "line_noise_power": line_power,
        "high_freq_power": high_freq_power,
    }