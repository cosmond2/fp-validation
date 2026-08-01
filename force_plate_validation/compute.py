"""Pure computation functions for validation metrics."""
import numpy as np
import pandas as pd
from .config import LBF_TO_N, IN_TO_MM, FT_TO_IN
from .calibration import convert_volt_to_mv
from .signal_processing import butterworth_filter, compute_fft_noise_signature


def compute_channel_stats(df, columns, units="", data_source="raw"):
    """Generic per-channel descriptive stats."""
    rows = []
    for col in columns:
        signal = df[col].values
        mean = np.mean(signal)
        std = np.std(signal, ddof=1)
        rms = np.sqrt(np.mean((signal - mean) ** 2))
        ptp = np.ptp(signal)
        rows.append({
            "Channel": col,
            "Mean": mean,
            "Std Dev": std,
            "RMS": rms,
            "Peak-to-Peak": ptp,
            "Units": units,
            "Data Source": data_source,
        })
    return pd.DataFrame(rows)


def compute_static_noise(raw_df, force_df):
    """Unloaded noise test metrics."""
    mv_df = convert_volt_to_mv(raw_df)
    raw_noise = compute_channel_stats(
        mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], units="mV"
    )
    force_noise = compute_channel_stats(
        force_df, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], units="lbf / lbf-in"
    )
    return raw_noise, force_noise


def compute_drift(force_df, n_samples=1000):
    """Static load drift test metrics."""
    fz = force_df['Fz'].values
    initial_lbf = np.mean(fz[:n_samples])
    final_lbf = np.mean(fz[-n_samples:])
    drift_lbf = final_lbf - initial_lbf
    return pd.DataFrame([{
        "Initial Fz (lbf)": initial_lbf,
        "Final Fz (lbf)": final_lbf,
        "Drift (lbf)": drift_lbf,
        "Initial Fz (N)": initial_lbf * LBF_TO_N,
        "Final Fz (N)": final_lbf * LBF_TO_N,
        "Drift (N)": drift_lbf * LBF_TO_N,
    }])


def compute_corner_loading(force_df, moment_units, z_offset_in=0.0):
    """Corner loading test metrics."""
    fx_lbf = np.mean(force_df['Fx'].values)
    fy_lbf = np.mean(force_df['Fy'].values)
    fz_lbf = np.mean(force_df['Fz'].values)
    mx = np.mean(force_df['Mx'].values)
    my = np.mean(force_df['My'].values)
    
    if moment_units == 'lbf-ft':
        mx_in = mx * FT_TO_IN
        my_in = my * FT_TO_IN
    else:
        mx_in = mx
        my_in = my
    
    cop_x_in = (-my_in + (fx_lbf * z_offset_in)) / fz_lbf
    cop_y_in = (mx_in + (fy_lbf * z_offset_in)) / fz_lbf
    
    return pd.DataFrame([{
        "Fx (N)": fx_lbf * LBF_TO_N,
        "Fy (N)": fy_lbf * LBF_TO_N,
        "Fz (N)": fz_lbf * LBF_TO_N,
        "CoP X (mm)": cop_x_in * IN_TO_MM,
        "CoP Y (mm)": cop_y_in * IN_TO_MM,
        "Fx (lbf)": fx_lbf,
        "Fy (lbf)": fy_lbf,
        "Fz (lbf)": fz_lbf,
    }])


def compute_phase3_center_drift_channel_metrics(raw_df, force_df, fs=1000.0, cutoff_hz=20.0):
    """Summarize per-channel metrics for Phase 3 center-drift case."""
    raw_mv_df = convert_volt_to_mv(raw_df)
    filtered_mv_df = butterworth_filter(
        raw_mv_df[['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']], 
        fs=fs, cutoff_hz=cutoff_hz
    )
    
    raw_stats = compute_channel_stats(
        raw_mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], 
        units='mV', data_source='raw'
    )
    filtered_stats = compute_channel_stats(
        filtered_mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], 
        units='mV', data_source='filtered'
    )
    stats = pd.concat([raw_stats, filtered_stats], ignore_index=True)
    
    if 'Fx' in force_df.columns:
        force_stats = compute_channel_stats(
            force_df, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], 
            units='lbf / lbf-in', data_source='raw'
        )
        force_filtered = force_df.copy()
        for col in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
            force_filtered[col] = butterworth_filter(
                force_filtered[col].values, fs=fs, cutoff_hz=cutoff_hz
            )
        force_filtered_stats = compute_channel_stats(
            force_filtered, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], 
            units='lbf / lbf-in', data_source='filtered'
        )
        stats = pd.concat([stats, force_stats, force_filtered_stats], ignore_index=True)
    
    stats['Std Dev Relative to Median'] = (
        stats['Std Dev'] / np.median(stats[stats['Data Source'] == 'raw']['Std Dev'])
    )
    stats['RMS Relative to Median'] = (
        stats['RMS'] / np.median(stats[stats['Data Source'] == 'raw']['RMS'])
    )
    return stats


def summarize_loaded_fft_noise(raw_df, channel_columns, fs=1000.0, cutoff_hz=20.0):
    """Summarize FFT noise signatures for loaded-case data."""
    rows = []
    for channel in channel_columns:
        raw_signal = raw_df[channel].values
        filtered_signal = butterworth_filter(raw_signal, fs=fs, cutoff_hz=cutoff_hz)
        raw_summary = compute_fft_noise_signature(raw_signal, fs=fs)
        filtered_summary = compute_fft_noise_signature(filtered_signal, fs=fs)
        rows.append({
            "Channel": channel,
            "FFT line-noise share (raw, %)": raw_summary["line_noise_share_pct"],
            "FFT high-freq share (raw, %)": raw_summary["high_freq_share_pct"],
            "FFT line-noise share (filtered, %)": filtered_summary["line_noise_share_pct"],
            "FFT high-freq share (filtered, %)": filtered_summary["high_freq_share_pct"],
            "FFT noise reduction (%)": (
                raw_summary["line_noise_share_pct"] - filtered_summary["line_noise_share_pct"]
            ),
        })
    return pd.DataFrame(rows)


def compare_center_corner_fx_fy(center_force_df, corner_force_df, applied_load_lbs=50.0):
    """Check Fx/Fy consistency between center and corner loading."""
    true_load_n = applied_load_lbs * LBF_TO_N
    crosstalk_tol_n = 0.002 * true_load_n
    
    center_mean = center_force_df[['Fx', 'Fy']].mean()
    corner_mean = corner_force_df[['Fx', 'Fy']].mean()
    
    rows = [{
        'Metric': 'Fx',
        'Center mean (N)': center_mean['Fx'] * LBF_TO_N,
        'Corner mean (N)': corner_mean['Fx'] * LBF_TO_N,
        'Difference (N)': (corner_mean['Fx'] - center_mean['Fx']) * LBF_TO_N,
        'Tolerance (N)': crosstalk_tol_n,
        'Approximately same': abs((corner_mean['Fx'] - center_mean['Fx']) * LBF_TO_N) <= crosstalk_tol_n,
    }, {
        'Metric': 'Fy',
        'Center mean (N)': center_mean['Fy'] * LBF_TO_N,
        'Corner mean (N)': corner_mean['Fy'] * LBF_TO_N,
        'Difference (N)': (corner_mean['Fy'] - center_mean['Fy']) * LBF_TO_N,
        'Tolerance (N)': crosstalk_tol_n,
        'Approximately same': abs((corner_mean['Fy'] - center_mean['Fy']) * LBF_TO_N) <= crosstalk_tol_n,
    }]
    return pd.DataFrame(rows)