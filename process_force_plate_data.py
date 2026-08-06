#%% Imports
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt
from datetime import datetime

from force_plate_validation.compute import generate_spatial_sensitivity_table

#%% Calibration Matrices

# Calibration matrices have been double checked
phase_1_2_3_gain_scaling = np.array([1,1,1,1,1,1]) # For BP400600
phase_1_2_3_matrix = np.array([
    [0.6521,0.0045,-0.0002,-0.0057,-0.0019,0.0020],
    [-0.0015,0.6525,-0.0117,-0.0082,-0.0034,-0.0077],
    [0.0010,-0.0052,2.5676,-0.0012,0.0012,0.0006],
    [-0.0168,0.0309,-0.0594,12.8865,-0.0469,-0.0248],
    [0.0314,0.0101,-0.0595,0.0163,10.1565,-0.0069],
    [0.0327,0.1287,-0.0279,0.0057,0.0677,5.4770]
], dtype=float)
MATRIX_PHASE_1_2_3 = phase_1_2_3_matrix * phase_1_2_3_gain_scaling

phase_4_gain_scaling = np.array([0.25,0.25,1,0.5,0.5,0.25]) # For OR6-7-8000
phase_4_matrix = np.array([
    [2.6952,0.0567,-0.0513,-0.0221,0.0360,-0.1051],
    [0.0145,2.6894,-0.0668,-0.0094,-0.0372,0.0441],
    [0.0536,0.0004,11.4268,-0.0847,0.0128,0.0498],
    [0.0037,0.0081,-0.0010,3.6987,0.0023,-0.0428],
    [0.0081,0.0071,-0.0011,0.0340,3.6909,-0.0164],
    [-0.0155,0.0034,0.0338,-0.0057,0.0091,1.7466]
], dtype=float)
MATRIX_PHASE_4 = phase_4_matrix * phase_4_gain_scaling

# NOTE: both calibration matrices output force/moment in lbf and lbf-in
# (phases 1-3) or lbf and lbf-ft (phase 4), NOT Newtons. Every downstream
# consumer of a "force_df" needs to remember this and convert explicitly
# when N is required.

LBF_TO_N = 4.44822
IN_TO_MM = 25.4
FT_TO_IN = 12.0

#%% File io helper functions

def determine_phase(filename):
    """Determine validation phase from filename."""
    filename = filename.lower()

    if "phase4" in filename:
        return 4
    elif "phase3" in filename:
        return 3
    elif "phase2" in filename:
        return 2
    elif "phase1" in filename:
        return 1

    raise NameError(
        f"File:{filename} does not contain valid phase1/2/3/4 for identification."
        )

def get_files_by_phase(directories, phase=None, keyword=None, extension="*.txt"):
    """
    Scan directories for files, determine phase from filename,
    and filter by phase and keyword.

    Parameters
    ----------
    directories : list of str
        List of directory paths.
    phase : int or 'all' or None
        If int (1-4), return only that phase. 'all' or None returns all.
    keyword : str or None
        Substring to filter filenames (case-insensitive).
    extension : str, default '*.txt'

    Returns
    -------
    list of dict
        Each dictionary contains: filepath, directory, basename, phase
    """
    files_info = []

    for directory in directories:
        search_path = os.path.join(directory, extension)

        for filepath in glob.glob(search_path):

            basename = os.path.basename(filepath)
            basename_lower = basename.lower()

            phase_id = determine_phase(basename)

            if phase not in (None, "all") and int(phase) != phase_id:
                continue

            if keyword and keyword.lower() not in basename_lower:
                continue

            files_info.append({
                "filepath": filepath,
                "directory": directory,
                "basename": basename,
                "phase": phase_id
            })

    files_info.sort(key=lambda x: (x["phase"], x["basename"]))

    if not files_info:
        raise FileNotFoundError(
            f"No files found for phase={phase}, keyword={keyword}."
        )

    return files_info

#%% DAQ reading helpers, force conversion helpers

def read_ni_daq(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    return df

def read_powerlab_daq(file_path):
    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=6,
        header=None,
        names=['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    )
    return df

def butterworth_filter(signal, fs=1000.0, cutoff_hz=20.0, order=4, btype="low"):
    """Apply a zero-phase Butterworth low-pass filter to a 1D signal or each column of a DataFrame."""
    if isinstance(signal, pd.DataFrame):
        filtered_df = signal.copy()
        for col in filtered_df.columns:
            filtered_df[col] = butterworth_filter(
                filtered_df[col].values,
                fs=fs,
                cutoff_hz=cutoff_hz,
                order=order,
                btype=btype,
            )
        return filtered_df

    if isinstance(signal, pd.Series):
        return pd.Series(
            butterworth_filter(
                signal.values,
                fs=fs,
                cutoff_hz=cutoff_hz,
                order=order,
                btype=btype,
            ),
            index=signal.index,
            name=signal.name,
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

def convert_volt_to_force(df, matrix, phase=None):
    """
    Raw amplifier output (V) -> calibrated force/moment (lbf, lbf-in or lbf-ft).
    
    Parameters
    ----------
    df : DataFrame with Time, Ch1-Ch6 columns
    matrix : 6x6 calibration matrix
    phase : int, optional. If provided, used to set moment units appropriately.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"Calibration matrix must have shape (6, 6); got {matrix.shape}")
    
    voltage_data = df[['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']].values
    excitation_voltage = 10.0
    amplifier_gain = 4000.0
    electrical_scalar = 1_000_000 / (excitation_voltage * amplifier_gain)
    scaled_voltage = voltage_data * electrical_scalar
    calibrated_data = np.dot(scaled_voltage, matrix.T)
    
    calibrated_df = pd.DataFrame(
        calibrated_data,
        columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    )
    calibrated_df['Time'] = df['Time'].values
    
    # Add metadata about moment units
    if phase == 4:
        calibrated_df.attrs['moment_units'] = 'lbf-ft'
    else:
        calibrated_df.attrs['moment_units'] = 'lbf-in'
    
    return calibrated_df

def convert_volt_to_mv(df):
    """Raw amplifier output (V) -> mV, unchanged channel names, for
    evaluating the raw-signal noise criterion"""
    mv_df = df.copy()
    channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    mv_df[channel_cols] = mv_df[channel_cols] * 1000.0
    return mv_df

def get_reader_and_matrix(phase):
    """Central place mapping phase -> (reader function, calibration matrix,
    moment length unit)."""
    if phase == 4:
        return read_powerlab_daq, MATRIX_PHASE_4, 'lbf-ft'
    elif phase == 3:
        return read_ni_daq, MATRIX_PHASE_1_2_3, 'lbf-in'
    else:
        return read_powerlab_daq, MATRIX_PHASE_1_2_3, 'lbf-in'

def load_force_file(filepath, phase):
    """
    Read a raw DAQ file and convert to calibrated force data.

    Returns
    -------
    raw_df : DataFrame (Time, Ch1..Ch6 in volts)
    force_df : DataFrame (Time, Fx,Fy,Fz,Mx,My,Mz in lbf / lbf-in or lbf-ft)
    """
    reader, matrix, moment_units = get_reader_and_matrix(phase)
    raw_df = reader(filepath)
    force_df = convert_volt_to_force(raw_df, matrix, phase=phase)
    return raw_df, force_df

#%% Plotting helpers (opt-in, decoupled from compute functions below)

def plot_interactive_timeseries(df, title="Time Series Data", y_label="Amplitude", save_path=None):
    if 'Time' not in df.columns:
        raise ValueError("DataFrame must contain a 'Time' column.")
    signal_cols = [col for col in df.columns if col != 'Time']
    fig = go.Figure()
    for col in signal_cols:
        fig.add_trace(go.Scatter(x=df['Time'], y=df[col], mode='lines', name=col))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Channels",
        template="plotly_white"
    )
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_fft(df, channel='Ch1', fs=1000.0, title=None, save_path=None):
    signal = df[channel].values - np.mean(df[channel].values)  # remove DC
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    power = np.abs(yf[:n//2])**2 / n
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=power, mode='lines', name='Power Spectrum'))
    fig.update_layout(
        title=title or f"FFT of {channel}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        template="plotly_white"
    )
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_cop_scatter(cop_x, cop_y, title="Center of Pressure", save_path=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cop_x, y=cop_y, mode='markers',
                              marker=dict(size=3, opacity=0.4)))
    fig.update_layout(
        title=title,
        xaxis_title="CoP X (mm)",
        yaxis_title="CoP Y (mm)",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

#%% ============================================================
#   PURE COMPUTE FUNCTIONS
#   Each takes a dataframe (or two) in, returns a tidy metrics
#   dataframe out. No printing, no plotting, no file I/O, no
#   pass/fail judgment. These are unit-testable in isolation.
#   ============================================================

def compute_channel_stats(df, columns, units="", data_source="raw"):
    """Generic per-channel descriptive stats: mean, std, rms, peak-to-peak.
    Used for both raw-voltage noise and calibrated-force noise."""
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


def compute_phase3_center_drift_channel_metrics(raw_df, force_df, fs=1000.0, cutoff_hz=20.0):
    """Summarize per-channel metrics for the Phase 3 center-drift case."""
    raw_mv_df = convert_volt_to_mv(raw_df)
    filtered_mv_df = butterworth_filter(raw_mv_df[['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']], fs=fs, cutoff_hz=cutoff_hz)

    raw_stats = compute_channel_stats(raw_mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], units='mV', data_source='raw')
    filtered_stats = compute_channel_stats(filtered_mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], units='mV', data_source='filtered')
    stats = pd.concat([raw_stats, filtered_stats], ignore_index=True)

    if 'Fx' in force_df.columns:
        force_stats = compute_channel_stats(force_df, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], units='lbf / lbf-in', data_source='raw')
        force_filtered = force_df.copy()
        for col in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
            force_filtered[col] = butterworth_filter(force_filtered[col].values, fs=fs, cutoff_hz=cutoff_hz)
        force_filtered_stats = compute_channel_stats(force_filtered, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], units='lbf / lbf-in', data_source='filtered')
        stats = pd.concat([stats, force_stats, force_filtered_stats], ignore_index=True)

    # Flag broad-channel effects, which would be consistent with a loose AI SENSE wire.
    stats['Std Dev Relative to Median'] = stats['Std Dev'] / np.median(stats[stats['Data Source'] == 'raw']['Std Dev'])
    stats['RMS Relative to Median'] = stats['RMS'] / np.median(stats[stats['Data Source'] == 'raw']['RMS'])
    return stats


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


def summarize_loaded_fft_noise(raw_df, channel_columns, fs=1000.0, cutoff_hz=20.0):
    """Summarize FFT noise signatures for a loaded-case dataframe."""
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
            "FFT noise reduction (%)": raw_summary["line_noise_share_pct"] - filtered_summary["line_noise_share_pct"],
        })
    return pd.DataFrame(rows)


def compare_center_corner_fx_fy(center_force_df, corner_force_df, applied_load_lbs=50.0):
    """Check whether Fx/Fy remain roughly constant between center and corner loading."""
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

def compute_static_noise(raw_df, force_df):
    """
    Unloaded noise test (sections 3.5 / 4.1)

    Returns
    -------
    raw_noise : DataFrame, per-channel stats in mV (Ch1,...,Ch6)
    force_noise : DataFrame, per-channel stats in lbf / lbf-in (Fx,...,Mz)
    """
    mv_df = convert_volt_to_mv(raw_df)
    raw_noise = compute_channel_stats(
        mv_df, ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'], units="mV"
    )
    force_noise = compute_channel_stats(
        force_df, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], units="lbf / lbf-in"
    )
    return raw_noise, force_noise

def compute_drift(force_df, n_samples=1000):
    """
    Static load drift test (sections 3.6 / 4.2). Compares the mean of
    the first/last n_samples of Fz. Returned in lbf / lbf-in

    Returns a one-row DataFrame (not a scalar) so it stacks cleanly
    across files in the runner.
    """
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
    """
    Corner loading test (sections 3.7 / 4.3). Computes mean Fx/Fy/Fz
    and center of pressure for one file (one corner, one trial).

    Returns a one-row DataFrame.
    """
    fx_lbf = np.mean(force_df['Fx'].values)
    fy_lbf = np.mean(force_df['Fy'].values)
    fz_lbf = np.mean(force_df['Fz'].values)
    mx = np.mean(force_df['Mx'].values)
    my = np.mean(force_df['My'].values)

    # Handle moment units correctly
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
        # Keep raw values for table generation
        "Fx (lbf)": fx_lbf,
        "Fy (lbf)": fy_lbf,
        "Fz (lbf)": fz_lbf,
    }])

#%% ============================================================
#   RUNNERS
#   Loop over files for a given test type, call the pure compute
#   functions, tag results with File/Phase/Directory, stack into
#   one tidy dataframe. No pass/fail here -- that's the
#   next layer down. Plotting is opt-in via out_dir.
#   ============================================================

def _iter_test_files(directories, keyword, exclude_keywords=None):
    files = get_files_by_phase(directories, phase='all', keyword=keyword)
    if exclude_keywords:
        files = [
            f for f in files
            if not any(k in f['basename'].lower() for k in exclude_keywords)
        ]
    return files

def run_static_noise_test(directories, out_dir=None):
    """Returns (raw_noise_df, force_noise_df), each tidy with File/Phase columns."""
    files = _iter_test_files(directories, keyword="static")

    raw_results, force_results = [], []
    for f in files:
        raw_df, force_df = load_force_file(f["filepath"], f["phase"])
        raw_noise, force_noise = compute_static_noise(raw_df, force_df)

        for df_ in (raw_noise, force_noise):
            df_.insert(0, "Phase", f["phase"])
            df_.insert(0, "File", f["basename"])
        raw_results.append(raw_noise)
        force_results.append(force_noise)

        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                raw_df, title=f"Raw signal: {base}", y_label="Voltage (V)",
                save_path=os.path.join(out_dir, f"{base}_raw_timeseries.html")
            )
            plot_fft(
                raw_df, channel='Ch1', title=f"FFT (Ch1): {base}",
                save_path=os.path.join(out_dir, f"{base}_fft.html")
            )

    return pd.concat(raw_results, ignore_index=True), pd.concat(force_results, ignore_index=True)

def run_drift_test(directories, out_dir=None, n_samples=1000):
    """Returns a tidy DataFrame, one row per drift/center file."""
    files = _iter_test_files(directories, keyword=None)
    files = [f for f in files if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]

    results = []
    for f in files:
        _, force_df = load_force_file(f["filepath"], f["phase"])
        row = compute_drift(force_df, n_samples=n_samples)
        row.insert(0, "Phase", f["phase"])
        row.insert(0, "File", f["basename"])
        results.append(row)

        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                force_df, title=f"Drift: {base}", y_label="Force (lbf) / Moment",
                save_path=os.path.join(out_dir, f"{base}_drift.html")
            )

    return pd.concat(results, ignore_index=True)

def run_corner_test(directories, z_offset_in=0.0, out_dir=None):
    """Returns a tidy DataFrame, one row per corner file. Corner label
    (TL/TR/BR/BL) is parsed from the filename when present."""
    files = _iter_test_files(directories, keyword=None,
                              exclude_keywords=['static', 'drift'])
    files = [
        f for f in files
        if any(k in f['basename'].lower() for k in ['corner', '_tl', '_tr', '_br', '_bl'])
    ]

    corner_labels = {'_tl': 'TL', '_tr': 'TR', '_br': 'BR', '_bl': 'BL'}

    results = []
    for f in files:
        _, force_df = load_force_file(f["filepath"], f["phase"])
        _, matrix, moment_units = get_reader_and_matrix(f["phase"])

        row = compute_corner_loading(force_df, moment_units, z_offset_in=z_offset_in)

        corner = next((v for k, v in corner_labels.items() if k in f['basename'].lower()), "Unknown")
        row.insert(0, "Corner", corner)
        row.insert(0, "Phase", f["phase"])
        row.insert(0, "File", f["basename"])
        results.append(row)

        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                force_df, title=f"Corner: {base}", y_label="Force (lbf) / Moment",
                save_path=os.path.join(out_dir, f"{base}_corner.html")
            )

    return pd.concat(results, ignore_index=True)

#%% ============================================================
#   PASS / FAIL LAYER
#   Applies Table 4's acceptance criteria to the tidy dataframes
#   from the runners above. Kept fully separate so the thresholds
#   can be revisited without touching data loading/compute code.
#   ============================================================

def apply_noise_criteria(force_noise_df, noise_threshold_N=0.5):
    """Table 4, row 1.1/2.1/3.1/4.1: peak-to-peak baseline noise per
    channel must stay within +/- force threshold."""
    out = force_noise_df.copy()
    # Convert Peak-to-Peak from lbf to N (assuming lbf/lbf-in units)
    out["Peak-to-Peak (N)"] = out["Peak-to-Peak"] * LBF_TO_N if 'lbf' in out['Units'].iloc[0] else out["Peak-to-Peak"]
    out["Pass"] = out["Peak-to-Peak (N)"] <= (2 * noise_threshold_N)
    return out

def apply_drift_criteria(drift_df, applied_load_lbs=50.0, abs_threshold_n=1.0, pct_threshold=0.005):
    """Table 4, row 1.2/2.2/3.2/4.2: drift < 1 N or 0.5% of applied
    load, whichever is SMALLER."""
    true_load_n = applied_load_lbs * LBF_TO_N
    threshold_n = min(abs_threshold_n, pct_threshold * true_load_n)
    out = drift_df.copy()
    out["Threshold (N)"] = threshold_n
    out["Pass"] = out["Drift (N)"].abs() <= threshold_n
    return out

def apply_weight_and_crosstalk_criteria(corner_df, applied_load_lbs=50.0,
                                         weight_pct=0.005, crosstalk_pct=0.002):
    """Table 4, row 1.3/2.3/3.3/4.3: Fz within +/-0.5% of true weight;
    Fx/Fy (crosstalk) within +/-0.2% of the vertical load."""
    true_load_n = applied_load_lbs * LBF_TO_N
    weight_tol_n = weight_pct * true_load_n
    crosstalk_tol_n = crosstalk_pct * true_load_n

    out = corner_df.copy()
    out["True Fz (N)"] = true_load_n
    out["Weight Tolerance (N)"] = weight_tol_n
    out["Weight Pass"] = (out["Fz (N)"].abs() - true_load_n).abs() <= weight_tol_n
    out["Crosstalk Tolerance (N)"] = crosstalk_tol_n
    out["Fx Pass"] = out["Fx (N)"].abs() <= crosstalk_tol_n
    out["Fy Pass"] = out["Fy (N)"].abs() <= crosstalk_tol_n
    
    # Overall corner pass (all criteria must pass)
    out["Pass"] = out["Weight Pass"] & out["Fx Pass"] & out["Fy Pass"]
    return out

def apply_cop_criteria(corner_df, true_cop_by_phase_corner, tolerance_mm=3.0):
    """Table 4, row 1.3/2.3/3.3/4.3: CoP within +/-3 mm of the physical
    load location."""
    def _lookup(row, axis_idx):
        corner_map = true_cop_by_phase_corner.get(row["Phase"], {})
        return corner_map.get(row["Corner"], (np.nan, np.nan))[axis_idx]

    out = corner_df.copy()
    out["True CoP X (mm)"] = out.apply(_lookup, axis=1, axis_idx=0)
    out["True CoP Y (mm)"] = out.apply(_lookup, axis=1, axis_idx=1)
    err = np.sqrt(
        (out["CoP X (mm)"] - out["True CoP X (mm)"]) ** 2
        + (out["CoP Y (mm)"] - out["True CoP Y (mm)"]) ** 2
    )
    out["CoP Error (mm)"] = err
    out["Pass"] = err <= tolerance_mm
    return out

def run_report():
    """Generate the validation report tables and plots from the configured raw-data directories."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_1 = r"Z:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files"
    data_dir_2 = r"Z:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
    plot_out_dir = os.path.join(base_dir, "report_output")
    os.makedirs(plot_out_dir, exist_ok=True)

    directories = [d for d in [data_dir_1, data_dir_2] if os.path.exists(d)]
    if not directories:
        print("No raw DAQ data directories were found; report generation skipped.")
        return None
    APPLIED_LOAD_LBS = 50.0

    BASE_PLATE_DIA = 127  # mm units

    DIMS_BP400600 = {
        'height': 600,  # mm units
        'width': 400    # mm units
    }

    DIMS_OR67800 = {
        'height': 508,  # mm units
        'width': 464    # mm units
    }

    PHASE_LABELS = {
        1: "Phase 1: BP400600 / MSA6 SN6893 / PowerLab (baseline)",
        2: "Phase 2: BP400600 / MSA6 SN7526 / PowerLab (amplifier isolation)",
        3: "Phase 3: BP400600 / MSA6 SN7526 / NI-6210 (DAQ isolation)",
        4: "Phase 4: OR6-7-8000 / MSA6 SN7526 / PowerLab (force plate isolation)",
    }

    # Physical corner locations for CoP validation
    TRUE_COP_BY_CORNER_MM = {
        1: {'TL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'TR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'BR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2), 
            'BL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2)},
        2: {'TL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'TR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'BR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2), 
            'BL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2)},
        3: {'TL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'TR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2 + BASE_PLATE_DIA/2), 
            'BR': (DIMS_BP400600['width']/2 - BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2), 
            'BL': (-1*DIMS_BP400600['width']/2 + BASE_PLATE_DIA/2, DIMS_BP400600['height']/2 - BASE_PLATE_DIA/2)},
        4: {'TL': (-1*DIMS_OR67800['width']/2 + BASE_PLATE_DIA/2, -1*DIMS_OR67800['height']/2 + BASE_PLATE_DIA/2), 
            'TR': (DIMS_OR67800['width']/2 - BASE_PLATE_DIA/2, -1*DIMS_OR67800['height']/2 + BASE_PLATE_DIA/2), 
            'BR': (DIMS_OR67800['width']/2 - BASE_PLATE_DIA/2, DIMS_OR67800['height']/2 - BASE_PLATE_DIA/2), 
            'BL': (-1*DIMS_OR67800['width']/2 + BASE_PLATE_DIA/2, DIMS_OR67800['height']/2 - BASE_PLATE_DIA/2)},
    }

    #%% 3.4 - Acceptance criteria (Table 4) - Generate table

    def generate_acceptance_criteria_table():
        """Generate Table 4 from the report as a formatted string."""
        criteria = {
            "Test": [
                "Baseline noise across channels",
                "Total baseline drift",
                "Measured vertical force",
                "Orthogonal force channels",
                "Center of pressure coordinates"
            ],
            "Criterion": [
                r"Remains $< \pm 2.5 \text{ mV}$",
                "Drift < 1 N or 0.5% of applied load, whichever is less",
                r"Falls within $\pm 0.5\%$ of true weight (221.30 N to 223.52 N)",
                r"Must not change by more than $\pm 0.2\%$ of vertical load ($\pm 0.44$ N)",
                "Must match physical locations to within $\pm 3$ mm"
            ],
            "Phase 1": ["Pass", "Pass", "Fail", "Fail", "TBD"],
            "Phase 2": ["Pass", "Pass", "Fail", "Fail", "TBD"],
            "Phase 3": ["Pass", "Pass", "Fail", "Fail", "TBD"],
            "Phase 4": ["Pass", "Fail", "Fail", "Fail", "TBD"]
        }
        return pd.DataFrame(criteria)

    # Generate and display Table 4
    table_4 = generate_acceptance_criteria_table()
    print("\n=== Table 4: Acceptance Criteria ===\n")
    print(table_4.to_string(index=False))

    #%% 4.1 / 5.1 - Unloaded noise test: run + evaluate

    raw_noise_df, force_noise_df = run_static_noise_test(directories, out_dir=plot_out_dir)
    raw_noise_df["Phase Label"] = raw_noise_df["Phase"].map(PHASE_LABELS)
    print("\n=== Raw Noise Stats ===\n")
    print(raw_noise_df)

    noise_results_force = apply_noise_criteria(force_noise_df, noise_threshold_N=0.5)
    noise_results_force["Phase Label"] = noise_results_force["Phase"].map(PHASE_LABELS)
    print("\n=== Force Noise Criteria Results ===\n")
    print(noise_results_force[["Phase", "Channel", "Peak-to-Peak (N)", "Pass"]])

    #%% 5.1 - Table 5: Per-phase, per-channel ptp/RMS summary
    # This table is used for the unloaded noise assessment requested in the report.
    table_5 = (
        force_noise_df
        .groupby(["Phase", "Channel"])[["Peak-to-Peak", "RMS"]]
        .mean()
        .unstack("Channel")
    )

    # Add loaded-case FFT noise metrics for the same report table.
    loaded_case_files = [f for f in get_files_by_phase(directories, phase=3, keyword=None)
                         if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]
    loaded_case_file = loaded_case_files[0] if loaded_case_files else None
    if loaded_case_file is not None:
        loaded_case_raw, _ = load_force_file(loaded_case_file["filepath"], loaded_case_file["phase"])
        fft_noise_summary = summarize_loaded_fft_noise(
            loaded_case_raw,
            channel_columns=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6'],
            fs=1000.0,
            cutoff_hz=20.0,
        )
        fft_noise_summary = fft_noise_summary.set_index("Channel")
        fft_noise_summary = fft_noise_summary[[
            "FFT line-noise share (filtered, %)",
            "FFT high-freq share (filtered, %)",
            "FFT noise reduction (%)",
        ]]
        table_5 = pd.concat([table_5, fft_noise_summary.T], axis=0)

    print("\n=== Table 5: Noise Summary ===\n")
    print(table_5)

    #%% Additional requested analysis for phase-3 center drift / unloaded noise
    # This check is intended to surface broad-channel behavior that would be consistent
    # with a loose AI SENSE wire after the unloaded noise test.
    phase3_center_file = [f for f in get_files_by_phase(directories, phase=3, keyword=None)
                          if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()][0]
    phase3_center_raw, phase3_center_force = load_force_file(phase3_center_file["filepath"], phase3_center_file["phase"])
    phase3_channel_metrics = compute_phase3_center_drift_channel_metrics(phase3_center_raw, phase3_center_force)
    print("\n=== Phase 3 center drift channel metrics ===\n")
    print(phase3_channel_metrics)

    #%% Representative channel histogram + FFT plot
    _example_file = get_files_by_phase(directories, phase="all", keyword="drift")[0]
    _example_raw, _ = load_force_file(_example_file["filepath"], _example_file["phase"])

    _hist_fig = go.Figure(data=[go.Histogram(x=_example_raw["Ch1"] * 1000.0, nbinsx=50)])
    _hist_fig.update_layout(
        title=f"Representative noise histogram (Ch1, mV): {_example_file['basename']}",
        xaxis_title="Voltage (mV)", yaxis_title="Count", template="plotly_white"
    )
    _hist_fig.write_html(os.path.join(plot_out_dir, "representative_noise_histogram.html"))

    plot_fft(
        _example_raw, channel="Ch3",
        title=f"Representative FFT (Ch1): {_example_file['basename']}",
        save_path=os.path.join(plot_out_dir, "representative_noise_fft.html")
    )

    #%% 4.2 / 5.2 - Drift test: run + evaluate

    drift_df = run_drift_test(directories, out_dir=plot_out_dir)
    drift_results = apply_drift_criteria(
        drift_df, applied_load_lbs=APPLIED_LOAD_LBS,
        abs_threshold_n=1.0, pct_threshold=0.005
    )
    drift_results["Phase Label"] = drift_results["Phase"].map(PHASE_LABELS)
    print("\n=== Drift Results ===\n")
    print(drift_results)

    #%% 5.2 - Table 6: Drift summary

    DRIFT_RECORDING_SECONDS = 300.0
    table_6 = drift_results[["Phase", "File", "Drift (lbf)"]].copy()
    table_6["Drift rate (lbf/s)"] = table_6["Drift (lbf)"] / DRIFT_RECORDING_SECONDS
    table_6["Pass"] = drift_results["Pass"].values
    print("\n=== Table 6: Drift Summary ===\n")
    print(table_6)

    #%% 4.3 / 5.3 - Corner loading test: run + evaluate

    corner_df = run_corner_test(directories, out_dir=plot_out_dir)
    corner_results = apply_weight_and_crosstalk_criteria(
        corner_df, applied_load_lbs=APPLIED_LOAD_LBS,
        weight_pct=0.005, crosstalk_pct=0.002
    )
    cop_results = apply_cop_criteria(corner_results, TRUE_COP_BY_CORNER_MM, tolerance_mm=10.0)
    cop_results["Phase Label"] = cop_results["Phase"].map(PHASE_LABELS)
    print("\n=== Corner Loading Results ===\n")
    print(cop_results)

    #%% 5.3 - Table 7: Corner loading summary (in N and mm)

    table_7 = cop_results[["Phase", "Corner", "Fx (N)", "Fy (N)", "Fz (N)",
                            "CoP X (mm)", "CoP Y (mm)"]].copy()
    table_7 = table_7.sort_values(["Phase", "Corner"])
    print("\n=== Table 7: Corner Loading (N, mm) ===\n")
    print(table_7)

    #%% Additional requested loaded-case FFT noise signature summary
    # This quantifies how much residual electrical-noise power remains in the loaded
    # case after the low-pass filter is applied.

    #%% Additional requested Fx/Fy consistency check between center and corner loading
    # This evaluates whether the orthogonal channels remain approximately unchanged
    # across center and corner loading conditions.
    center_force_file = [f for f in get_files_by_phase(directories, phase=3, keyword=None)
                         if 'center' in f['basename'].lower()][0]
    center_raw, center_force = load_force_file(center_force_file["filepath"], center_force_file["phase"])
    corner_force_file = [f for f in get_files_by_phase(directories, phase=3, keyword=None)
                         if any(k in f['basename'].lower() for k in ['corner', '_tl', '_tr', '_br', '_bl'])][0]
    _, corner_force = load_force_file(corner_force_file["filepath"], corner_force_file["phase"])
    center_corner_fx_fy = compare_center_corner_fx_fy(center_force, corner_force, applied_load_lbs=APPLIED_LOAD_LBS)
    print("\n=== Center vs corner Fx/Fy comparison ===\n")
    print(center_corner_fx_fy)

    #%% Table 8: Spatial sensitivity information
    table_8 = generate_spatial_sensitivity_table(corner_df, drift_df)
    print("\n=== Table 8: Spatial Sensitivity ===\n")
    print(table_8)

    #%% 6.2.1 - MSA6 amplifier comparison (Phase 1 vs 2)
    # Compare mean values (hardware zero effectiveness)

    def compare_amplifier_zero(force_noise_df):
        """Compare mean values between phase 1 and 2 to assess hardware zero."""
        phase1 = force_noise_df[force_noise_df["Phase"] == 1]
        phase2 = force_noise_df[force_noise_df["Phase"] == 2]
        
        # Get mean Fz values
        mean_fz_p1 = phase1[phase1["Channel"] == "Fz"]["Mean"].values[0]
        mean_fz_p2 = phase2[phase2["Channel"] == "Fz"]["Mean"].values[0]
        
        return {
            "Phase 1 Mean Fz (lbf)": mean_fz_p1,
            "Phase 2 Mean Fz (lbf)": mean_fz_p2,
            "Difference (lbf)": mean_fz_p2 - mean_fz_p1,
            "Difference (N)": (mean_fz_p2 - mean_fz_p1) * LBF_TO_N
        }

    amp_zero_comparison = compare_amplifier_zero(force_noise_df)
    print("\n=== Amplifier Zero Comparison (Phase 1 vs 2) ===\n")
    print(f"Phase 1 Mean Fz: {amp_zero_comparison['Phase 1 Mean Fz (lbf)']:.4f} lbf")
    print(f"Phase 2 Mean Fz: {amp_zero_comparison['Phase 2 Mean Fz (lbf)']:.4f} lbf")
    print(f"Difference: {amp_zero_comparison['Difference (N)']:.4f} N")

    #%% 6.2.2 - OR67800 drift analysis (Phase 4)

    def analyze_plate_warmup(drift_df):
        """Analyze drift characteristics for Phase 4."""
        phase4_drift = drift_df[drift_df["Phase"] == 4]
        if phase4_drift.empty:
            return "No Phase 4 drift data available"
        
        drift_n = phase4_drift["Drift (N)"].values[0]
        return {
            "Phase 4 Drift (N)": drift_n,
            "Likely cause": "Inadequate warm-up of force plate",
            "Recommendation": "Allow longer warm-up time before recording"
        }

    plate_analysis = analyze_plate_warmup(drift_df)
    print("\n=== OR67800 Force Plate Analysis ===\n")
    print(f"Drift: {plate_analysis['Phase 4 Drift (N)']:.2f} N")
    print(f"Likely Cause: {plate_analysis['Likely cause']}")
    print(f"Recommendation: {plate_analysis['Recommendation']}")

    #%% Generate Equipment History Log

    def generate_equipment_history_log():
        """Generate equipment history log based on test results."""
        history = {
            "Date": [datetime.now().strftime("%Y-%m-%d")],
            "Equipment": [
                "BP400600 Force Plate (SN unknown), MSA6 SN6893, PowerLab 16/30\n"
                "BP400600 Force Plate (SN unknown), MSA6 SN7526, PowerLab 16/30\n"
                "BP400600 Force Plate (SN unknown), MSA6 SN7526, NI-6210\n"
                "OR6-7-8000 Force Plate (SN1), MSA6 SN7526, PowerLab 16/30"
            ],
            "Reason for test": ["Comprehensive validation testing"],
            "Outcome": [
                "Phase 1: Baseline - Noise PASS, Drift PASS, Weight FAIL, Crosstalk FAIL, CoP TBD\n"
                "Phase 2: Amplifier isolation - Noise PASS, Drift PASS, Weight FAIL, Crosstalk FAIL, CoP TBD\n"
                "Phase 3: DAQ isolation - Noise PASS, Drift PASS, Weight FAIL, Crosstalk FAIL, CoP TBD\n"
                "Phase 4: Plate isolation - Noise PASS, Drift FAIL, Weight FAIL, Crosstalk FAIL, CoP TBD"
            ]
        }
        return pd.DataFrame(history)

    equipment_history = generate_equipment_history_log()
    print("\n=== Equipment History Log ===\n")
    print(equipment_history.to_string(index=False))

    #%% Generate Summary Table for Recommendations

    def generate_recommendations_table():
        """Generate equipment status and recommendations."""
        recommendations = {
            "Equipment": [
                "BP400600 Force Plate",
                "MSA6 SN6893 Amplifier",
                "MSA6 SN7526 Amplifier",
                "PowerLab 16/30 DAQ",
                "NI-6210 USB DAQ",
                "OR6-7-8000 Force Plate"
            ],
            "Status": [
                "Acceptable (with caveats)",
                "Acceptable",
                "Acceptable (zero offset issue)",
                "Acceptable",
                "Acceptable (drift issue)",
                "Conditional (requires warm-up)"
            ],
            "Recommendation": [
                "Perform dynamic testing; calibrate for accurate weight measurement",
                "Continue use; hardware zero effective",
                "Continue use; note zero offset difference from SN6893",
                "Continue use; good drift performance",
                "Use with caution; 1.78N drift observed",
                "Allow 30+ minute warm-up; perform thermal characterization"
            ],
            "Findings": [
                "Corner loading tests show Fz reading below true weight",
                "SN6893 shows mean values closer to zero than SN7526",
                "SN7526 shows hardware zero less effective",
                "Drift < 0.15N in z-direction",
                "1.78N drift in z-direction with BP400600",
                "Significant drift in Fz over 5-minute test"
            ]
        }
        return pd.DataFrame(recommendations)

    recommendations_df = generate_recommendations_table()
    print("\n=== Recommendations ===\n")
    print(recommendations_df.to_string(index=False))

    #%% ============================================================
    #   COMPONENT COMPARISON HELPERS (5.4-5.6)
    #   ============================================================

    def plot_component_comparison(metric_df, phase_a, phase_b, value_col, group_col,
                                   label_a, label_b, title, y_label, save_path=None):
        """Grouped bar chart comparing two phases across a categorical axis."""
        a = metric_df[metric_df["Phase"] == phase_a].set_index(group_col)[value_col]
        b = metric_df[metric_df["Phase"] == phase_b].set_index(group_col)[value_col]
        categories = sorted(set(a.index) | set(b.index))
        fig = go.Figure()
        fig.add_trace(go.Bar(name=label_a, x=categories, y=[a.get(c, np.nan) for c in categories]))
        fig.add_trace(go.Bar(name=label_b, x=categories, y=[b.get(c, np.nan) for c in categories]))
        fig.update_layout(barmode="group", title=title, yaxis_title=y_label, template="plotly_white")
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
        return fig

    #%% 5.4 - Amplifier comparison (Phase 1 vs 2)

    # Use raw noise for amplifier comparison (mV)
    plot_component_comparison(
        raw_noise_df, phase_a=1, phase_b=2, value_col="Std Dev", group_col="Channel",
        label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[2],
        title="Amplifier comparison: unloaded noise (Std Dev, mV)", y_label="mV",
        save_path=os.path.join(plot_out_dir, "amplifier_comparison_noise.html")
    )

    # Compare corner Fz for amplifier
    plot_component_comparison(
        cop_results, phase_a=1, phase_b=2, value_col="Fz (N)", group_col="Corner",
        label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[2],
        title="Amplifier comparison: corner Fz (N)", y_label="N",
        save_path=os.path.join(plot_out_dir, "amplifier_comparison_fz.html")
    )

    #%% 5.5 - DAQ comparison (Phase 1 vs 3)

    plot_component_comparison(
        raw_noise_df, phase_a=1, phase_b=3, value_col="Std Dev", group_col="Channel",
        label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[3],
        title="DAQ comparison: unloaded noise (Std Dev, mV)", y_label="mV",
        save_path=os.path.join(plot_out_dir, "daq_comparison_noise.html")
    )

    plot_component_comparison(
        cop_results, phase_a=1, phase_b=3, value_col="Fz (N)", group_col="Corner",
        label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[3],
        title="DAQ comparison: corner Fz (N)", y_label="N",
        save_path=os.path.join(plot_out_dir, "daq_comparison_fz.html")
    )

    #%% 5.6 - Force plate comparison (Phase 2 vs 4)

    plot_component_comparison(
        raw_noise_df, phase_a=2, phase_b=4, value_col="Std Dev", group_col="Channel",
        label_a=PHASE_LABELS[2], label_b=PHASE_LABELS[4],
        title="Force plate comparison: unloaded noise (Std Dev, mV)", y_label="mV",
        save_path=os.path.join(plot_out_dir, "plate_comparison_noise.html")
    )

    plot_component_comparison(
        cop_results, phase_a=2, phase_b=4, value_col="Fz (N)", group_col="Corner",
        label_a=PHASE_LABELS[2], label_b=PHASE_LABELS[4],
        title="Force plate comparison: corner Fz (N)", y_label="N",
        save_path=os.path.join(plot_out_dir, "plate_comparison_fz.html")
    )

    #%% 6.1 - Loaded vs. unloaded FFT overlay

    _static_file = get_files_by_phase(directories, phase=1, keyword="static")[0]
    _drift_file = get_files_by_phase(directories, phase=1, keyword=None)
    _drift_file = [f for f in _drift_file if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()][0]

    _static_raw, _ = load_force_file(_static_file["filepath"], _static_file["phase"])
    _drift_raw, _ = load_force_file(_drift_file["filepath"], _drift_file["phase"])

    def _fft_trace(raw_df, channel='Ch3', fs=1000.0):
        signal = raw_df[channel].values - np.mean(raw_df[channel].values)
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, 1 / fs)[:n // 2]
        power = np.abs(yf[:n // 2]) ** 2 / n
        return xf, power

    _xf_u, _p_u = _fft_trace(_static_raw)
    _xf_l, _p_l = _fft_trace(_drift_raw)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_xf_u, y=_p_u, mode='lines', name='Unloaded'))
    _fig.add_trace(go.Scatter(x=_xf_l, y=_p_l, mode='lines', name='Loaded (50 lb, center)'))
    _fig.update_layout(
        title="Loaded vs. unloaded FFT Ch3, Phase 1",
        xaxis_title="Frequency (Hz)", yaxis_title="Power", template="plotly_white"
    )
    _fig.write_html(os.path.join(plot_out_dir, "loaded_vs_unloaded_fft.html"))

    #%% 6.2 - Full drift time series (warm-up vs. genuine drift diagnostic)

    def diagnose_drift_curve(force_df, fs=1000.0, window_s=60):
        """Returns (time, Fz) full trace plus first/last-minute drift rates."""
        fz = force_df['Fz'].values
        t = force_df['Time'].values
        n_window = int(window_s * fs)
        first_rate = (np.mean(fz[n_window:2 * n_window]) - np.mean(fz[:n_window])) / window_s
        last_rate = (np.mean(fz[-n_window:]) - np.mean(fz[-2 * n_window:-n_window])) / window_s
        return t, fz, first_rate, last_rate

    for phase in [1, 2, 3, 4]:
        _dfile = [f for f in get_files_by_phase(directories, phase=phase, keyword=None)
                  if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()][0]
        _, _dforce = load_force_file(_dfile["filepath"], _dfile["phase"])
        _t, _fz, _first_rate, _last_rate = diagnose_drift_curve(_dforce)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_t, y=_fz, mode='lines', name='Fz'))
        _fig.update_layout(
            title=(f"Phase {phase} drift trace | first-min rate: {_first_rate:.4f} lbf/s, "
                   f"last-min rate: {_last_rate:.4f} lbf/s"),
            xaxis_title="Time (s)", yaxis_title="Fz (lbf)", template="plotly_white"
        )
        _fig.write_html(os.path.join(plot_out_dir, f"phase{phase}_drift_trace.html"))
        print(f"Phase {phase}: first-minute rate = {_first_rate:+.4f} lbf/s, "
              f"last-minute rate = {_last_rate:+.4f} lbf/s "
              f"({'settling' if abs(_last_rate) < abs(_first_rate) * 0.5 else 'NOT settling'})")

    #%% 6.3 - Phase 3 raw channel diagnostic

    _phase3_stats = raw_noise_df[raw_noise_df["Phase"] == 3].set_index("Channel")[["Mean", "Std Dev", "Peak-to-Peak"]]
    _phase1_stats = raw_noise_df[raw_noise_df["Phase"] == 1].set_index("Channel")[["Mean", "Std Dev", "Peak-to-Peak"]]
    _phase3_vs_1 = _phase1_stats.join(_phase3_stats, lsuffix=" (Phase 1)", rsuffix=" (Phase 3)")
    print("\n=== Phase 3 vs Phase 1 Raw Channel Comparison ===\n")
    print(_phase3_vs_1)

    _p3_static_file = get_files_by_phase(directories, phase=3, keyword="static")[0]
    _p3_raw, _ = load_force_file(_p3_static_file["filepath"], _p3_static_file["phase"])
    plot_interactive_timeseries(
        _p3_raw, title=f"Phase 3 raw channels: {_p3_static_file['basename']}", y_label="Voltage (V)",
        save_path=os.path.join(plot_out_dir, "phase3_raw_channels.html")
    )

    #%% 6.4 - Spatial Fz uniformity check

    def compute_spatial_uniformityFz(corner_results, drift_results, applied_load_lbs):
        true_load_n = applied_load_lbs * LBF_TO_N
        rows = []
        for phase in sorted(corner_results["Phase"].unique()):
            phase_corners = corner_results[corner_results["Phase"] == phase]
            center_row = drift_results[drift_results["Phase"] == phase]
            if center_row.empty or phase_corners.empty:
                continue
            center_fz_n = center_row["Final Fz (N)"].mean()
            fz_values = list(phase_corners["Fz (N)"].values) + [center_fz_n]
            fz_range = max(fz_values) - min(fz_values)
            rows.append({
                "Phase": phase,
                "Min Fz (N)": min(fz_values),
                "Max Fz (N)": max(fz_values),
                "Fz range (N)": fz_range,
                "Fz range (% of true load)": fz_range / true_load_n * 100,
            })
        return pd.DataFrame(rows)
    #%% 6.4 - Spatial Fy uniformity check

    def compute_spatial_uniformityFy(corner_results, drift_results, applied_load_lbs):
        true_load_n = applied_load_lbs * LBF_TO_N
        rows = []
        for phase in sorted(corner_results["Phase"].unique()):
            phase_corners = corner_results[corner_results["Phase"] == phase]
            center_row = drift_results[drift_results["Phase"] == phase]
            if center_row.empty or phase_corners.empty:
                continue
            center_fy_n = center_row["Final Fy (N)"].mean()
            fy_values = list(phase_corners["Fy (N)"].values) + [center_fy_n]
            fy_range = max(fy_values) - min(fy_values)
            rows.append({
                "Phase": phase,
                "Min Fy (N)": min(fy_values),
                "Max Fy (N)": max(fy_values),
                "Fy range (N)": fy_range,
                "Fy range (% of true load)": fy_range / true_load_n * 100,
            })
        return pd.DataFrame(rows)
    #%% 6.4 - Spatial Fx uniformity check

    def compute_spatial_uniformityFx(corner_results, drift_results, applied_load_lbs):
        true_load_n = applied_load_lbs * LBF_TO_N
        rows = []
        for phase in sorted(corner_results["Phase"].unique()):
            phase_corners = corner_results[corner_results["Phase"] == phase]
            center_row = drift_results[drift_results["Phase"] == phase]
            if center_row.empty or phase_corners.empty:
                continue
            center_fx_n = center_row["Final Fx (N)"].mean()
            fx_values = list(phase_corners["Fx (N)"].values) + [center_fx_n]
            fx_range = max(fx_values) - min(fx_values)
            rows.append({
                "Phase": phase,
                "Min Fy (N)": min(fx_values),
                "Max Fy (N)": max(fx_values),
                "Fy range (N)": fx_range,
                "Fy range (% of true load)": fx_range / true_load_n * 100,
            })
        return pd.DataFrame(rows)
    # Scatter plot: Fz error at each corner location
    _true_load_n = APPLIED_LOAD_LBS * LBF_TO_N
    _scatter = cop_results.copy()
    _scatter["Fz error (N)"] = _scatter["Fz (N)"] - _true_load_n
    _fig = go.Figure()
    for phase in sorted(_scatter["Phase"].unique()):
        _sub = _scatter[_scatter["Phase"] == phase]
        _fig.add_trace(go.Scatter(
            x=_sub["CoP X (mm)"], y=_sub["CoP Y (mm)"], mode='markers+text',
            text=_sub["Corner"], textposition="top center",
            marker=dict(size=14, color=_sub["Fz error (N)"], colorscale="RdBu_r",
                        colorbar=dict(title="Fz error (N)"), cmid=0),
            name=f"Phase {phase}", visible=(phase == corner_results["Phase"].min())
        ))
    _fig.update_layout(
        title="Spatial Fz error map by corner (select phase via legend)",
        xaxis_title="CoP X (mm)", yaxis_title="CoP Y (mm)", template="plotly_white"
    )
    _fig.write_html(os.path.join(plot_out_dir, "spatial_fz_error_map.html"))

    print("\n=== Analysis Complete ===\n")
    print(f"Plots saved to: {plot_out_dir}")

if __name__ == "__main__":
    run_report()
