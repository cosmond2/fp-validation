#%% Imports
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq

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

# Check on this. I feel like it may need to take in phase number to determine if moment outputs will be in lb-in or lb-ft
def convert_volt_to_force(df, matrix):
    """Raw amplifier output (V) -> calibrated force/moment (lbf, lbf-in)."""
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
    return calibrated_df

def convert_volt_to_mv(df):
    """Raw amplifier output (V) -> mV, unchanged channel names, for
    evaluating the raw-signal noise criterion"""
    mv_df = df.copy()
    channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    mv_df[channel_cols] = mv_df[channel_cols] * 1000.0
    return mv_df

# Might be easier to do phase-aware conversion in convert_volt_to_force and have consistent moment units
def get_reader_and_matrix(phase):
    """Central place mapping phase -> (reader function, calibration matrix,
    moment length unit). Avoids duplicating this if/else in every test."""
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
    reader, matrix, _ = get_reader_and_matrix(phase)
    raw_df = reader(filepath)
    force_df = convert_volt_to_force(raw_df, matrix)
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

def compute_channel_stats(df, columns, units=""):
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
            "Units": units
        })
    return pd.DataFrame(rows)

# At this phase, we assume lbf/lbf-in, but I think it could still be lbf-ft in some cases
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
    channel must stay within +/- force threshold. Using peak-to-peak
    (not RMS) since the criterion is phrased as a +/- bound."""
    out = force_noise_df.copy()
    out["Pass"] = out["Peak-to-Peak"] <= (2 * noise_threshold_N)
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
    return out

def apply_cop_criteria(corner_df, true_cop_by_phase_corner, tolerance_mm=3.0):
    """Table 4, row 1.3/2.3/3.3/4.3: CoP within +/-3 mm of the physical
    load location.

    true_cop_by_phase_corner must be supplied by the caller, keyed by
    phase first since the physical corner offsets depend on which force
    plate was under test (BP400600 for phases 1-3, OR6-7-8000 for phase
    4), e.g.:
        {1: {'TL': (x_mm, y_mm), 'TR': (...), 'BR': (...), 'BL': (...)},
         2: {...same plate as phase 1...},
         3: {...same plate as phase 1...},
         4: {...OR6-7-8000 offsets...}}
    measured from each plate's electrical/geometric center using the
    weight base plate placement in Figures 5-9. Report itself lists
    this criterion as TBD -- these coordinates aren't in the raw data
    and must come from physical measurement of each setup."""
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

#%% 2.2 / 3.1 - Test configuration: paths, phase labels, applied load
# Report Table 3 maps phase number -> (force plate, amplifier, DAQ, purpose).
# Keeping that mapping here, next to the paths, makes it easy to label
# every downstream table with the same phase descriptions used in the report.

data_dir_1 = r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files"
data_dir_2 = r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
plot_out_dir = r"G:\mkersh\Studies\77EDSTissueFunction\Processed Data\carle_force_plate_validation_dataset\diagnostic_plots"
os.makedirs(plot_out_dir, exist_ok=True)

directories = [data_dir_1, data_dir_2]
APPLIED_LOAD_LBS = 50.0  # Sections 4.2 step 5, 4.3 step 5: 50 lb weight + base plate

BASE_PLATE_DIA = 127 #mm units

DIMS_BP400600 = {
    'height':600, #mm units
    'width':400 #mm units
}

DIMS_OR67800 = {
    'height':508, #mm units
    'width':464 #mm units
}

PHASE_LABELS = {
    1: "Phase 1: BP400600 / MSA6 SN6893 / PowerLab (baseline)",
    2: "Phase 2: BP400600 / MSA6 SN7526 / PowerLab (amplifier isolation)",
    3: "Phase 3: BP400600 / MSA6 SN7526 / NI-6210 (DAQ isolation)",
    4: "Phase 4: OR6-7-8000 / MSA6 SN7526 / PowerLab (force plate isolation)",
}

#
# Keyed by phase, not just corner: phases 1-3 test the BP400600 and
# phase 4 tests the OR6-7-8000, which are different footprints, so "TL"
# on one plate is not the same physical offset from center as "TL" on
# the other. Phases 1-3 share the same plate (only amplifier/DAQ change
# between them), so those three phases can reuse one set of measurements.
TRUE_COP_BY_CORNER_MM = {
    1: {'TL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'TR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2)},  # BP400600
    2: {'TL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'TR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2)},  # BP400600
    3: {'TL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'TR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, -1*DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BR': (DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2), 
        'BL': (-1*DIMS_BP400600['width']/2-BASE_PLATE_DIA/2, DIMS_BP400600['height']/2-BASE_PLATE_DIA/2)},  # BP400600
    4: {'TL': (-1*DIMS_OR67800['width']/2-BASE_PLATE_DIA/2, -1*DIMS_OR67800['height']/2-BASE_PLATE_DIA/2), 
        'TR': (DIMS_OR67800['width']/2-BASE_PLATE_DIA/2, -1*DIMS_OR67800['height']/2-BASE_PLATE_DIA/2), 
        'BR': (DIMS_OR67800['width']/2-BASE_PLATE_DIA/2, DIMS_OR67800['height']/2-BASE_PLATE_DIA/2), 
        'BL': (-1*DIMS_OR67800['width']/2-BASE_PLATE_DIA/2, DIMS_OR67800['height']/2-BASE_PLATE_DIA/2)},  # BP400600
}

#%% 3.4 - Acceptance criteria (Table 4)
# Pulling these out as named constants so every number in Table 4 has a
# single, obvious home. The apply_*_criteria functions default to these
# same values, but naming them here means a reader can match this cell
# 1:1 against the report table instead of hunting through function defaults.

NOISE_THRESHOLD_N = 0.5          # Row 1.1/2.1/3.1/4.1: baseline noise < +/- 0.5N (ref: bertec force plates)
DRIFT_ABS_THRESHOLD_N = 1.0       # Row 1.2/2.2/3.2/4.2: drift < 1 N ...
DRIFT_PCT_THRESHOLD = 0.005       # ... or 0.5% of applied load, whichever is less
WEIGHT_PCT_THRESHOLD = 0.005      # Row 1.3/2.3/3.3/4.3: Fz within +/-0.5% of true weight
CROSSTALK_PCT_THRESHOLD = 0.002   # Row 1.3/2.3/3.3/4.3: Fx/Fy within +/-0.2% of vertical load
COP_TOLERANCE_MM = 10.0            # Row 1.3/2.3/3.3/4.3: CoP within +/-3 mm of physical location

#%% 4.1 / 5.1 - Unloaded noise test: run + evaluate (Table 5)

raw_noise_df, force_noise_df = run_static_noise_test(directories, out_dir=plot_out_dir)
raw_noise_df["Phase Label"] = raw_noise_df["Phase"].map(PHASE_LABELS)
print(raw_noise_df)
noise_results_force = apply_noise_criteria(force_noise_df, noise_threshold_N=NOISE_THRESHOLD_N)
noise_results_force["Phase Label"] = noise_results_force["Phase"].map(PHASE_LABELS)
print(noise_results_force)

#%% 5.1 - Table 5 layout: per-phase, per-channel ptp/RMS summary

table_5 = (
    force_noise_df
    .groupby(["Phase", "Channel"])[["Peak-to-Peak", "RMS"]]
    .mean()
    .unstack("Channel")
)
print(table_5)

#%% 5.1 - Representative channel histogram + FFT plot
# Report asks for one representative channel histogram and one FFT plot
# to look for electrical noise signatures (e.g. 60 Hz mains hum).

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

#%% 4.2 / 5.2 - Drift test: run + evaluate (Table 6)

drift_df = run_drift_test(directories, out_dir=plot_out_dir)
drift_results = apply_drift_criteria(
    drift_df, applied_load_lbs=APPLIED_LOAD_LBS,
    abs_threshold_n=DRIFT_ABS_THRESHOLD_N, pct_threshold=DRIFT_PCT_THRESHOLD
)
drift_results["Phase Label"] = drift_results["Phase"].map(PHASE_LABELS)
print(drift_results)

#%% 5.2 - Table 6 layout: drift (lbf) and drift rate (lbf/s) per phase
# Each drift recording is 5 minutes (300 s) per section 4.2 step 6.

DRIFT_RECORDING_SECONDS = 300.0
table_6 = drift_results[["Phase", "File", "Drift (lbf)"]].copy()
table_6["Drift rate (lbf/s)"] = table_6["Drift (lbf)"] / DRIFT_RECORDING_SECONDS
print(table_6)

#%% 4.3 / 5.3 - Corner loading test: run + evaluate (Table 7)

corner_df = run_corner_test(directories, out_dir=plot_out_dir)
corner_results = apply_weight_and_crosstalk_criteria(
    corner_df, applied_load_lbs=APPLIED_LOAD_LBS,
    weight_pct=WEIGHT_PCT_THRESHOLD, crosstalk_pct=CROSSTALK_PCT_THRESHOLD
)
cop_results = apply_cop_criteria(corner_results, TRUE_COP_BY_CORNER_MM, tolerance_mm=COP_TOLERANCE_MM)
cop_results["Phase Label"] = cop_results["Phase"].map(PHASE_LABELS)
print(cop_results)

#%% 5.3 - Table 7 layout: Phase x Corner x Fx/Fy/Fz/CoP, in lbf and inches
# Report's Table 7 header is in lbf and inches, unlike the N/mm used
# internally above -- convert back for that specific table.

table_7 = cop_results[["Phase", "Corner", "Fx (N)", "Fy (N)", "Fz (N)",
                        "CoP X (mm)", "CoP Y (mm)"]].copy()
table_7["Fx (lbf)"] = table_7.pop("Fx (N)") / LBF_TO_N
table_7["Fy (lbf)"] = table_7.pop("Fy (N)") / LBF_TO_N
table_7["Fz (lbf)"] = table_7.pop("Fz (N)") / LBF_TO_N
table_7["CoP X (in)"] = table_7.pop("CoP X (mm)") / IN_TO_MM
table_7["CoP Y (in)"] = table_7.pop("CoP Y (mm)") / IN_TO_MM
table_7 = table_7.sort_values(["Phase", "Corner"])
print(table_7)

#%% ============================================================
#   COMPONENT COMPARISON HELPERS (5.4-5.6)
#   Phases were designed as controlled A/B swaps (section 4.4-4.6):
#   1 vs 2 isolates the amplifier, 1 vs 3 isolates the DAQ,
#   2 vs 4 isolates the force plate. One generic bar-chart function
#   covers all three comparisons since the phase pairs are the only
#   thing that changes.
#   ============================================================
 
def plot_component_comparison(metric_df, phase_a, phase_b, value_col, group_col,
                               label_a, label_b, title, y_label, save_path=None):
    """Grouped bar chart comparing two phases across a categorical axis
    (e.g. Channel, or Corner). metric_df must have Phase, group_col, value_col."""
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
 
#%% 5.4 - Amplifier comparison (Phase 1 vs 2): per-channel noise + Fz weight error
 
plot_component_comparison(
    raw_noise_df, phase_a=1, phase_b=2, value_col="Std Dev", group_col="Channel",
    label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[2],
    title="Amplifier comparison: unloaded noise (Std Dev, mV)", y_label="mV",
    save_path=os.path.join(plot_out_dir, "amplifier_comparison_noise.html")
)
plot_component_comparison(
    cop_results, phase_a=1, phase_b=2, value_col="Fz (N)", group_col="Corner",
    label_a=PHASE_LABELS[1], label_b=PHASE_LABELS[2],
    title="Amplifier comparison: corner Fz (N)", y_label="N",
    save_path=os.path.join(plot_out_dir, "amplifier_comparison_fz.html")
)
 
#%% 5.5 - DAQ comparison (Phase 1 vs 3): per-channel noise + Fz weight error
 
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
 
#%% 5.6 - Force plate comparison (Phase 2 vs 4): per-channel noise + Fz weight error
 
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
# Observation 4: 60/120/240 Hz spikes show up unloaded. Overlaying a
# loaded (drift test) trial's FFT tells us whether that noise is small
# relative to the actual signal of interest, or large enough to matter
# for anything beyond static measurements.
 
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
# Observation 5: Table 6 only reports first/last-1000-sample means, which
# can't distinguish a thermal settling curve (decaying, then flat) from
# ongoing linear/random drift. Plot the full trace and split first vs.
# last minute to check whether the rate of change is decreasing.
 
def diagnose_drift_curve(force_df, fs=1000.0, window_s=60):
    """Returns (time, Fz) full trace plus first/last-minute drift rates
    (lbf/s) so a settling curve can be distinguished from ongoing drift."""
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
# Observation 6: phase 3 Fz reads ~150 N instead of ~225-250 N like the
# other phases, despite sharing the same plate/amp as phase 2. Compare
# raw (unconverted) per-channel stats phase 1/2 vs 3 to see whether one
# specific channel looks different (loose wire / connector) or all
# channels are uniformly scaled down (NI-6210 range/gain config issue).
 
_phase3_stats = raw_noise_df[raw_noise_df["Phase"] == 3].set_index("Channel")[["Mean", "Std Dev", "Peak-to-Peak"]]
_phase1_stats = raw_noise_df[raw_noise_df["Phase"] == 1].set_index("Channel")[["Mean", "Std Dev", "Peak-to-Peak"]]
_phase3_vs_1 = _phase1_stats.join(_phase3_stats, lsuffix=" (Phase 1)", rsuffix=" (Phase 3)")
print(_phase3_vs_1)
 
_p3_static_file = get_files_by_phase(directories, phase=3, keyword="static")[0]
_p3_raw, _ = load_force_file(_p3_static_file["filepath"], _p3_static_file["phase"])
plot_interactive_timeseries(
    _p3_raw, title=f"Phase 3 raw channels: {_p3_static_file['basename']}", y_label="Voltage (V)",
    save_path=os.path.join(plot_out_dir, "phase3_raw_channels.html")
)
# Read this plot for: (a) one channel flatlined/pinned at rail -> loose
# connector on that channel; (b) all channels uniformly smaller
# amplitude than phase 1's equivalent plot -> NI-6210 voltage range or
# NRSE/differential wiring mismatch (section 3.3) rather than a single
# bad connection.
 
#%% 6.4 - Spatial Fz uniformity check (corners + center)
# Observation 8: comparing Fz at the 4 corners AND the center (same
# nominal 50 lb load, 5 locations) is a coarse spatial sensitivity map.
# A flat, undamaged plate should read ~true weight everywhere; a
# consistent spatial pattern with near-zero Fx/Fy points toward
# localized strain-gage sensitivity variation (warping/damage) rather
# than a calibration matrix error, which would typically also show up
# as cross-axis (Fx/Fy) error.
 
def compute_spatial_uniformity(corner_results, drift_results, applied_load_lbs):
    true_load_n = applied_load_lbs * LBF_TO_N
    rows = []
    for phase in sorted(corner_results["Phase"].unique()):
        phase_corners = corner_results[corner_results["Phase"] == phase]
        center_row = drift_results[drift_results["Phase"] == phase]
        if center_row.empty or phase_corners.empty:
            continue
        center_fz_n = center_row["Final Fz (N)"].mean() * (LBF_TO_N if False else 1.0)
        # Final Fz (N) column is already in N (see compute_drift); no extra conversion needed.
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
 
spatial_uniformity = compute_spatial_uniformity(corner_results, drift_results, APPLIED_LOAD_LBS)
print(spatial_uniformity)
 
# Scatter plot: Fz error at each corner location, colored by magnitude,
# to see if the pattern is spatially coherent (e.g. one edge consistently
# reads high) rather than random -- coherent = physical, random = noise.
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


# %%
