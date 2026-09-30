#%% Import everything
from force_plate_validation.quickstart import *
from scipy.fft import fft, fftfreq
from plotly.subplots import make_subplots
from pathlib import Path

#%% Define data directories
directories = [
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files",
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
]

#%% Configuration block
FS = 1000
STATIC_TRIM_SECONDS = 2
DRIFT_TRIM_SECONDS = 15
CORNER_TRIM_SECONDS = 2
WARMUP_TRIM_SECONDS = 15
BUTTERWORTH_CUTOFF = 20

raw_channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
force_channel_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

CHANNEL_TO_AXIS = {
    'Ch1': 'Fx', 'Ch2': 'Fy', 'Ch3': 'Fz',
    'Ch4': 'Mx', 'Ch5': 'My', 'Ch6': 'Mz'
}

COLUMN_DISPLAY_NAMES = {
    'Fx': 'Fx range [lbf]',
    'Fy': 'Fy range [lbf]',
    'Fz': 'Fz range [lbf]',
    'Mx': 'Mx range [lbf-in]',
    'My': 'My range [lbf-in]',
    'Mz': 'Mz range [lbf-in]',
}

#%% ================================================================
# PLOT TOGGLE — controls every cell marked "[OPTIONAL PLOT]" below.
# Leave False during normal runs; flip to True when you want figures.
# ================================================================
SHOW_PLOTS = True

PLOT_OUTDIR = Path(r"G:\mkersh\Studies\77EDSTissueFunction\force_plate_plots")
PLOT_OUTDIR.mkdir(parents=True, exist_ok=True)

def show_plot(fig, name):
    """Render inline if SHOW_PLOTS, otherwise save an HTML file."""
    if SHOW_PLOTS:
        fig.show()
    else:
        out = PLOT_OUTDIR / f"{name}.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"[plot saved] {out}")

#%% Helpers: trimming, formatting, CoP, signal accessors

def trim_seconds(df, seconds, time_col='Time'):
    t_start = df[time_col].iloc[0]
    t_end = df[time_col].iloc[-1]
    mask = (df[time_col] >= t_start + seconds) & (df[time_col] <= t_end - seconds)
    trimmed = df[mask].reset_index(drop=True)
    if len(trimmed) == 0:
        raise ValueError(f"Trimming {seconds}s off each end leaves no data (trial too short).")
    return trimmed


def format_mean_sd(mean, sd, decimals=1):
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def compute_cop_timeseries(force_df, moment_units, phase):
    try:
        z_offset_in = get_z_offset_in(phase)
    except NotImplementedError as e:
        print(f"WARNING: {e} — returning NaN CoP for this file.")
        n = len(force_df)
        return np.full(n, np.nan), np.full(n, np.nan)

    fx = force_df['Fx'].values
    fy = force_df['Fy'].values
    fz = force_df['Fz'].values
    mx = force_df['Mx'].values
    my = force_df['My'].values

    if moment_units == 'lbf-ft':
        mx_in = mx * FT_TO_IN
        my_in = my * FT_TO_IN
    else:
        mx_in = mx
        my_in = my

    cop_x_in = (-my_in + (fx * z_offset_in)) / fz
    cop_y_in = (mx_in + (fy * z_offset_in)) / fz
    return cop_x_in, cop_y_in


def _raw_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    return force_df, force_df.attrs.get('moment_units', 'lbf-in')


def _filtered_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    filtered = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered['Time'] = force_df['Time'].values
    return filtered, force_df.attrs.get('moment_units', 'lbf-in')

#%% Helper: stationary-segment detector (velocity-based)

def detect_static_segments(time, cop_x, cop_y,
                            velocity_smooth_sec=0.3, velocity_threshold_in_s=0.01,
                            min_duration_sec=2.0, trim_sec=1.0):
    dt = np.gradient(time)
    dx = np.gradient(cop_x)
    dy = np.gradient(cop_y)
    velocity = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)

    fs_est = 1.0 / np.median(dt)
    window = max(int(velocity_smooth_sec * fs_est), 1)
    velocity_smooth = pd.Series(velocity).rolling(window, center=True, min_periods=1).mean().values

    stationary_mask = velocity_smooth < velocity_threshold_in_s

    n = len(time)
    segments = []
    in_seg, start = False, None
    for i, val in enumerate(stationary_mask):
        if val and not in_seg:
            start, in_seg = i, True
        elif not val and in_seg:
            end, in_seg = i, False
            if time[end - 1] - time[start] >= min_duration_sec:
                segments.append((start, end))
    if in_seg:
        end = n
        if time[end - 1] - time[start] >= min_duration_sec:
            segments.append((start, end))

    trim_n = int(trim_sec * fs_est)
    trimmed = [(s + trim_n, e - trim_n) for s, e in segments if (e - trim_n) > (s + trim_n)]
    return trimmed, velocity_smooth


def plot_cop_and_speed_diagnostics(result, velocity_threshold, tag):
    """CoP magnitude + smoothed speed, with detected static segments highlighted.

    Defined here (rather than in an OPTIONAL PLOT cell) so it can be reused both
    for tuning VELOCITY_THRESHOLD_IN_S_* against the real signal and for the
    later post-hoc diagnostic pass.
    """
    time = result['time']
    segments = result['segments']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=result['cop_mag'], mode='lines',
                             name='CoP magnitude (in)', line=dict(color='steelblue')))
    for s, e in segments:
        fig.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.2, line_width=0)
    fig.update_layout(title=f"CoP magnitude — {result['file_info']['basename']}",
                      xaxis_title="Time (s)", yaxis_title="CoP magnitude (in)",
                      template="plotly_white", width=1000, height=350)
    show_plot(fig, f"{tag}_cop_magnitude_{Path(result['file_info']['basename']).stem}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=result['velocity_smooth'], mode='lines',
                              name='Smoothed speed (in/s)', line=dict(color='darkorange')))
    fig2.add_hline(y=velocity_threshold, line_dash='dash', line_color='red',
                   annotation_text='threshold')
    for s, e in segments:
        fig2.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.15, line_width=0)
    fig2.update_layout(title="CoP speed (the actual signal being thresholded)",
                       xaxis_title="Time (s)", yaxis_title="Speed (in/s)",
                       template="plotly_white", width=1000, height=350)
    show_plot(fig2, f"{tag}_cop_speed_{Path(result['file_info']['basename']).stem}")

#%% Helper: corner-loading identification

CORNER_LABELS = {'_tl': 'TL', '_tr': 'TR', '_br': 'BR', '_bl': 'BL'}
CORNER_ORDER = ['TL', 'TR', 'BR', 'BL']

def identify_corner(basename):
    basename_lower = basename.lower()
    for key, label in CORNER_LABELS.items():
        if key in basename_lower:
            return label
    return 'Unknown'


def mean_corner_metrics(force_df, moment_units, phase):
    fx = force_df['Fx'].mean()
    fy = force_df['Fy'].mean()
    fz = force_df['Fz'].mean()
    cop_x_series, cop_y_series = compute_cop_timeseries(force_df, moment_units, phase)
    return {
        'Fx [lbf]': fx, 'Fy [lbf]': fy, 'Fz [lbf]': fz,
        'CoPx [in]': np.nanmean(cop_x_series),
        'CoPy [in]': np.nanmean(cop_y_series),
    }

#%% 5.1a Static voltage: raw + filtered mean (SD) tables
static_files = get_files_by_phase(directories, phase='all', keyword='static')

raw_rows, filt_rows = [], []
for phase in sorted(set(f['phase'] for f in static_files)):
    phase_files = [f for f in static_files if f['phase'] == phase]

    raw_chunks, filt_chunks = [], []
    for file_info in phase_files:
        raw_df, _ = load_force_file(file_info['filepath'], file_info['phase'])
        mv_df = convert_volt_to_mv(raw_df)
        filtered_mv_df = butterworth_filter(mv_df[raw_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
        filtered_mv_df['Time'] = mv_df['Time'].values

        raw_chunks.append(trim_seconds(mv_df[raw_channel_cols + ['Time']], STATIC_TRIM_SECONDS))
        filt_chunks.append(trim_seconds(filtered_mv_df, STATIC_TRIM_SECONDS))

    raw_pooled = pd.concat(raw_chunks, ignore_index=True)
    filt_pooled = pd.concat(filt_chunks, ignore_index=True)

    raw_row = {'Phase': phase}
    filt_row = {'Phase': phase}
    for ch, axis in CHANNEL_TO_AXIS.items():
        raw_row[f'{axis} mean (SD) [mV]'] = format_mean_sd(raw_pooled[ch].mean(), raw_pooled[ch].std(ddof=1))
        filt_row[f'{axis} mean (SD) [mV]'] = format_mean_sd(filt_pooled[ch].mean(), filt_pooled[ch].std(ddof=1))
    raw_rows.append(raw_row)
    filt_rows.append(filt_row)

raw_mean_sd_table = pd.DataFrame(raw_rows)
filtered_mean_sd_table = pd.DataFrame(filt_rows)

print(f'\n5.1a complete. Raw static voltage — mean (SD) [mV] by phase (trimmed {STATIC_TRIM_SECONDS}s each end):')
raw_mean_sd_table

#%% 5.1b Static force/moment peak-to-peak by phase

def compute_ptp_by_phase(get_signal_df, columns):
    rows = []
    for phase in sorted(set(f['phase'] for f in static_files)):
        phase_files = [f for f in static_files if f['phase'] == phase]
        trial_ptps = []
        for file_info in phase_files:
            signal_df, mu = get_signal_df(file_info)
            trimmed = trim_seconds(signal_df, STATIC_TRIM_SECONDS)
            ptp = {col: np.ptp(trimmed[col].values) for col in columns}
            if mu == 'lbf-ft':
                for m in ['Mx', 'My', 'Mz']:
                    if m in ptp:
                        ptp[m] *= FT_TO_IN
            trial_ptps.append(ptp)
        ptp_df = pd.DataFrame(trial_ptps)
        row = {'Phase': phase}
        for col in columns:
            row[col] = ptp_df[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


raw_ptp_table = (compute_ptp_by_phase(_raw_force_signal, force_channel_cols)
                 .round(2).rename(columns=COLUMN_DISPLAY_NAMES))
filtered_ptp_table = (compute_ptp_by_phase(_filtered_force_signal, force_channel_cols)
                      .round(2).rename(columns=COLUMN_DISPLAY_NAMES))

print(f'\n5.1b complete. Force/moment PTP (trimmed {STATIC_TRIM_SECONDS}s each end):')
raw_ptp_table

#%% 5.1c/d prep — representative file, mv signal, filtered signal, FFT noise signatures
# (This cell is COMPUTE ONLY; figures are drawn in the OPTIONAL PLOT section.)

target_file = next(
    (f for f in static_files if 'phase1' in f['basename'].lower() and 'trial1' in f['basename'].lower()),
    None
)
if target_file is None:
    raise FileNotFoundError("Could not find a Phase 1 static trial 1 file matching 'phase1'/'trial1' in filename.")

print(f"Representative file: {target_file['basename']}")

raw_df, _ = load_force_file(target_file['filepath'], target_file['phase'])
mv_df = convert_volt_to_mv(raw_df)
ch3_mv = mv_df['Ch3'].values
ch3_filtered_mv = butterworth_filter(ch3_mv, fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)

def compute_power_spectrum(signal, fs=1000.0):
    signal = signal - np.mean(signal)
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    power = np.abs(yf[:n // 2]) ** 2 / n
    return xf, power

xf_raw, power_raw = compute_power_spectrum(ch3_mv, fs=FS)
xf_filt, power_filt = compute_power_spectrum(ch3_filtered_mv, fs=FS)

noise_sig_raw = compute_fft_noise_signature(ch3_mv, fs=FS)
noise_sig_filt = compute_fft_noise_signature(ch3_filtered_mv, fs=FS)

print(f"Ch3 raw voltage: mean = {ch3_mv.mean():.4f} mV, std = {ch3_mv.std(ddof=1):.4f} mV, n = {len(ch3_mv)}")
print(f"Raw      — line-noise share: {noise_sig_raw['line_noise_share_pct']:.1f}%, "
      f"high-freq (>{BUTTERWORTH_CUTOFF:.0f} Hz) share: {noise_sig_raw['high_freq_share_pct']:.1f}%")
print(f"Filtered — line-noise share: {noise_sig_filt['line_noise_share_pct']:.1f}%, "
      f"high-freq (>{BUTTERWORTH_CUTOFF:.0f} Hz) share: {noise_sig_filt['high_freq_share_pct']:.1f}%")

#%% 5.2 Drift trials: initial/final Fz + CoP range table
drift_files = get_files_by_phase(directories, phase='all', keyword=None)
drift_files = [f for f in drift_files
               if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]
print(f"Found {len(drift_files)} drift/center files")

drift_rows = []
for file_info in drift_files:
    raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    force_trimmed = trim_seconds(force_df, DRIFT_TRIM_SECONDS)
    filtered_full = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_full['Time'] = force_df['Time'].values
    filtered_trimmed = trim_seconds(filtered_full, DRIFT_TRIM_SECONDS)

    fz_raw_initial = force_trimmed['Fz'].values[:FS].mean()
    fz_raw_final = force_trimmed['Fz'].values[-FS:].mean()
    fz_filt_initial = filtered_trimmed['Fz'].values[:FS].mean()
    fz_filt_final = filtered_trimmed['Fz'].values[-FS:].mean()

    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_trimmed, moment_units, file_info['phase'])

    drift_rows.append({
        'Phase': file_info['phase'],
        'Fz raw initial [lbf]': fz_raw_initial,
        'Fz raw final [lbf]': fz_raw_final,
        'Fz filtered initial [lbf]': fz_filt_initial,
        'Fz filtered final [lbf]': fz_filt_final,
        'CoPx range filtered [in]': np.ptp(cop_x_filt),
        'CoPy range filtered [in]': np.ptp(cop_y_filt),
    })

drift_summary_table = (pd.DataFrame(drift_rows)
                       .sort_values(['Phase']).round(2).reset_index(drop=True))
drift_summary_table

#%% 5.2 (diagnostic table) Phase 3: filtered PTP noise, static vs. drift, per channel

phase3_static_files = [f for f in static_files if f['phase'] == 3]
phase3_drift_files = [f for f in drift_files if f['phase'] == 3]

if not phase3_static_files:
    raise FileNotFoundError("No Phase 3 static files found.")
if not phase3_drift_files:
    raise FileNotFoundError("No Phase 3 drift/center files found.")

def filtered_channel_ptp(file_info, trim_seconds_val):
    raw_df, _ = load_force_file(file_info['filepath'], file_info['phase'])
    mv_df = convert_volt_to_mv(raw_df)
    filtered_mv_df = butterworth_filter(mv_df[raw_channel_cols], fs=1000, cutoff_hz=20)
    filtered_mv_df['Time'] = mv_df['Time'].values
    trimmed = trim_seconds(filtered_mv_df, trim_seconds_val)
    return {ch: np.ptp(trimmed[ch].values) for ch in raw_channel_cols}

static_ptp_avg = pd.DataFrame([filtered_channel_ptp(f, STATIC_TRIM_SECONDS) for f in phase3_static_files]).mean()
drift_ptp_avg = pd.DataFrame([filtered_channel_ptp(f, DRIFT_TRIM_SECONDS) for f in phase3_drift_files]).mean()

ground_check_table = pd.DataFrame({
    'Channel': raw_channel_cols,
    'Axis': [CHANNEL_TO_AXIS[ch] for ch in raw_channel_cols],
    'Static PTP filtered (mV)': static_ptp_avg.values,
    'Drift PTP filtered (mV)': drift_ptp_avg.values,
})
ground_check_table['Ratio (drift / static)'] = (
    ground_check_table['Drift PTP filtered (mV)'] / ground_check_table['Static PTP filtered (mV)']
)
ground_check_table['Flag: possible loose connection'] = ground_check_table['Ratio (drift / static)'] > 3
ground_check_table = ground_check_table.round(2)
ground_check_table

#%% 5.3 Corner-loading tables (raw + filtered)
corner_files = get_files_by_phase(directories, phase='all', keyword=None)
corner_files = [
    f for f in corner_files
    if f['phase'] in (1, 2, 3, 4)
    and any(k in f['basename'].lower() for k in ['corner', '_tl', '_tr', '_br', '_bl'])
    and 'static' not in f['basename'].lower()
    and 'drift' not in f['basename'].lower()
]

print(f"Found {len(corner_files)} corner files")
for f in corner_files:
    print(f"  Phase {f['phase']}: {f['basename']}")

raw_rows, filt_rows = [], []
for file_info in corner_files:
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')
    corner = identify_corner(file_info['basename'])

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_df['Time'] = force_df['Time'].values
    filtered_df.attrs['moment_units'] = moment_units

    force_trimmed = trim_seconds(force_df, CORNER_TRIM_SECONDS)
    filtered_trimmed = trim_seconds(filtered_df, CORNER_TRIM_SECONDS)

    raw_metrics = mean_corner_metrics(force_trimmed, moment_units, file_info['phase'])
    filt_metrics = mean_corner_metrics(filtered_trimmed, moment_units, file_info['phase'])

    raw_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    filt_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    raw_rows.append(raw_metrics)
    filt_rows.append(filt_metrics)

col_order = ['Phase', 'Corner', 'Fx [lbf]', 'Fy [lbf]', 'Fz [lbf]', 'CoPx [in]', 'CoPy [in]']
corner_raw_table = pd.DataFrame(raw_rows)[col_order]
corner_filtered_table = pd.DataFrame(filt_rows)[col_order]

corner_order_map = {c: i for i, c in enumerate(CORNER_ORDER)}
corner_raw_table = (corner_raw_table.assign(_ord=corner_raw_table['Corner'].map(corner_order_map))
                    .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True))
corner_filtered_table = (corner_filtered_table.assign(_ord=corner_filtered_table['Corner'].map(corner_order_map))
                         .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True))

corner_raw_table = corner_raw_table.round(2)
corner_filtered_table = corner_filtered_table.round(2)
corner_raw_table

#%% 5.4 Warm-up tables (Phase 5/6)
warmup_files_5 = get_files_by_phase(directories, phase=5, keyword=None)
warmup_files_6 = get_files_by_phase(directories, phase=6, keyword=None)

print(f"Found {len(warmup_files_5)} Phase 5 (OR6-7-8000) warm-up file(s)")
print(f"Found {len(warmup_files_6)} Phase 6 (BP400600) warm-up file(s)")

warmup_data = {}
for f in warmup_files_5:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    warmup_data['OR6-7-8000'] = {'raw_df': raw_df, 'force_df': force_df,
                                 'basename': f['basename'], 'phase': f['phase']}
for f in warmup_files_6:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    warmup_data['BP400600'] = {'raw_df': raw_df, 'force_df': force_df,
                               'basename': f['basename'], 'phase': f['phase']}

warmup_rows = []
for plate, data in warmup_data.items():
    force_df = data['force_df']
    phase = data['phase']
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    force_trimmed = trim_seconds(force_df, seconds=WARMUP_TRIM_SECONDS)
    filtered_full = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_full['Time'] = force_df['Time'].values
    filtered_trimmed = trim_seconds(filtered_full, seconds=WARMUP_TRIM_SECONDS)

    fz_filt_initial = filtered_trimmed['Fz'].values[:FS].mean()
    fz_filt_final = filtered_trimmed['Fz'].values[-FS:].mean()
    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_trimmed, moment_units, phase)

    warmup_rows.append({
        'Phase': phase,
        'Fz filtered initial [lbf]': fz_filt_initial,
        'Fz filtered final [lbf]': fz_filt_final,
        'Fz filtered drift [lbf]': fz_filt_final - fz_filt_initial,
        'CoPx range filtered [in]': np.ptp(cop_x_filt),
        'CoPy range filtered [in]': np.ptp(cop_y_filt),
    })

warmup_summary_table = pd.DataFrame(warmup_rows).round(2)
warmup_summary_table

#%% Phase 7: discover files (COMPUTE ONLY) — threshold is set later, in the tuning cell
phase7_files = get_files_by_phase(directories, phase=7, keyword=None)
for f in phase7_files:
    name_lower = f['basename'].lower()
    f['path_type'] = 'perimeter' if 'perimeter' in name_lower else ('diagonal' if 'diagonal' in name_lower else 'unknown')

print(f"Found {len(phase7_files)} phase 7 file(s)")
for f in phase7_files:
    print(f"  {f['basename']} -> {f['path_type']}")

#%% Phase 7: compute per-file signals (I/O + filtering + CoP) — independent of any
# stationary-segment threshold, so this only needs to run once per data load.

def compute_file_signal(file_info):
    """Load, filter, and compute CoP/Fz time series. Does NOT depend on a velocity threshold."""
    raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_df['Time'] = force_df['Time'].values

    cop_x, cop_y = compute_cop_timeseries(filtered_df, moment_units, file_info['phase'])
    time = filtered_df['Time'].values
    fz = filtered_df['Fz'].values
    cop_mag = np.sqrt(cop_x ** 2 + cop_y ** 2)

    return {
        'file_info': file_info, 'time': time,
        'cop_x': cop_x, 'cop_y': cop_y, 'cop_mag': cop_mag, 'fz': fz,
    }


phase7_signals = {f['basename']: compute_file_signal(f) for f in phase7_files}

#%% Phase 7: TUNE the threshold — cheap (no I/O/filtering), rerun freely against the
# real CoP-speed signal before committing to a value for the batch below.
VELOCITY_THRESHOLD_IN_S_OR6 = 2.0  # <- adjust and re-run this cell until segments look right

if SHOW_PLOTS:
    _p7_tune_key = phase7_files[0]['basename']
    _sig = phase7_signals[_p7_tune_key]
    _segments, _velocity_smooth = detect_static_segments(
        _sig['time'], _sig['cop_x'], _sig['cop_y'],
        velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6
    )
    plot_cop_and_speed_diagnostics(
        {**_sig, 'segments': _segments, 'velocity_smooth': _velocity_smooth},
        VELOCITY_THRESHOLD_IN_S_OR6, "phase7_tuning"
    )

#%% Phase 7: apply the tuned threshold across every file — no plotting
phase7_analysis = {}
for f in phase7_files:
    sig = phase7_signals[f['basename']]
    segments, velocity_smooth = detect_static_segments(
        sig['time'], sig['cop_x'], sig['cop_y'],
        velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6
    )
    result = {**sig, 'segments': segments, 'velocity_smooth': velocity_smooth}
    phase7_analysis[f['basename']] = result

    print(f"{f['basename']} ({f['path_type']}): {len(result['segments'])} segment(s)")
    for i, (s, e) in enumerate(result['segments']):
        seg_fz = result['fz'][s:e]
        print(f"  Segment {i+1}: n={e-s}, Fz mean={seg_fz.mean():.3f} lbf, "
              f"Fz std={seg_fz.std(ddof=1):.3f} lbf, Fz range={np.ptp(seg_fz):.3f} lbf")
    print()

#%% Phase 7: spatial uniformity table
spatial_columns = [
    'File', 'Path Type', 'Segment', 'CoPx [in]', 'CoPy [in]',
    'Fz [lbf]', 'Fz Std [lbf]',
]
spatial_rows = []
for f in phase7_files:
    a = phase7_analysis[f['basename']]
    for seg_idx, (s, e) in enumerate(a['segments']):
        spatial_rows.append({
            'File': f['basename'],
            'Path Type': f['path_type'],
            'Segment': seg_idx,
            'CoPx [in]': np.mean(a['cop_x'][s:e]),
            'CoPy [in]': np.mean(a['cop_y'][s:e]),
            'Fz [lbf]': np.mean(a['fz'][s:e]),
            'Fz Std [lbf]': np.std(a['fz'][s:e], ddof=1),
        })

spatial_uniformity_df = pd.DataFrame(spatial_rows, columns=spatial_columns)
print(f"\nTotal static segments across all phase 7 files: {len(spatial_uniformity_df)}")
if spatial_uniformity_df.empty:
    print("No static segments were detected. Lower VELOCITY_THRESHOLD_IN_S_OR6 "
        "or review the CoP signal before interpreting spatial uniformity.")
else:
    fz_min = spatial_uniformity_df['Fz [lbf]'].min()
    fz_max = spatial_uniformity_df['Fz [lbf]'].max()
    print(f"Fz range: {fz_min:.2f} to {fz_max:.2f} lbf "
        f"(spread: {fz_max - fz_min:.2f} lbf)")
spatial_uniformity_df

#%% Phase 8: discover files (COMPUTE ONLY) — threshold is set later, in the tuning cell
phase8_files = get_files_by_phase(directories, phase=8, keyword=None)
for f in phase8_files:
    name_lower = f['basename'].lower()
    f['path_type'] = 'perimeter' if 'perimeter' in name_lower else ('diagonal' if 'diagonal' in name_lower else 'unknown')

print(f"Found {len(phase8_files)} phase 8 file(s)")
for f in phase8_files:
    print(f"  {f['basename']} -> {f['path_type']}")

#%% Phase 8: compute per-file signals (I/O + filtering + CoP) — independent of any
# stationary-segment threshold.
phase8_signals = {f['basename']: compute_file_signal(f) for f in phase8_files}

#%% Phase 8: TUNE the threshold — cheap, rerun freely against the real signal.
VELOCITY_THRESHOLD_IN_S_BP = 2.76  # <- adjust and re-run this cell until segments look right

if SHOW_PLOTS:
    _p8_tune_key = phase8_files[0]['basename']
    _sig = phase8_signals[_p8_tune_key]
    _segments, _velocity_smooth = detect_static_segments(
        _sig['time'], _sig['cop_x'], _sig['cop_y'],
        velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_BP
    )
    plot_cop_and_speed_diagnostics(
        {**_sig, 'segments': _segments, 'velocity_smooth': _velocity_smooth},
        VELOCITY_THRESHOLD_IN_S_BP, "phase8_tuning"
    )

#%% Phase 8: apply the tuned threshold across every file — no plotting
phase8_analysis = {}
for f in phase8_files:
    sig = phase8_signals[f['basename']]
    segments, velocity_smooth = detect_static_segments(
        sig['time'], sig['cop_x'], sig['cop_y'],
        velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_BP
    )
    result = {**sig, 'segments': segments, 'velocity_smooth': velocity_smooth}
    phase8_analysis[f['basename']] = result

    print(f"{f['basename']} ({f['path_type']}): {len(result['segments'])} segment(s)")
    for i, (s, e) in enumerate(result['segments']):
        seg_fz = result['fz'][s:e]
        print(f"  Segment {i+1}: n={e-s}, Fz mean={seg_fz.mean():.3f} lbf, "
              f"Fz std={seg_fz.std(ddof=1):.3f} lbf, Fz range={np.ptp(seg_fz):.3f} lbf")
    print()

#%% Phase 8: spatial uniformity table
spatial_rows_bp = []
for f in phase8_files:
    a = phase8_analysis[f['basename']]
    for seg_idx, (s, e) in enumerate(a['segments']):
        spatial_rows_bp.append({
            'File': f['basename'],
            'Path Type': f['path_type'],
            'Segment': seg_idx,
            'CoPx [in]': np.mean(a['cop_x'][s:e]),
            'CoPy [in]': np.mean(a['cop_y'][s:e]),
            'Fz [lbf]': np.mean(a['fz'][s:e]),
            'Fz Std [lbf]': np.std(a['fz'][s:e], ddof=1),
        })

spatial_uniformity_df_bp = pd.DataFrame(spatial_rows_bp, columns=spatial_columns)
print(f"\nTotal static segments across all phase 8 files: {len(spatial_uniformity_df_bp)}")
if spatial_uniformity_df_bp.empty:
    print("No static segments were detected. Lower VELOCITY_THRESHOLD_IN_S_BP "
        "or review the CoP signal before interpreting spatial uniformity.")
else:
    fz_min = spatial_uniformity_df_bp['Fz [lbf]'].min()
    fz_max = spatial_uniformity_df_bp['Fz [lbf]'].max()
    print(f"Fz range: {fz_min:.2f} to {fz_max:.2f} lbf "
        f"(spread: {fz_max - fz_min:.2f} lbf)")
spatial_uniformity_df_bp

#%% [OPTIONAL PLOT] 5.1c — Ch3 (Fz) raw voltage noise histogram
if SHOW_PLOTS:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ch3_mv, nbinsx=60,
                               marker=dict(line=dict(width=0.5, color='white'))))
    fig.update_layout(
        title=f"Ch3 (Fz) Raw Voltage Noise Histogram — {target_file['basename']}",
        xaxis_title="Voltage (mV)", yaxis_title="Count",
        template="plotly_white", bargap=0.02,
    )
    show_plot(fig, "5.1c_static_ch3_histogram")

#%% [OPTIONAL PLOT] 5.1d — Ch3 (Fz) FFT, raw vs. Butterworth
if SHOW_PLOTS:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf_raw, y=power_raw, mode='lines', name='Raw', opacity=0.8))
    fig.add_trace(go.Scatter(x=xf_filt, y=power_filt, mode='lines',
                             name=f'Filtered ({BUTTERWORTH_CUTOFF} Hz cutoff)'))
    fig.add_vline(x=20, line_dash='dash', line_color='gray',
                  annotation_text='Cutoff (20 Hz)', annotation_position='top right')
    fig.update_layout(
        width=900, height=500,
        title=f"Ch3 (Fz) FFT: Raw vs. Butterworth-Filtered — {target_file['basename']}",
        xaxis_title="Frequency [Hz]", yaxis_title="Power",
        yaxis_type="log", template="plotly_white", legend_title="Signal",
        yaxis=dict(type="log", range=[np.log10(1e-6), np.log10(1e2)]),
    )
    show_plot(fig, "5.1d_static_ch3_fft")

#%% [OPTIONAL PLOT] 5.4 — Warm-up Fz vs. elapsed time
if SHOW_PLOTS:
    fig = go.Figure()
    for plate, data in warmup_data.items():
        force_df = data['force_df']
        filtered_fz = butterworth_filter(force_df['Fz'].values, fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
        elapsed_min = (force_df['Time'].values - force_df['Time'].values[0]) / 60.0
        fig.add_trace(go.Scatter(x=elapsed_min, y=filtered_fz, mode='lines',
                                 name=f"{plate} (filtered)"))
    fig.update_layout(
        title="Phase 5/6 Warm-Up Test: Fz (Vertical Force) vs. Elapsed Time",
        xaxis_title="Elapsed Time [min]", yaxis_title="Fz [lbf]",
        template="plotly_white", legend_title="Force Plate",
        width=900, height=500,
    )
    show_plot(fig, "5.4_warmup_fz_vs_time")

#%% [OPTIONAL PLOT] Phase 7 — CoP magnitude + speed diagnostics, final confirmation (first file)
# NOTE: the threshold was already tuned against this plot above, before phase7_analysis
# was built. This is just a final confirmation using the committed threshold/segments —
# see the "TUNE the threshold" cell if the segments here look wrong.

if SHOW_PLOTS:
    _p7_diag_key = phase7_files[0]['basename']
    plot_cop_and_speed_diagnostics(phase7_analysis[_p7_diag_key],
                                   VELOCITY_THRESHOLD_IN_S_OR6, "phase7")

#%% [OPTIONAL PLOT] Phase 7 — Fz stability (Fz vs. time + Fz vs. CoP magnitude), one plot per file

def plot_fz_stability_diagnostic(result):
    file_info = result['file_info']
    time, fz, cop_mag, segments = result['time'], result['fz'], result['cop_mag'], result['segments']

    in_segment = np.zeros(len(time), dtype=bool)
    for s, e in segments:
        in_segment[s:e] = True

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Fz vs. Time", "Fz vs. CoP Magnitude"))

    fig.add_trace(go.Scatter(x=time, y=fz, mode='lines', name='Fz (filtered)',
                             line=dict(color='steelblue')), row=1, col=1)
    for s, e in segments:
        seg_fz = fz[s:e]
        fig.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.15,
                      line_width=0, row=1, col=1)
        fig.add_annotation(x=(time[s] + time[e - 1]) / 2, y=seg_fz.max(),
                           text=f"\u03c3={seg_fz.std(ddof=1):.3f}", showarrow=False,
                           font=dict(size=9, color='darkgreen'), row=1, col=1)

    fig.add_trace(go.Scatter(x=cop_mag[~in_segment], y=fz[~in_segment], mode='markers',
                             marker=dict(size=3, color='lightgray'),
                             name='Outside segment', opacity=0.5), row=1, col=2)
    fig.add_trace(go.Scatter(x=cop_mag[in_segment], y=fz[in_segment], mode='markers',
                             marker=dict(size=4, color='green'),
                             name='Inside segment', opacity=0.7), row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Fz (lbf)", row=1, col=1)
    fig.update_xaxes(title_text="CoP magnitude (in)", row=1, col=2)
    fig.update_yaxes(title_text="Fz (lbf)", row=1, col=2)
    fig.update_layout(
        title=f"Fz stability check — {file_info['basename']} ({file_info['path_type']}) "
              f"— {len(segments)} segment(s)",
        template="plotly_white", width=1200, height=500, showlegend=True,
    )
    show_plot(fig, f"phase7_fz_stability_{Path(file_info['basename']).stem}")


if SHOW_PLOTS:
    for f in phase7_files:
        plot_fz_stability_diagnostic(phase7_analysis[f['basename']])

#%% [OPTIONAL PLOT] Phase 7 — spatial Fz uniformity map

def plot_spatial_fz_map(spatial_df, phase, plate_label, tag):
    """Fz vs. CoP position, with the true plate footprint overlaid as a rectangle.

    Shared by phase 7 (OR6-7-8000) and phase 8 (BP400600) — origin (0, 0) is
    plate center for both plates (see get_plate_dims_in).
    """
    dims = get_plate_dims_in(phase)

    fig = go.Figure()
    hover_text = [
        f"{filename} - seg {segment}"
        for filename, segment in zip(
            spatial_df['File'].tolist(),
            spatial_df['Segment'].tolist(),
        )
    ]
    fig.add_trace(go.Scatter(
        x=spatial_df['CoPx [in]'],
        y=spatial_df['CoPy [in]'],
        mode='markers',
        marker=dict(
            size=12,
            color=spatial_df['Fz [lbf]'],
            colorscale='Viridis', showscale=True,
            colorbar=dict(title=dict(text='Fz [lbf]', side='right'),
                          thickness=15, len=0.75, x=1.02),
            line=dict(width=1, color='black'),
        ),
        text=hover_text,
        hovertemplate='CoPx: %{x:.1f} in<br>CoPy: %{y:.1f} in<br>Fz: %{marker.color:.2f} lbf<br>%{text}<extra></extra>',
    ))
    fig.add_shape(
        type="rect",
        x0=-dims['width'] / 2, x1=dims['width'] / 2,
        y0=-dims['height'] / 2, y1=dims['height'] / 2,
        line=dict(color="black", width=2, dash="dash"),
    )
    fig.update_layout(
        title=f"{plate_label}: Vertical Force (Fz) vs. Position on Plate Surface",
        xaxis_title="CoP X (in)", yaxis_title="CoP Y (in)",
        template="plotly_white", width=850, height=700,
        yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(r=100),
    )
    show_plot(fig, f"{tag}_spatial_fz_map")


if SHOW_PLOTS and not spatial_uniformity_df.empty:
    plot_spatial_fz_map(spatial_uniformity_df, 7, "OR6-7-8000", "phase7")

#%% [OPTIONAL PLOT] Phase 8 — CoP magnitude + speed diagnostics, final confirmation (first file)
# NOTE: same as Phase 7 above — the threshold was already tuned before phase8_analysis
# was built. This is a final confirmation pass, not the tuning step.
if SHOW_PLOTS:
    _p8_diag_key = phase8_files[0]['basename']
    plot_cop_and_speed_diagnostics(phase8_analysis[_p8_diag_key],
                                   VELOCITY_THRESHOLD_IN_S_BP, "phase8")

#%% [OPTIONAL PLOT] Phase 8 — Fz stability, one plot per file
if SHOW_PLOTS:
    for f in phase8_files:
        plot_fz_stability_diagnostic(phase8_analysis[f['basename']])

#%% [OPTIONAL PLOT] Phase 8 — spatial Fz uniformity map
if SHOW_PLOTS and not spatial_uniformity_df_bp.empty:
    plot_spatial_fz_map(spatial_uniformity_df_bp, 8, "BP400600", "phase8")


#%% Gait (phases 9-10): configuration
BODYWEIGHT_LBF = None            # <- REQUIRED: participant bodyweight in lbf
GAIT_CUTOFF_HZ = 20              # low-pass cutoff for gait GRF
GAIT_FZ_THRESHOLD_LBF = 10.0     # Fz above this = foot on plate (tune against Fz trace)
GAIT_MIN_STANCE_SEC = 0.3        # reject spurious contacts shorter than this
GAIT_MAX_STANCE_SEC = 1.5        # reject contacts longer than this
GAIT_N_POINTS = 101              # 0-100% of stance
GAIT_COP_ORIGIN_SAMPLES = 10     # samples averaged at heel strike to define CoP origin

# Toggle: True -> shift each cycle's CoP so heel strike = (0, 0)
ALIGN_COP_TO_HEEL_STRIKE = False

# Flip to -1 if a plate reports a axis with the opposite sign from your convention
# (e.g. Fz negative under load). Applied to GRF outputs/detection only; CoP is
# always computed from the unsigned calibrated data.
GRF_SIGN = {'Fx': 1, 'Fy': 1, 'Fz': 1}

#%% Gait helpers: stance detection, time normalization, cycle extraction

def detect_stance_phases(time, fz, threshold, min_duration_sec, max_duration_sec):
    """Heel strike -> toe off, as Fz threshold crossings.

    Returns list of (start, end) index pairs; `end` is exclusive (first unloaded
    sample). Contacts already in progress at the start of the file or still in
    progress at the end are dropped (partial cycles).
    """
    loaded = (fz > threshold).astype(int)
    edges = np.diff(loaded)
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return []
    ends = ends[ends > starts[0]]              # drop an end with no matching start
    starts = starts[:len(ends)]                # drop a trailing start with no end

    stance = []
    for s, e in zip(starts, ends):
        dur = time[e - 1] - time[s]
        if min_duration_sec <= dur <= max_duration_sec:
            stance.append((s, e))
    return stance


def time_normalize(y, n_points=GAIT_N_POINTS):
    """Resample a 1-D signal onto 0-100% in n_points samples."""
    x_old = np.linspace(0, 100, len(y))
    x_new = np.linspace(0, 100, n_points)
    return np.interp(x_new, x_old, y)


def extract_gait_cycles(file_info, bodyweight_lbf):
    """Load one gait file and return a list of per-cycle dicts (stance phase only)."""
    if bodyweight_lbf is None:
        raise ValueError("Set BODYWEIGHT_LBF in the gait configuration cell first.")

    phase = file_info['phase']
    _, force_df = load_force_file(file_info['filepath'], phase)
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    filt = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=GAIT_CUTOFF_HZ)
    filt['Time'] = force_df['Time'].values
    time = filt['Time'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        cop_x, cop_y = compute_cop_timeseries(filt, moment_units, phase)

    fz_detect = filt['Fz'].values * GRF_SIGN['Fz']
    stance = detect_stance_phases(time, fz_detect, GAIT_FZ_THRESHOLD_LBF,
                                  GAIT_MIN_STANCE_SEC, GAIT_MAX_STANCE_SEC)

    cycles = []
    for i, (s, e) in enumerate(stance, start=1):
        seg = slice(s, e)
        n0 = GAIT_COP_ORIGIN_SAMPLES
        cycles.append({
            'file_info': file_info,
            'phase': phase,
            'cycle': i,
            'start_time': time[s],
            'end_time': time[e - 1],
            'stance_sec': time[e - 1] - time[s],
            'pct': np.linspace(0, 100, GAIT_N_POINTS),
            # GRF, % bodyweight. Fx = medial/lateral, Fy = anterior/posterior, Fz = vertical
            'fx': time_normalize(filt['Fx'].values[seg] * GRF_SIGN['Fx']) / bodyweight_lbf * 100,
            'fy': time_normalize(filt['Fy'].values[seg] * GRF_SIGN['Fy']) / bodyweight_lbf * 100,
            'fz': time_normalize(filt['Fz'].values[seg] * GRF_SIGN['Fz']) / bodyweight_lbf * 100,
            # CoP in plate frame (in), origin = plate center
            'cop_x': time_normalize(cop_x[seg]),
            'cop_y': time_normalize(cop_y[seg]),
            'cop_origin': (np.nanmean(cop_x[s:s + n0]), np.nanmean(cop_y[s:s + n0])),
        })
    return cycles


def gait_summary_table(cycles):
    return pd.DataFrame([{
        'Cycle': c['cycle'],
        'Start [s]': c['start_time'],
        'Stance [s]': c['stance_sec'],
        'Peak Fz [%BW]': c['fz'].max(),
        'Heel-strike CoPx [in]': c['cop_origin'][0],
        'Heel-strike CoPy [in]': c['cop_origin'][1],
    } for c in cycles]).round(2)

#%% Phase 9 (OR6-7-8000 gait): extract cycles — COMPUTE ONLY
phase9_files = get_files_by_phase(directories, phase=9)
if len(phase9_files) != 1:
    print(f"WARNING: expected 1 phase 9 file, found {len(phase9_files)}; using the first.")
print(f"Phase 9 file: {phase9_files[0]['basename']}")

phase9_cycles = extract_gait_cycles(phase9_files[0], BODYWEIGHT_LBF)
print(f"Extracted {len(phase9_cycles)} gait cycle(s)")
phase9_summary = gait_summary_table(phase9_cycles)
phase9_summary

#%% Phase 10 (BP400600 gait): extract cycles — COMPUTE ONLY
phase10_files = get_files_by_phase(directories, phase=10)
if len(phase10_files) != 1:
    print(f"WARNING: expected 1 phase 10 file, found {len(phase10_files)}; using the first.")
print(f"Phase 10 file: {phase10_files[0]['basename']}")

phase10_cycles = extract_gait_cycles(phase10_files[0], BODYWEIGHT_LBF)
print(f"Extracted {len(phase10_cycles)} gait cycle(s)")
phase10_summary = gait_summary_table(phase10_cycles)
phase10_summary

#%% [OPTIONAL PLOT] Gait — plotting helpers (per-cycle figure + CoP overlay)

def _plate_outline(dims, origin=(0.0, 0.0)):
    hw, hh = dims['width'] / 2, dims['height'] / 2
    xs = np.array([-hw, hw, hw, -hw, -hw]) - origin[0]
    ys = np.array([-hh, -hh, hh, hh, -hh]) - origin[1]
    return xs, ys


def _cop_frame(cycle, align):
    """CoP coordinates and the plate-outline origin shift for one cycle."""
    ox, oy = cycle['cop_origin'] if align else (0.0, 0.0)
    return cycle['cop_x'] - ox, cycle['cop_y'] - oy, (ox, oy)


def plot_gait_cycle(cycle, plate_label, align, tag):
    dims = get_plate_dims_in(cycle['phase'])
    cx, cy, origin = _cop_frame(cycle, align)
    ox_line, oy_line = _plate_outline(dims, origin)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("GRF (% bodyweight) vs. % stance",
                                        "CoP trajectory" + (" (heel strike = origin)" if align else "")))
    for key, name in [('fx', 'Fx (M/L)'), ('fy', 'Fy (A/P)'), ('fz', 'Fz (vertical)')]:
        fig.add_trace(go.Scatter(x=cycle['pct'], y=cycle[key], mode='lines', name=name),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=ox_line, y=oy_line, mode='lines', showlegend=False,
                             line=dict(color='black', width=2, dash='dash'),
                             hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines', name='CoP path',
                             line=dict(color='steelblue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[cx[0]], y=[cy[0]], mode='markers', name='Heel strike',
                             marker=dict(size=10, color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=[cx[-1]], y=[cy[-1]], mode='markers', name='Toe off',
                             marker=dict(size=10, color='red')), row=1, col=2)

    fig.update_xaxes(title_text="% stance (heel strike to toe off)", row=1, col=1)
    fig.update_yaxes(title_text="GRF (% BW)", row=1, col=1)
    fig.update_xaxes(title_text="CoP X, M/L (in)", row=1, col=2)
    fig.update_yaxes(title_text="CoP Y, A/P (in)", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=f"{plate_label} — {cycle['file_info']['basename']} — cycle {cycle['cycle']} "
              f"(stance {cycle['stance_sec']:.2f} s)",
        template="plotly_white", width=1300, height=550,
    )
    show_plot(fig, f"{tag}_cycle{cycle['cycle']:02d}")


def plot_cop_overlay(cycles, plate_label, align, tag):
    dims = get_plate_dims_in(cycles[0]['phase'])
    fig = go.Figure()

    if align:
        # Each cycle has its own shift, so each cycle gets its own (faint) outline
        for c in cycles:
            _, _, origin = _cop_frame(c, True)
            ox_line, oy_line = _plate_outline(dims, origin)
            fig.add_trace(go.Scatter(x=ox_line, y=oy_line, mode='lines', showlegend=False,
                                     line=dict(color='lightgray', width=1, dash='dash'),
                                     hoverinfo='skip'))
    else:
        ox_line, oy_line = _plate_outline(dims)
        fig.add_trace(go.Scatter(x=ox_line, y=oy_line, mode='lines', showlegend=False,
                                 line=dict(color='black', width=2, dash='dash'),
                                 hoverinfo='skip'))

    for c in cycles:
        cx, cy, _ = _cop_frame(c, align)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines', name=f"Cycle {c['cycle']}"))
        fig.add_trace(go.Scatter(x=[cx[0]], y=[cy[0]], mode='markers', showlegend=False,
                                 marker=dict(size=7, color='green'), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[cx[-1]], y=[cy[-1]], mode='markers', showlegend=False,
                                 marker=dict(size=7, color='red'), hoverinfo='skip'))

    fig.update_layout(
        title=f"{plate_label}: CoP trajectories, all cycles"
              + (" (heel strike = origin)" if align else " (plate frame)")
              + " — green = heel strike, red = toe off",
        xaxis_title="CoP X, M/L (in)", yaxis_title="CoP Y, A/P (in)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white", width=850, height=700,
    )
    show_plot(fig, f"{tag}_cop_overlay")

#%% [OPTIONAL PLOT] Phase 9 — per-cycle GRF + CoP, and CoP overlay
if SHOW_PLOTS:
    for c in phase9_cycles:
        plot_gait_cycle(c, "OR6-7-8000", ALIGN_COP_TO_HEEL_STRIKE, "phase9")
    plot_cop_overlay(phase9_cycles, "OR6-7-8000", ALIGN_COP_TO_HEEL_STRIKE, "phase9")

#%% [OPTIONAL PLOT] Phase 10 — per-cycle GRF + CoP, and CoP overlay
if SHOW_PLOTS:
    for c in phase10_cycles:
        plot_gait_cycle(c, "BP400600", ALIGN_COP_TO_HEEL_STRIKE, "phase10")
    plot_cop_overlay(phase10_cycles, "BP400600", ALIGN_COP_TO_HEEL_STRIKE, "phase10")

# %%