#%% Import everything
from force_plate_validation.quickstart import *

#%% Define data directories
directories = [
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files",
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
]

#%% Configuration block

FS = 1000 # i.e. sampling rate

STATIC_TRIM_SECONDS = 2     # static noise trials are 30 seconds long
DRIFT_TRIM_SECONDS = 15     # center drift trials are 5 minutes long
CORNER_TRIM_SECONDS = 2     # corner loading trials are 30 seconds long
WARMUP_TRIM_SECONDS = 15    # warmup trials are 1 hour long

BUTTERWORTH_CUTOFF = 20 # filter cutoff frequency, Hz

raw_channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
force_channel_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

CHANNEL_TO_AXIS = {
    'Ch1': 'Fx', 'Ch2': 'Fy', 'Ch3': 'Fz',
    'Ch4': 'Mx', 'Ch5': 'My', 'Ch6': 'Mz'
}

#%% Shared trial trimming helper — fixed seconds off each end, not a percentage of trial length

def trim_seconds(df, seconds, time_col='Time'):
    """Drop the first/last `seconds` of a trial based on the Time column."""
    t_start = df[time_col].iloc[0]
    t_end = df[time_col].iloc[-1]
    mask = (df[time_col] >= t_start + seconds) & (df[time_col] <= t_end - seconds)
    trimmed = df[mask].reset_index(drop=True)
    if len(trimmed) == 0:
        raise ValueError(f"Trimming {seconds}s off each end leaves no data (trial too short).")
    return trimmed


#%% 5.1a Static voltage: per-phase mean (SD), trimmed — separate raw and filtered tables

static_files = get_files_by_phase(directories, phase='all', keyword='static')


def format_mean_sd(mean, sd, decimals=1):
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"

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

print(f'\nComplete. Raw static voltage — mean (SD) [mV] by phase (trimmed {STATIC_TRIM_SECONDS}s each end can now be copied to clipboard. \n Here is the raw table as an example:):')
raw_mean_sd_table

# raw_mean_sd_table.to_clipboard(index=False)

#%% 5.1b Static force/moment peak-to-peak: per-phase mean PTP, trimmed — raw vs. filtered

def compute_ptp_by_phase(get_signal_df, columns):
    rows = []
    for phase in sorted(set(f['phase'] for f in static_files)):
        phase_files = [f for f in static_files if f['phase'] == phase]
        trial_ptps = []
        moment_units = None
        for file_info in phase_files:
            signal_df, mu = get_signal_df(file_info)
            trimmed = trim_seconds(signal_df, STATIC_TRIM_SECONDS)
            ptp = {col: np.ptp(trimmed[col].values) for col in columns}
            # Normalize moments to lbf-in regardless of source units
            if mu == 'lbf-ft':
                for m in ['Mx', 'My', 'Mz']:
                    if m in ptp:
                        ptp[m] *= FT_TO_IN
            trial_ptps.append(ptp)
            moment_units = 'lbf-in'  # now always true after normalization
        ptp_df = pd.DataFrame(trial_ptps)
        row = {'Phase': phase}
        for col in columns:
            row[col] = ptp_df[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    return force_df, force_df.attrs.get('moment_units', 'lbf-in')

def _filtered_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    filtered = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered['Time'] = force_df['Time'].values
    return filtered, force_df.attrs.get('moment_units', 'lbf-in')

COLUMN_DISPLAY_NAMES = {
    'Fx': 'Fx range [lbf]',
    'Fy': 'Fy range [lbf]',
    'Fz': 'Fz range [lbf]',
    'Mx': 'Mx range [lbf-in]',
    'My': 'My range [lbf-in]',
    'Mz': 'Mz range [lbf-in]',
}

raw_ptp_table = compute_ptp_by_phase(_raw_force_signal, force_channel_cols).round(2).rename(columns=COLUMN_DISPLAY_NAMES)
filtered_ptp_table = compute_ptp_by_phase(_filtered_force_signal, force_channel_cols).round(2).rename(columns=COLUMN_DISPLAY_NAMES)

print(f'\nComplete. Force/moment peak-to-peak (trimmed {STATIC_TRIM_SECONDS}s each end) by phase can now be copied to clipboard. \n Here is the raw table as an example:')
raw_ptp_table

#%% 5.1c Representative noise histogram: Ch3 (Fz) raw voltage, Phase 1 static trial 1

target_file = next(
    (f for f in static_files if 'phase1' in f['basename'].lower() and 'trial1' in f['basename'].lower()),
    None
)
if target_file is None:
    raise FileNotFoundError("Could not find a Phase 1 static trial 1 file matching 'phase1'/'trial1' in filename.")

print(f"Using file: {target_file['basename']}")

raw_df, _ = load_force_file(target_file['filepath'], target_file['phase'])
mv_df = convert_volt_to_mv(raw_df)
ch3_mv = mv_df['Ch3'].values

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=ch3_mv,
    nbinsx=60,
    marker=dict(line=dict(width=0.5, color='white'))
))
fig.update_layout(
    title=f"Ch3 (Fz) Raw Voltage Noise Histogram — {target_file['basename']}",
    xaxis_title="Voltage (mV)",
    yaxis_title="Count",
    template="plotly_white",
    bargap=0.02
)
fig.show()

print(f"Ch3 raw voltage: mean = {ch3_mv.mean():.4f} mV, std = {ch3_mv.std(ddof=1):.4f} mV, "
      f"n = {len(ch3_mv)} samples")


#%% 5.1d Representative FFT: Ch3 (Fz) raw voltage, before/after Butterworth filter — Phase 1 static trial 1

from scipy.fft import fft, fftfreq

# Reuses target_file/mv_df/ch3_mv from 5.1c
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

fig = go.Figure()
fig.add_trace(go.Scatter(x=xf_raw, y=power_raw, mode='lines', name='Raw', opacity=0.8))
fig.add_trace(go.Scatter(x=xf_filt, y=power_filt, mode='lines', name=f'Filtered ({BUTTERWORTH_CUTOFF} Hz cutoff)'))
fig.add_vline(x=20, line_dash='dash', line_color='gray',
              annotation_text='Cutoff (20 Hz)', annotation_position='top right')
fig.update_layout(
    width=900,
    height=500,
    title=f"Ch3 (Fz) FFT: Raw vs. Butterworth-Filtered — {target_file['basename']}",
    xaxis_title="Frequency [Hz]",
    yaxis_title="Power",
    yaxis_type="log",
    template="plotly_white",
    legend_title="Signal",
     yaxis=dict(type="log", range=[np.log10(1e-6), np.log10(1e2)]),
)
fig.show()

# Quantify how much high-frequency content the filter removed
noise_sig_raw = compute_fft_noise_signature(ch3_mv, fs=FS)
noise_sig_filt = compute_fft_noise_signature(ch3_filtered_mv, fs=FS)

print(f"Raw      — line-noise share: {noise_sig_raw['line_noise_share_pct']:.1f}%, "
      f"high-freq (>{BUTTERWORTH_CUTOFF:.0f} Hz) share: {noise_sig_raw['high_freq_share_pct']:.1f}%")
print(f"Filtered — line-noise share: {noise_sig_filt['line_noise_share_pct']:.1f}%, "
      f"high-freq (>{BUTTERWORTH_CUTOFF:.0f} Hz) share: {noise_sig_filt['high_freq_share_pct']:.1f}%")


#%% 5.2 Drift trials: initial/final Fz averages (1 s windows) and CoP range — raw vs. filtered, phases 1-4

drift_files = get_files_by_phase(directories, phase='all', keyword=None)
drift_files = [f for f in drift_files
               if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]

print(f"Found {len(drift_files)} drift/center files")


def compute_cop_timeseries(force_df, moment_units, phase):
    """Vectorized per-sample CoP (in) for a full force_df, using the plate's z-offset."""
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
    return cop_x_in , cop_y_in


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

    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_trimmed, moment_units,file_info['phase'])

    drift_rows.append({
        'Phase': file_info['phase'],
        'Fz raw initial [lbf]': fz_raw_initial,
        'Fz raw final [lbf]': fz_raw_final,
        'Fz filtered initial [lbf]': fz_filt_initial,
        'Fz filtered final [lbf]': fz_filt_final,
        'CoPx range filtered [in]': np.ptp(cop_x_filt),
        'CoPy range filtered [in]': np.ptp(cop_y_filt),
    })

drift_summary_table = pd.DataFrame(drift_rows).sort_values(['Phase']).round(2).reset_index(drop=True)
drift_summary_table

#%% 5.2 (diagnostic) Phase 3: filtered PTP noise, static (unloaded) vs. center drift trial — per channel

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

static_ptp_trials = [filtered_channel_ptp(f, STATIC_TRIM_SECONDS) for f in phase3_static_files]
drift_ptp_trials = [filtered_channel_ptp(f, DRIFT_TRIM_SECONDS) for f in phase3_drift_files]

static_ptp_avg = pd.DataFrame(static_ptp_trials).mean()
drift_ptp_avg = pd.DataFrame(drift_ptp_trials).mean()

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


#%% 5.3 Corner loading: mean Fx, Fy, Fz, CoPx, CoPy per phase/corner (trimmed) — raw vs. filtered

corner_files = get_files_by_phase(directories, phase='all', keyword=None)
corner_files = [
    f for f in corner_files
    if f['phase'] in (1, 2, 3, 4)  # corner-loading trials only exist for phases 1-4
    and any(k in f['basename'].lower() for k in ['corner', '_tl', '_tr', '_br', '_bl'])
    and 'static' not in f['basename'].lower()
    and 'drift' not in f['basename'].lower()
]

print(f"Found {len(corner_files)} corner files")
for f in corner_files:
    print(f"  Phase {f['phase']}: {f['basename']}")

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

    raw_metrics = mean_corner_metrics(force_trimmed, moment_units,file_info['phase'])
    filt_metrics = mean_corner_metrics(filtered_trimmed, moment_units,file_info['phase'])

    raw_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    filt_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    raw_rows.append(raw_metrics)
    filt_rows.append(filt_metrics)

col_order = ['Phase', 'Corner', 'Fx [lbf]', 'Fy [lbf]', 'Fz [lbf]', 'CoPx [in]', 'CoPy [in]']
corner_raw_table = pd.DataFrame(raw_rows)[col_order]
corner_filtered_table = pd.DataFrame(filt_rows)[col_order]

corner_order_map = {c: i for i, c in enumerate(CORNER_ORDER)}
corner_raw_table = corner_raw_table.assign(_ord=corner_raw_table['Corner'].map(corner_order_map)) \
    .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True)
corner_filtered_table = corner_filtered_table.assign(_ord=corner_filtered_table['Corner'].map(corner_order_map)) \
    .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True)

corner_raw_table = corner_raw_table.round(2)
corner_filtered_table = corner_filtered_table.round(2)
corner_raw_table

#%% 5.4 Phase 5/6 Warm-up test: 

warmup_files_5 = get_files_by_phase(directories, phase=5, keyword=None)
warmup_files_6 = get_files_by_phase(directories, phase=6, keyword=None)

print(f"Found {len(warmup_files_5)} Phase 5 (OR6-7-8000) warm-up file(s)")
print(f"Found {len(warmup_files_6)} Phase 6 (BP400600) warm-up file(s)")

warmup_data = {}

for f in warmup_files_5:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    warmup_data['OR6-7-8000'] = {'raw_df': raw_df, 'force_df': force_df, 'basename': f['basename'], 'phase': f['phase']}

for f in warmup_files_6:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    warmup_data['BP400600'] = {'raw_df': raw_df, 'force_df': force_df, 'basename': f['basename'], 'phase': f['phase']}


warmup_rows = []
for plate, data in warmup_data.items():
    force_df = data['force_df']
    phase = data['phase']
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    force_trimmed = trim_seconds(force_df, seconds=WARMUP_TRIM_SECONDS)
    filtered_full = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_full['Time'] = force_df['Time'].values
    filtered_trimmed = trim_seconds(filtered_full, seconds=WARMUP_TRIM_SECONDS)

    fz_raw_initial = force_trimmed['Fz'].values[:FS].mean()
    fz_raw_final = force_trimmed['Fz'].values[-FS:].mean()
    fz_filt_initial = filtered_trimmed['Fz'].values[:FS].mean()
    fz_filt_final = filtered_trimmed['Fz'].values[-FS:].mean()

    # Filtered CoP range only — raw CoP range is unreliable near-zero-Fz division artifact (established in 5.1e)
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


#%% 5.4 Phase 5/6: Fz vs. elapsed time — BP400600 vs. OR6-7-8000 warm-up comparison

fig = go.Figure()
for plate, data in warmup_data.items():
    force_df = data['force_df']
    filtered_fz = butterworth_filter(force_df['Fz'].values, fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    elapsed_min = (force_df['Time'].values - force_df['Time'].values[0]) / 60.0
    fig.add_trace(go.Scatter(x=elapsed_min, y=filtered_fz, mode='lines', name=f"{plate} (filtered)"))

fig.update_layout(
    title="Phase 5/6 Warm-Up Test: Fz (Vertical Force) vs. Elapsed Time",
    xaxis_title="Elapsed Time [min]",
    yaxis_title="Fz [lbf]",
    template="plotly_white",
    legend_title="Force Plate",
    width=900,
    height=500
)
fig.show()

#%% Phase 7: discover perimeter/diagonal traverse files

phase7_files = get_files_by_phase(directories, phase=7, keyword=None)
for f in phase7_files:
    name_lower = f['basename'].lower()
    f['path_type'] = 'perimeter' if 'perimeter' in name_lower else ('diagonal' if 'diagonal' in name_lower else 'unknown')

print(f"Found {len(phase7_files)} phase 7 file(s)")
for f in phase7_files:
    print(f"  {f['basename']} -> {f['path_type']}")

#%% Stationary-segment detector v2: velocity-based, robust to slow ramps

def detect_static_segments(time, cop_x, cop_y,
                            velocity_smooth_sec=0.3, velocity_threshold_in_s=0.01,
                            min_duration_sec=2.0, trim_sec=1.0):
    # Identify index ranges where the CoP is stationary, based on position VELOCITY
    # (not local std) — robust to slow/gradual transitions between dwell points.
    # velocity_smooth_sec     : smoothing window applied to the velocity signal
    # velocity_threshold_in_s : below this smoothed speed (in/s), treated as stationary
    # min_duration_sec        : minimum dwell length to count as a real static hold
    # trim_sec                : seconds trimmed off each end of a detected segment

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

#%% Diagnostic v2: CoP magnitude + velocity trace, both with detected segments overlaid

diagnostic_file = phase7_files[0]
raw_df, force_df = load_force_file(diagnostic_file['filepath'], diagnostic_file['phase'])
moment_units = force_df.attrs.get('moment_units', 'lbf-in')

filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
filtered_df['Time'] = force_df['Time'].values

cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_df, moment_units, diagnostic_file['phase'])
time = filtered_df['Time'].values
cop_mag = np.sqrt(cop_x_filt**2 + cop_y_filt**2)

VELOCITY_THRESHOLD_IN_S_OR6 = 0.079   # phase 7 / OR6-7-8000

segments, velocity_smooth = detect_static_segments(
    time, cop_x_filt, cop_y_filt, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6
)
print(f"File: {diagnostic_file['basename']} ({diagnostic_file['path_type']})")
print(f"Detected {len(segments)} static segment(s) at threshold {VELOCITY_THRESHOLD_IN_S_OR6} in/s")

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=cop_mag, mode='lines', name='CoP magnitude (in)', line=dict(color='steelblue')))
for i, (s, e) in enumerate(segments):
    fig.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.2, line_width=0)
fig.update_layout(title=f"CoP magnitude — {diagnostic_file['basename']}",
                   xaxis_title="Time (s)", yaxis_title="CoP magnitude (in)",
                   template="plotly_white", width=1000, height=350)
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=time, y=velocity_smooth, mode='lines', name='Smoothed speed (in/s)', line=dict(color='darkorange')))
fig2.add_hline(y=VELOCITY_THRESHOLD_IN_S_OR6, line_dash='dash', line_color='red', annotation_text='threshold')
for i, (s, e) in enumerate(segments):
    fig2.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.15, line_width=0)
fig2.update_layout(title="CoP speed (the actual signal being thresholded)",
                    xaxis_title="Time (s)", yaxis_title="Speed (in/s)",
                    template="plotly_white", width=1000, height=350)
fig2.show()

#%% Diagnostic: Fz vs. Time and Fz vs. CoP magnitude, one plot per phase 7 file

from plotly.subplots import make_subplots

def plot_fz_stability_diagnostic(file_info, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6):
    raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_df['Time'] = force_df['Time'].values

    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_df, moment_units, file_info['phase'])
    time = filtered_df['Time'].values
    fz_filt = filtered_df['Fz'].values
    cop_mag = np.sqrt(cop_x_filt**2 + cop_y_filt**2)

    segments, _ = detect_static_segments(time, cop_x_filt, cop_y_filt, velocity_threshold_in_s=velocity_threshold_in_s)

    in_segment = np.zeros(len(time), dtype=bool)
    for s, e in segments:
        in_segment[s:e] = True

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Fz vs. Time", "Fz vs. CoP Magnitude"))

    fig.add_trace(go.Scatter(x=time, y=fz_filt, mode='lines', name='Fz (filtered)',
                              line=dict(color='steelblue')), row=1, col=1)
    for s, e in segments:
        seg_fz = fz_filt[s:e]
        fig.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.15, line_width=0, row=1, col=1)
        fig.add_annotation(x=(time[s] + time[e - 1]) / 2, y=fz_filt[s:e].max(),
                            text=f"\u03c3={seg_fz.std(ddof=1):.3f}", showarrow=False,
                            font=dict(size=9, color='darkgreen'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cop_mag[~in_segment], y=fz_filt[~in_segment], mode='markers',
        marker=dict(size=3, color='lightgray'), name='Outside segment', opacity=0.5
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=cop_mag[in_segment], y=fz_filt[in_segment], mode='markers',
        marker=dict(size=4, color='green'), name='Inside segment', opacity=0.7
    ), row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Fz (lbf)", row=1, col=1)
    fig.update_xaxes(title_text="CoP magnitude (in)", row=1, col=2)
    fig.update_yaxes(title_text="Fz (lbf)", row=1, col=2)

    fig.update_layout(
        title=f"Fz stability check — {file_info['basename']} ({file_info['path_type']}) — {len(segments)} segment(s)",
        template="plotly_white", width=1200, height=500, showlegend=True
    )
    fig.show()

    print(f"{file_info['basename']} ({file_info['path_type']}): {len(segments)} segment(s)")
    for i, (s, e) in enumerate(segments):
        seg_fz = fz_filt[s:e]
        print(f"  Segment {i+1}: n={e-s}, Fz mean={seg_fz.mean():.3f} lbf, "
              f"Fz std={seg_fz.std(ddof=1):.3f} lbf, Fz range={np.ptp(seg_fz):.3f} lbf")
    print()


for f in phase7_files:
    plot_fz_stability_diagnostic(f, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6)
    
#%% Process all phase 7 files: extract per-segment mean CoP position + Fz

spatial_rows = []
for f in phase7_files:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_df['Time'] = force_df['Time'].values

    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_df, moment_units, f['phase'])
    time = filtered_df['Time'].values
    fz_filt = filtered_df['Fz'].values

    segments, _ = detect_static_segments(
        time, cop_x_filt, cop_y_filt, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_OR6
    )
    print(f"{f['basename']} ({f['path_type']}): {len(segments)} static segment(s)")

    for seg_idx, (s, e) in enumerate(segments):
        spatial_rows.append({
            'File': f['basename'],
            'Path Type': f['path_type'],
            'Segment': seg_idx,
            'CoPx [in]': np.mean(cop_x_filt[s:e]),
            'CoPy [in]': np.mean(cop_y_filt[s:e]),
            'Fz [lbf]': np.mean(fz_filt[s:e]),
            'Fz Std [lbf]': np.std(fz_filt[s:e], ddof=1),
        })

spatial_uniformity_df = pd.DataFrame(spatial_rows)
print(f"\nTotal static segments across all files: {len(spatial_uniformity_df)}")
spatial_uniformity_df

#%% Spatial Fz uniformity map: 2D scatter, colored by Fz reading (explicit colorbar formatting)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=spatial_uniformity_df['CoPx [in]'],
    y=spatial_uniformity_df['CoPy [in]'],
    mode='markers',
    marker=dict(
        size=12,
        color=spatial_uniformity_df['Fz [lbf]'],
        colorscale='Viridis',
        showscale=True,          # explicitly force the colorbar on
        colorbar=dict(
            title=dict(text='Fz [lbf]', side='right'),
            thickness=15,
            len=0.75,
            x=1.02                # nudge right so it doesn't overlap the plot area
        ),
        line=dict(width=1, color='black')
    ),
    text=spatial_uniformity_df['File'] + ' — seg ' + spatial_uniformity_df['Segment'].astype(str),
    hovertemplate='CoPx: %{x:.1f}in<br>CoPy: %{y:.1f} in<br>Fz: %{marker.color:.2f} lbf<br>%{text}<extra></extra>'
))
fig.update_layout(
    title="OR6-7-8000: Vertical Force (Fz) vs. Position on Plate Surface",
    xaxis_title="CoP X (in)", yaxis_title="CoP Y (in)",
    template="plotly_white", width=850, height=700,
    yaxis=dict(scaleanchor="x", scaleratio=1),
    margin=dict(r=100)   # extra right margin so the colorbar isn't clipped
)
fig.show()

print(f"Fz range across all static positions: {spatial_uniformity_df['Fz [lbf]'].min():.2f} to "
      f"{spatial_uniformity_df['Fz [lbf]'].max():.2f} lbf "
      f"(spread: {spatial_uniformity_df['Fz [lbf]'].max() - spatial_uniformity_df['Fz [lbf]'].min():.2f} lbf)")

#%% Phase 8: discover perimeter/diagonal traverse files (BP400600)

phase8_files = get_files_by_phase(directories, phase=8, keyword=None)
for f in phase8_files:
    name_lower = f['basename'].lower()
    f['path_type'] = 'perimeter' if 'perimeter' in name_lower else ('diagonal' if 'diagonal' in name_lower else 'unknown')

print(f"Found {len(phase8_files)} phase 8 file(s)")
for f in phase8_files:
    print(f"  {f['basename']} -> {f['path_type']}")

#%% determine velocity threshold for static segment detection (BP400600) — diagnostic plot

diagnostic_file = phase8_files[0]
raw_df, force_df = load_force_file(diagnostic_file['filepath'], diagnostic_file['phase'])
moment_units = force_df.attrs.get('moment_units', 'lbf-in')

filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
filtered_df['Time'] = force_df['Time'].values

cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_df, moment_units, diagnostic_file['phase'])
time = filtered_df['Time'].values
cop_mag = np.sqrt(cop_x_filt**2 + cop_y_filt**2)

VELOCITY_THRESHOLD_IN_S_BP = 2.76    # phase 8 / BP400600

segments, velocity_smooth = detect_static_segments(
    time, cop_x_filt, cop_y_filt, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_BP
)
print(f"File: {diagnostic_file['basename']} ({diagnostic_file['path_type']})")
print(f"Detected {len(segments)} static segment(s) at threshold {VELOCITY_THRESHOLD_IN_S_BP} in/s")

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=cop_mag, mode='lines', name='CoP magnitude (in)', line=dict(color='steelblue')))
for i, (s, e) in enumerate(segments):
    fig.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.2, line_width=0)
fig.update_layout(title=f"CoP magnitude — {diagnostic_file['basename']}",
                   xaxis_title="Time (s)", yaxis_title="CoP magnitude (in)",
                   template="plotly_white", width=1000, height=350)
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=time, y=velocity_smooth, mode='lines', name='Smoothed speed (in/s)', line=dict(color='darkorange')))
fig2.add_hline(y=VELOCITY_THRESHOLD_IN_S_BP, line_dash='dash', line_color='red', annotation_text='threshold')
for i, (s, e) in enumerate(segments):
    fig2.add_vrect(x0=time[s], x1=time[e - 1], fillcolor='green', opacity=0.15, line_width=0)
fig2.update_layout(title="CoP speed (the actual signal being thresholded)",
                    xaxis_title="Time (s)", yaxis_title="Speed (in/s)",
                    template="plotly_white", width=1000, height=350)
fig2.show()

#%% Diagnostic: Fz stability check, one plot per phase 8 file (reuses plot_fz_stability_diagnostic from phase 7)

for f in phase8_files:
    plot_fz_stability_diagnostic(f, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_BP)

#%% Process all phase 8 files: extract per-segment mean CoP position + Fz

spatial_rows_bp = []
for f in phase8_files:
    raw_df, force_df = load_force_file(f['filepath'], f['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=FS, cutoff_hz=BUTTERWORTH_CUTOFF)
    filtered_df['Time'] = force_df['Time'].values

    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_df, moment_units, f['phase'])
    time = filtered_df['Time'].values
    fz_filt = filtered_df['Fz'].values

    segments, _ = detect_static_segments(
        time, cop_x_filt, cop_y_filt, velocity_threshold_in_s=VELOCITY_THRESHOLD_IN_S_BP
    )
    print(f"{f['basename']} ({f['path_type']}): {len(segments)} static segment(s)")

    for seg_idx, (s, e) in enumerate(segments):
        spatial_rows_bp.append({
            'File': f['basename'],
            'Path Type': f['path_type'],
            'Segment': seg_idx,
            'CoPx [in]': np.mean(cop_x_filt[s:e]),
            'CoPy [in]': np.mean(cop_y_filt[s:e]),
            'Fz [lbf]': np.mean(fz_filt[s:e]),
            'Fz Std [lbf]': np.std(fz_filt[s:e], ddof=1),
        })

spatial_uniformity_df_bp = pd.DataFrame(spatial_rows_bp)
print(f"\nTotal static segments across all phase 8 files: {len(spatial_uniformity_df_bp)}")
spatial_uniformity_df_bp

#%% Spatial Fz uniformity map — BP400600

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=spatial_uniformity_df_bp['CoPx [in]'],
    y=spatial_uniformity_df_bp['CoPy [in]'],
    mode='markers',
    marker=dict(
        size=12,
        color=spatial_uniformity_df_bp['Fz [lbf]'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title=dict(text='Fz [lbf]', side='right'), thickness=15, len=0.75, x=1.02),
        line=dict(width=1, color='black')
    ),
    text=spatial_uniformity_df_bp['File'] + ' — seg ' + spatial_uniformity_df_bp['Segment'].astype(str),
    hovertemplate='CoPx: %{x:.1f} in<br>CoPy: %{y:.1f} in<br>Fz: %{marker.color:.2f} lbf<br>%{text}<extra></extra>'
))
fig.update_layout(
    title="BP400600: Vertical Force (Fz) vs. Position on Plate Surface",
    xaxis_title="CoP X (in)", yaxis_title="CoP Y (in)",
    template="plotly_white", width=850, height=700,
    yaxis=dict(scaleanchor="x", scaleratio=1),
    margin=dict(r=100)
)
fig.show()

print(f"Fz range across all static positions: {spatial_uniformity_df_bp['Fz [lbf]'].min():.2f} to "
      f"{spatial_uniformity_df_bp['Fz [lbf]'].max():.2f} lbf "
      f"(spread: {spatial_uniformity_df_bp['Fz [lbf]'].max() - spatial_uniformity_df_bp['Fz [lbf]'].min():.2f} lbf)")

#%%




