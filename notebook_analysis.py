#%% Import everything
from force_plate_validation.quickstart import *

#%% Define data directories
directories = [
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files",
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
]


#%% 5.1 Unloaded static noise testing

'''
Load all files containing the "static" keyword, summarize per-channel statistics for the raw
and filtered channel signals, and report peak-to-peak force/moment values in a tidy table.
'''

static_files = get_files_by_phase(directories, phase='all', keyword='static')
raw_channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
force_channel_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

voltage_stats_frames = []
force_ptp_frames = []

for file_info in static_files:
    raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])

    mv_df = convert_volt_to_mv(raw_df)
    filtered_mv_df = butterworth_filter(mv_df[raw_channel_cols], fs=1000, cutoff_hz=20)

    raw_stats = compute_channel_stats(mv_df, raw_channel_cols, units='mV', data_source='raw')
    filtered_stats = compute_channel_stats(filtered_mv_df, raw_channel_cols, units='mV', data_source='filtered')
    stats = pd.concat([raw_stats, filtered_stats], ignore_index=True)
    stats.insert(0, 'Phase', file_info['phase'])
    stats.insert(0, 'File', file_info['basename'])
    voltage_stats_frames.append(stats)

    filtered_force_df = butterworth_filter(force_df[force_channel_cols], fs=1000, cutoff_hz=20)
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')
    force_stats = compute_channel_stats(filtered_force_df, force_channel_cols,
                                         units=f"lbf / {moment_units}", data_source='filtered')
    force_stats.insert(0, 'Phase', file_info['phase'])
    force_stats.insert(0, 'File', file_info['basename'])
    force_ptp_frames.append(force_stats)

voltage_stats_df = pd.concat(voltage_stats_frames, ignore_index=True).sort_values(
    ['Phase', 'File', 'Data Source', 'Channel']).reset_index(drop=True)
force_ptp_df = pd.concat(force_ptp_frames, ignore_index=True).sort_values(
    ['Phase', 'File', 'Channel']).reset_index(drop=True)

# Word-table-ready views
voltage_table = voltage_stats_df[['File','Phase','Channel','Data Source','Mean','Std Dev','RMS']].round(4)
force_ptp_table = force_ptp_df[['File','Phase','Channel','Peak-to-Peak','Units']].round(4)

voltage_table

#%% 5.1a Static voltage: per-phase mean (SD), middle 50% of each trial — separate raw and filtered tables

CHANNEL_TO_AXIS = {
    'Ch1': 'Fx', 'Ch2': 'Fy', 'Ch3': 'Fz',
    'Ch4': 'Mx', 'Ch5': 'My', 'Ch6': 'Mz'
}

def middle_50_pct(df):
    n = len(df)
    start, end = int(n * 0.25), int(n * 0.75)
    return df.iloc[start:end]

def format_mean_sd(mean, sd, decimals=1):
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"

raw_rows, filt_rows = [], []
for phase in sorted(set(f['phase'] for f in static_files)):
    phase_files = [f for f in static_files if f['phase'] == phase]

    raw_chunks, filt_chunks = [], []
    for file_info in phase_files:
        raw_df, _ = load_force_file(file_info['filepath'], file_info['phase'])
        mv_df = convert_volt_to_mv(raw_df)
        filtered_mv_df = butterworth_filter(mv_df[raw_channel_cols], fs=1000, cutoff_hz=20)

        raw_chunks.append(middle_50_pct(mv_df[raw_channel_cols]))
        filt_chunks.append(middle_50_pct(filtered_mv_df))

    # Pool trimmed samples across all static trials for this phase before computing stats
    raw_pooled = pd.concat(raw_chunks, ignore_index=True)
    filt_pooled = pd.concat(filt_chunks, ignore_index=True)

    raw_row = {'Phase': phase}
    filt_row = {'Phase': phase}
    for ch, axis in CHANNEL_TO_AXIS.items():
        raw_row[f'{axis} mean (SD) [mV]'] = format_mean_sd(
            raw_pooled[ch].mean(), raw_pooled[ch].std(ddof=1)
        )
        filt_row[f'{axis} mean (SD) [mV]'] = format_mean_sd(
            filt_pooled[ch].mean(), filt_pooled[ch].std(ddof=1)
        )
    raw_rows.append(raw_row)
    filt_rows.append(filt_row)

raw_mean_sd_table = pd.DataFrame(raw_rows)
filtered_mean_sd_table = pd.DataFrame(filt_rows)

print('\nRaw static voltage — mean (SD) [mV] by phase (middle 50% of each trial):')
raw_mean_sd_table

# raw_mean_sd_table.to_clipboard(index=False)

#%% 5.1b Static force/moment peak-to-peak: per-phase mean PTP, raw vs. filtered

def compute_ptp_by_phase(get_signal_df, columns):
    """For each phase, average per-trial peak-to-peak across all static files."""
    rows = []
    for phase in sorted(set(f['phase'] for f in static_files)):
        phase_files = [f for f in static_files if f['phase'] == phase]
        trial_ptps = []
        moment_units = None
        for file_info in phase_files:
            signal_df, mu = get_signal_df(file_info)
            trial_ptps.append({col: np.ptp(signal_df[col].values) for col in columns})
            moment_units = mu
        ptp_df = pd.DataFrame(trial_ptps)
        row = {'Phase': phase}
        for col in columns:
            row[col] = ptp_df[col].mean()
        row['Moment Units'] = moment_units
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    return force_df, force_df.attrs.get('moment_units', 'lbf-in')

def _filtered_force_signal(file_info):
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    filtered = butterworth_filter(force_df[force_channel_cols], fs=1000, cutoff_hz=20)
    return filtered, force_df.attrs.get('moment_units', 'lbf-in')


raw_ptp_table = compute_ptp_by_phase(_raw_force_signal, force_channel_cols).round(2)
filtered_ptp_table = compute_ptp_by_phase(_filtered_force_signal, force_channel_cols).round(2)

print('\nRaw force/moment peak-to-peak (mean across trials) by phase [lbf / lbf-in or lbf-ft]:')
raw_ptp_table

# raw_ptp_table.to_clipboard(index=False)


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
ch3_filtered_mv = butterworth_filter(ch3_mv, fs=1000, cutoff_hz=20)

def compute_power_spectrum(signal, fs=1000.0):
    signal = signal - np.mean(signal)
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    power = np.abs(yf[:n // 2]) ** 2 / n
    return xf, power

xf_raw, power_raw = compute_power_spectrum(ch3_mv)
xf_filt, power_filt = compute_power_spectrum(ch3_filtered_mv)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xf_raw, y=power_raw, mode='lines', name='Raw', opacity=0.8))
fig.add_trace(go.Scatter(x=xf_filt, y=power_filt, mode='lines', name='Filtered (20 Hz cutoff)'))
fig.add_vline(x=20, line_dash='dash', line_color='gray',
              annotation_text='Cutoff (20 Hz)', annotation_position='top right')
fig.update_layout(
    width=900,
    height=500,
    title=f"Ch3 (Fz) FFT: Raw vs. Butterworth-Filtered — {target_file['basename']}",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Power",
    yaxis_type="log",
    template="plotly_white",
    legend_title="Signal",
     yaxis=dict(type="log", range=[np.log10(1e-6), np.log10(1e2)]),
)
fig.show()

# Quantify how much high-frequency content the filter removed
noise_sig_raw = compute_fft_noise_signature(ch3_mv, fs=1000)
noise_sig_filt = compute_fft_noise_signature(ch3_filtered_mv, fs=1000)

print(f"Raw      — line-noise share: {noise_sig_raw['line_noise_share_pct']:.1f}%, "
      f"high-freq (>20 Hz) share: {noise_sig_raw['high_freq_share_pct']:.1f}%")
print(f"Filtered — line-noise share: {noise_sig_filt['line_noise_share_pct']:.1f}%, "
      f"high-freq (>20 Hz) share: {noise_sig_filt['high_freq_share_pct']:.1f}%")


#%% 5.1e Drift trials: initial/final Fz averages (1 s windows) and CoP range — raw vs. filtered, phases 1-4

drift_files = get_files_by_phase(directories, phase='all', keyword=None)
drift_files = [f for f in drift_files
               if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]

print(f"Found {len(drift_files)} drift/center files")

N_SAMPLES_1SEC = 1000  # 1 kHz sampling rate -> 1000 samples = 1 second

def middle_95_pct(df):
    n = len(df)
    start, end = int(n * 0.025), int(n * 0.975)
    return df.iloc[start:end].reset_index(drop=True)

def compute_cop_timeseries(force_df, moment_units, z_offset_in=0.0):
    """Vectorized per-sample CoP (mm) for a full force_df."""
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
    return cop_x_in * IN_TO_MM, cop_y_in * IN_TO_MM


drift_rows = []
for file_info in drift_files:
    raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')

    # Trim to middle 95% to exclude samples right at recording start/stop
    force_trimmed = middle_95_pct(force_df)
    filtered_full = butterworth_filter(force_df[force_channel_cols], fs=1000, cutoff_hz=20)
    filtered_full['Time'] = force_df['Time'].values
    filtered_trimmed = middle_95_pct(filtered_full)

    # Initial/final 1-second Fz averages, within the trimmed window
    fz_raw_initial = force_trimmed['Fz'].values[:N_SAMPLES_1SEC].mean()
    fz_raw_final = force_trimmed['Fz'].values[-N_SAMPLES_1SEC:].mean()
    fz_filt_initial = filtered_trimmed['Fz'].values[:N_SAMPLES_1SEC].mean()
    fz_filt_final = filtered_trimmed['Fz'].values[-N_SAMPLES_1SEC:].mean()

    # CoP range (max - min) over the trimmed window, raw vs. filtered
    cop_x_raw, cop_y_raw = compute_cop_timeseries(force_trimmed, moment_units)
    cop_x_filt, cop_y_filt = compute_cop_timeseries(filtered_trimmed, moment_units)

    drift_rows.append({
        'Phase': file_info['phase'],
        'Fz raw initial (lbf)': fz_raw_initial,
        'Fz raw final (lbf)': fz_raw_final,
        'Fz filtered initial (lbf)': fz_filt_initial,
        'Fz filtered final (lbf)': fz_filt_final,
        'CoPx range filtered (mm)': np.ptp(cop_x_filt),
        'CoPy range filtered (mm)': np.ptp(cop_y_filt),
    })

drift_summary_table = pd.DataFrame(drift_rows).sort_values(['Phase',]).round(1).reset_index(drop=True)
drift_summary_table

#%% 5.1e (diagnostic) Phase 3: filtered PTP noise, static (unloaded) vs. center drift trial — per channel

phase3_static_files = [f for f in static_files if f['phase'] == 3]
phase3_drift_files = [f for f in drift_files if f['phase'] == 3]

if not phase3_static_files:
    raise FileNotFoundError("No Phase 3 static files found.")
if not phase3_drift_files:
    raise FileNotFoundError("No Phase 3 drift/center files found.")

def filtered_channel_ptp(file_info, trim_fn=middle_50_pct):
    raw_df, _ = load_force_file(file_info['filepath'], file_info['phase'])
    mv_df = convert_volt_to_mv(raw_df)
    filtered_mv_df = butterworth_filter(mv_df[raw_channel_cols], fs=1000, cutoff_hz=20)
    trimmed = trim_fn(filtered_mv_df)
    return {ch: np.ptp(trimmed[ch].values) for ch in raw_channel_cols}

# Average across trials if multiple files exist for phase 3 static/drift
static_ptp_trials = [filtered_channel_ptp(f, middle_50_pct) for f in phase3_static_files]
drift_ptp_trials = [filtered_channel_ptp(f, middle_95_pct) for f in phase3_drift_files]

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
print(f"Phase 3 static file(s): {[f['basename'] for f in phase3_static_files]}")
print(f"Phase 3 drift file(s):  {[f['basename'] for f in phase3_drift_files]}")
ground_check_table


#%% 5.1f Corner loading: mean Fx, Fy, Fz, CoPx, CoPy per phase/corner (middle 50%) — raw vs. filtered

corner_files = get_files_by_phase(directories, phase='all', keyword=None)
corner_files = [
    f for f in corner_files
    if any(k in f['basename'].lower() for k in ['corner', '_tl', '_tr', '_br', '_bl'])
    and 'static' not in f['basename'].lower()
    and 'drift' not in f['basename'].lower()
]

print(f"Found {len(corner_files)} corner files")

CORNER_LABELS = {'_tl': 'TL', '_tr': 'TR', '_br': 'BR', '_bl': 'BL'}
CORNER_ORDER = ['TL', 'TR', 'BR', 'BL']

def identify_corner(basename):
    basename_lower = basename.lower()
    for key, label in CORNER_LABELS.items():
        if key in basename_lower:
            return label
    return 'Unknown'

def mean_corner_metrics(force_df, moment_units, z_offset_in=0.0):
    fx = force_df['Fx'].mean()
    fy = force_df['Fy'].mean()
    fz = force_df['Fz'].mean()
    mx = force_df['Mx'].mean()
    my = force_df['My'].mean()

    if moment_units == 'lbf-ft':
        mx_in = mx * FT_TO_IN
        my_in = my * FT_TO_IN
    else:
        mx_in = mx
        my_in = my

    cop_x_in = (-my_in + (fx * z_offset_in)) / fz
    cop_y_in = (mx_in + (fy * z_offset_in)) / fz

    return {
        'Fx [lbf]': fx,
        'Fy [lbf]': fy,
        'Fz [lbf]': fz,
        'CoPx [in]': cop_x_in,
        'CoPy [in]': cop_y_in,
    }

raw_rows, filt_rows = [], []
for file_info in corner_files:
    _, force_df = load_force_file(file_info['filepath'], file_info['phase'])
    moment_units = force_df.attrs.get('moment_units', 'lbf-in')
    corner = identify_corner(file_info['basename'])

    filtered_df = butterworth_filter(force_df[force_channel_cols], fs=1000, cutoff_hz=20)
    filtered_df.attrs['moment_units'] = moment_units  # preserve, since filter returns a plain DataFrame

    # Trim to middle 50% before computing means, for consistency with (a)/(b)
    force_trimmed = middle_50_pct(force_df)
    filtered_trimmed = middle_50_pct(filtered_df)

    raw_metrics = mean_corner_metrics(force_trimmed, moment_units)
    filt_metrics = mean_corner_metrics(filtered_trimmed, moment_units)

    raw_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    filt_metrics.update({'Phase': file_info['phase'], 'Corner': corner})
    raw_rows.append(raw_metrics)
    filt_rows.append(filt_metrics)

col_order = ['Phase', 'Corner', 'Fx [lbf]', 'Fy [lbf]', 'Fz [lbf]', 'CoPx [in]', 'CoPy [in]']

corner_raw_table = pd.DataFrame(raw_rows)[col_order]
corner_filtered_table = pd.DataFrame(filt_rows)[col_order]

# Sort so each phase's 4 corners are grouped together, in TL/TR/BR/BL order (for Word row-merge)
corner_order_map = {c: i for i, c in enumerate(CORNER_ORDER)}
corner_raw_table = corner_raw_table.assign(_ord=corner_raw_table['Corner'].map(corner_order_map)) \
    .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True)
corner_filtered_table = corner_filtered_table.assign(_ord=corner_filtered_table['Corner'].map(corner_order_map)) \
    .sort_values(['Phase', '_ord']).drop(columns='_ord').reset_index(drop=True)


corner_raw_table = corner_raw_table.round(2)
corner_filtered_table = corner_filtered_table.round(2)

print('\nCorner loading — raw signal, middle 50% of trial, mean values by phase/corner:')
corner_raw_table
#%%




