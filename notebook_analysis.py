#%% Import everything
from force_plate_validation.quickstart import *

#%% Define data directories
directories = [
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files",
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
]

#%% Explore available files

all_files = get_files_by_phase(directories, phase='all')
print(f"Found {len(all_files)} files")
all_files_df = pd.DataFrame(all_files)
all_files_df.head()

#%% 5.1 Unloaded static noise testing

'''
Load all files containing the "static" keyword, summarize per-channel statistics for the raw
and filtered channel signals, and report peak-to-peak force/moment values in a tidy table.
'''

#%% 5.1 Unloaded static noise testing

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


#%% EXAMPLES 
# 
# Load and visualize a single file
file_info = get_files_by_phase(directories, phase=1, keyword='static')[0]
raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])

print(f"Loaded: {file_info['basename']}")
print(f"Shape: {raw_df.shape}")
print(f"Duration: {raw_df['Time'].iloc[-1] - raw_df['Time'].iloc[0]:.1f} seconds")

# Plot raw data
plot_interactive_timeseries(raw_df, title=f"Raw: {file_info['basename']}")

# Plot FFT
plot_fft(raw_df, channel='Ch3', title=f"FFT Ch3: {file_info['basename']}")

# Compute noise statistics
mv_df = convert_volt_to_mv(raw_df)
filtered_df = butterworth_filter(mv_df, fs=1000, cutoff_hz=20)
stats = compute_channel_stats(mv_df, ['Ch1','Ch2','Ch3','Ch4','Ch5','Ch6'], units='mV')
stats

# Compare raw vs filtered
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw_df['Time'][:1000], y=mv_df['Ch1'][:1000], 
                         mode='lines', name='Raw'))
fig.add_trace(go.Scatter(x=raw_df['Time'][:1000], y=filtered_df['Ch1'][:1000], 
                         mode='lines', name='Filtered'))
fig.update_layout(title='Raw vs Filtered (Ch1, first second)', 
                 xaxis_title='Time (s)', yaxis_title='mV')
fig.show()

# Quick FFT noise analysis
noise_sig = compute_fft_noise_signature(raw_df['Ch3'].values, fs=1000)
print(f"Line noise share: {noise_sig['line_noise_share_pct']:.1f}%")
print(f"High freq share: {noise_sig['high_freq_share_pct']:.1f}%")

# Cell 10: Run full validation (optional - can be slow)
# raw_noise, force_noise = run_static_noise_test(directories)
# drift_results = run_drift_test(directories)
# corner_results = run_corner_test(directories)
# %%
