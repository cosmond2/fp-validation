#%% Cell 1: Import everything
from force_plate_validation.quickstart import *

#%% Cell 2: Define your data directories
directories = [
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files",
    r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
]

#%% Cell 3: Explore available files
all_files = get_files_by_phase(directories, phase='all')
print(f"Found {len(all_files)} files")
all_files_df = pd.DataFrame(all_files)
all_files_df.head()

#%% Cell 4: Load and visualize a single file
file_info = get_files_by_phase(directories, phase=1, keyword='static')[0]
raw_df, force_df = load_force_file(file_info['filepath'], file_info['phase'])

print(f"Loaded: {file_info['basename']}")
print(f"Shape: {raw_df.shape}")
print(f"Duration: {raw_df['Time'].iloc[-1] - raw_df['Time'].iloc[0]:.1f} seconds")

# Cell 5: Plot raw data
plot_interactive_timeseries(raw_df, title=f"Raw: {file_info['basename']}")

# Cell 6: Plot FFT
plot_fft(raw_df, channel='Ch3', title=f"FFT Ch3: {file_info['basename']}")

# Cell 7: Compute noise statistics
mv_df = convert_volt_to_mv(raw_df)
filtered_df = butterworth_filter(mv_df, fs=1000, cutoff_hz=20)
stats = compute_channel_stats(mv_df, ['Ch1','Ch2','Ch3','Ch4','Ch5','Ch6'], units='mV')
stats

# Cell 8: Compare raw vs filtered
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw_df['Time'][:1000], y=mv_df['Ch1'][:1000], 
                         mode='lines', name='Raw'))
fig.add_trace(go.Scatter(x=raw_df['Time'][:1000], y=filtered_df['Ch1'][:1000], 
                         mode='lines', name='Filtered'))
fig.update_layout(title='Raw vs Filtered (Ch1, first second)', 
                 xaxis_title='Time (s)', yaxis_title='mV')
fig.show()

# Cell 9: Quick FFT noise analysis
noise_sig = compute_fft_noise_signature(raw_df['Ch3'].values, fs=1000)
print(f"Line noise share: {noise_sig['line_noise_share_pct']:.1f}%")
print(f"High freq share: {noise_sig['high_freq_share_pct']:.1f}%")

# Cell 10: Run full validation (optional - can be slow)
# raw_noise, force_noise = run_static_noise_test(directories)
# drift_results = run_drift_test(directories)
# corner_results = run_corner_test(directories)
# %%
