"""
Quickstart module for interactive use in Jupyter notebooks.
Imports all commonly needed functions and constants for force plate validation.
Usage:
    from force_plate_validation.quickstart import *
"""

# Configuration and constants
from .config import (
    MATRIX_PHASE_1_2_3, MATRIX_PHASE_4,
    LBF_TO_N, IN_TO_MM, FT_TO_IN,
    BASE_PLATE_DIA, DIMS_BP400600, DIMS_OR67800,
    PHASE_LABELS, APPLIED_LOAD_LBS, 
    DRIFT_RECORDING_SECONDS, DEFAULT_FS, DEFAULT_CUTOFF_HZ
)

# File I/O
from .file_io import (
    determine_phase,
    get_files_by_phase,
    read_ni_daq,
    read_powerlab_daq
)

# Calibration
from .calibration import (
    get_reader_and_matrix,
    convert_volt_to_force,
    convert_volt_to_mv,
    load_force_file
)

# Signal processing
from .signal_processing import (
    butterworth_filter,
    compute_fft_noise_signature
)

# Computation
from .compute import (
    compute_channel_stats,
    compute_static_noise,
    compute_drift,
    compute_corner_loading,
    compute_phase3_center_drift_channel_metrics,
    summarize_loaded_fft_noise,
    compare_center_corner_fx_fy
)

# Criteria
from .criteria import (
    apply_noise_criteria,
    apply_drift_criteria,
    apply_weight_and_crosstalk_criteria,
    apply_cop_criteria
)

# Visualization
from .visualization import (
    plot_interactive_timeseries,
    plot_fft,
    plot_cop_scatter,
    plot_component_comparison
)

# Runners
from .runners import (
    run_static_noise_test,
    run_drift_test,
    run_corner_test
)

# Common imports you'll likely want in a notebook
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os

# Print a welcome message with available functions
def _print_welcome():
    """Print available functions and their docstrings."""
    print("=" * 80)
    print("Force Plate Validation - Quickstart Module Loaded")
    print("=" * 80)
    print("\n📁 FILE I/O:")
    print("  • determine_phase(filename)")
    print("  • get_files_by_phase(directories, phase, keyword)")
    print("  • read_ni_daq(file_path)")
    print("  • read_powerlab_daq(file_path)")
    print("  • load_force_file(filepath, phase) -> (raw_df, force_df)")
    
    print("\n🔧 CALIBRATION & PROCESSING:")
    print("  • convert_volt_to_force(df, matrix, phase)")
    print("  • convert_volt_to_mv(df)")
    print("  • butterworth_filter(signal, fs, cutoff_hz)")
    print("  • compute_fft_noise_signature(signal, fs)")
    
    print("\n📊 COMPUTATION:")
    print("  • compute_channel_stats(df, columns, units)")
    print("  • compute_static_noise(raw_df, force_df)")
    print("  • compute_drift(force_df, n_samples)")
    print("  • compute_corner_loading(force_df, moment_units)")
    
    print("\n✅ CRITERIA:")
    print("  • apply_noise_criteria(force_noise_df)")
    print("  • apply_drift_criteria(drift_df)")
    print("  • apply_weight_and_crosstalk_criteria(corner_df)")
    print("  • apply_cop_criteria(corner_df, true_cop_by_phase_corner)")
    
    print("\n📈 VISUALIZATION:")
    print("  • plot_interactive_timeseries(df, title)")
    print("  • plot_fft(df, channel)")
    print("  • plot_cop_scatter(cop_x, cop_y)")
    print("  • plot_component_comparison(...)")
    
    print("\n🏃 RUNNERS:")
    print("  • run_static_noise_test(directories)")
    print("  • run_drift_test(directories)")
    print("  • run_corner_test(directories)")
    
    print("\n📐 CONSTANTS:")
    print(f"  • LBF_TO_N = {LBF_TO_N}")
    print(f"  • IN_TO_MM = {IN_TO_MM}")
    print(f"  • FT_TO_IN = {FT_TO_IN}")
    print(f"  • BASE_PLATE_DIA = {BASE_PLATE_DIA} mm")
    print(f"  • APPLIED_LOAD_LBS = {APPLIED_LOAD_LBS}")
    
    print("\n💡 QUICK START EXAMPLES:")
    print("  # Load and plot a single file:")
    print("  raw_df, force_df = load_force_file('path/to/file.txt', phase=1)")
    print("  plot_interactive_timeseries(raw_df, title='Raw Data')")
    print("  plot_fft(raw_df, channel='Ch3')")
    print()
    print("  # Run full validation:")
    print("  raw_noise, force_noise = run_static_noise_test(directories)")
    print("  drift_results = run_drift_test(directories)")
    print("  corner_results = run_corner_test(directories)")
    print("=" * 80)

_print_welcome()