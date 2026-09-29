"""
Quickstart module for interactive use in Jupyter notebooks.
Imports the functions and constants notebook_analysis.py needs for
force plate validation.
Usage:
    from force_plate_validation.quickstart import *
"""

# Configuration and constants
from .config import (
    MATRIX_PHASE_1_2_3, MATRIX_PHASE_4, MATRIX_PHASE_5, MATRIX_PHASE_6,
    FT_TO_IN, M_TO_IN, Z_OFFSET_IN_BY_PLATE, PHASE_TO_PLATE, get_z_offset_in
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

# Common imports you'll likely want in a notebook
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os


def _print_welcome():
    """Print available functions and constants."""
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
    print("  • get_reader_and_matrix(phase)")
    print("  • convert_volt_to_force(df, matrix, moment_units)")
    print("  • convert_volt_to_mv(df)")
    print("  • butterworth_filter(signal, fs, cutoff_hz)")
    print("  • compute_fft_noise_signature(signal, fs)")

    print("\n📐 CONSTANTS & Z-OFFSET:")
    print(f"  • FT_TO_IN = {FT_TO_IN}")
    print(f"  • M_TO_IN = {M_TO_IN}")
    print(f"  • Z_OFFSET_IN_BY_PLATE = {Z_OFFSET_IN_BY_PLATE}")
    print("  • get_z_offset_in(phase) -> raises NotImplementedError if unconfirmed")

    print("\n💡 QUICK START EXAMPLE:")
    print("  raw_df, force_df = load_force_file('path/to/file.txt', phase=1)")
    print("  z_offset_in = get_z_offset_in(phase=1)")
    print("=" * 80)


_print_welcome()
