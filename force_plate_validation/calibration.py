"""Calibration matrix application and force conversion functions."""
import numpy as np
import pandas as pd
from .config import MATRIX_PHASE_1_2_3, MATRIX_PHASE_4
from .file_io import read_ni_daq, read_powerlab_daq


def get_reader_and_matrix(phase):
    """Map phase to reader function, calibration matrix, and moment units."""
    if phase == 4:
        return read_powerlab_daq, MATRIX_PHASE_4, 'lbf-ft'
    elif phase == 3:
        return read_ni_daq, MATRIX_PHASE_1_2_3, 'lbf-in'
    else:
        return read_powerlab_daq, MATRIX_PHASE_1_2_3, 'lbf-in'


def convert_volt_to_force(df, matrix, phase=None):
    """Convert raw amplifier output (V) to calibrated force/moment."""
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
    
    if phase == 4:
        calibrated_df.attrs['moment_units'] = 'lbf-ft'
    else:
        calibrated_df.attrs['moment_units'] = 'lbf-in'
    
    return calibrated_df


def convert_volt_to_mv(df):
    """Convert raw amplifier output (V) to mV."""
    mv_df = df.copy()
    channel_cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    mv_df[channel_cols] = mv_df[channel_cols] * 1000.0
    return mv_df


def load_force_file(filepath, phase):
    """Read a raw DAQ file and convert to calibrated force data."""
    reader, matrix, moment_units = get_reader_and_matrix(phase)
    raw_df = reader(filepath)
    force_df = convert_volt_to_force(raw_df, matrix, phase=phase)
    return raw_df, force_df