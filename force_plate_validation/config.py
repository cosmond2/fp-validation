"""Configuration constants, calibration matrices, and dimensions."""
import numpy as np

# Calibration matrices
phase_1_2_3_gain_scaling = np.array([1, 1, 1, 1, 1, 1])
phase_1_2_3_matrix = np.array([
    [0.6521, 0.0045, -0.0002, -0.0057, -0.0019, 0.0020],
    [-0.0015, 0.6525, -0.0117, -0.0082, -0.0034, -0.0077],
    [0.0010, -0.0052, 2.5676, -0.0012, 0.0012, 0.0006],
    [-0.0168, 0.0309, -0.0594, 12.8865, -0.0469, -0.0248],
    [0.0314, 0.0101, -0.0595, 0.0163, 10.1565, -0.0069],
    [0.0327, 0.1287, -0.0279, 0.0057, 0.0677, 5.4770]
], dtype=float)

phase_4_gain_scaling = np.array([0.25, 0.25, 1, 0.5, 0.5, 0.25])
phase_4_matrix = np.array([
    [2.6952, 0.0567, -0.0513, -0.0221, 0.0360, -0.1051],
    [0.0145, 2.6894, -0.0668, -0.0094, -0.0372, 0.0441],
    [0.0536, 0.0004, 11.4268, -0.0847, 0.0128, 0.0498],
    [0.0037, 0.0081, -0.0010, 3.6987, 0.0023, -0.0428],
    [0.0081, 0.0071, -0.0011, 0.0340, 3.6909, -0.0164],
    [-0.0155, 0.0034, 0.0338, -0.0057, 0.0091, 1.7466]
], dtype=float)

MATRIX_PHASE_1_2_3 = phase_1_2_3_matrix * phase_1_2_3_gain_scaling
MATRIX_PHASE_4 = phase_4_matrix * phase_4_gain_scaling

# Unit conversions
LBF_TO_N = 4.44822
IN_TO_MM = 25.4
FT_TO_IN = 12.0

# Force plate dimensions
BASE_PLATE_DIA = 127  # mm

DIMS_BP400600 = {
    'height': 600,
    'width': 400
}

DIMS_OR67800 = {
    'height': 508,
    'width': 464
}

PHASE_LABELS = {
    1: "Phase 1: BP400600 / MSA6 SN6893 / PowerLab (baseline)",
    2: "Phase 2: BP400600 / MSA6 SN7526 / PowerLab (amplifier isolation)",
    3: "Phase 3: BP400600 / MSA6 SN7526 / NI-6210 (DAQ isolation)",
    4: "Phase 4: OR6-7-8000 / MSA6 SN7526 / PowerLab (force plate isolation)",
}

APPLIED_LOAD_LBS = 50.0
DRIFT_RECORDING_SECONDS = 300.0
DEFAULT_FS = 1000.0
DEFAULT_CUTOFF_HZ = 20.0