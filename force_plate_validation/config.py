"""Configuration constants, calibration matrices, and z-offset lookup.

Units convention: inches (in) and pounds-force (lbf) throughout.
"""
import numpy as np


M_TO_IN = 39.3701
MM_TO_IN = 1 / 25.4

Z_OFFSET_IN_BY_PLATE = {
    'BP400600': -37.645e-3 * M_TO_IN,
    'OR6-7-8000': None,  # UNCONFIRMED — not yet verified from the OR6-7-8000 manual.
                          # get_z_offset_in() warns and falls back to 0.0 in until this is set.
}

# Plate footprint, manufacturer spec in mm, converted once here. Origin (0, 0) is
# plate center for both plates.
PLATE_DIMS_IN_BY_PLATE = {
    'BP400600':   {'width': 400 * MM_TO_IN, 'height': 600 * MM_TO_IN},
    'OR6-7-8000': {'width': 464 * MM_TO_IN, 'height': 508 * MM_TO_IN},
}

PHASE_TO_PLATE = {
    1: 'BP400600', 2: 'BP400600', 3: 'BP400600',
    4: 'OR6-7-8000',
    5: 'OR6-7-8000',
    6: 'BP400600',
    7: 'OR6-7-8000',
    8: 'BP400600',
    9: 'OR6-7-8000',
    10: 'BP400600',
}


def get_z_offset_in(phase):
    """Look up the z-offset (in inches) for the plate used in a given phase.

    If the offset for that plate has not yet been confirmed from the manual
    (see Z_OFFSET_IN_BY_PLATE), prints a warning and falls back to a 0.0 in
    placeholder so downstream CoP computation still runs — the warning flags
    that an assumption is being made rather than blocking the calculation.
    """
    plate = PHASE_TO_PLATE.get(phase)
    if plate is None:
        raise ValueError(f"Unknown phase {phase}; cannot map to a plate for z-offset lookup.")
    offset = Z_OFFSET_IN_BY_PLATE[plate]
    if offset is None:
        print(
            f"WARNING: z-offset for {plate} (phase {phase}) has not been confirmed from the "
            f"manual yet — using a placeholder of 0.0 in. Confirm and set "
            f"Z_OFFSET_IN_BY_PLATE['{plate}'] before trusting CoP results for this phase."
        )
        return 0.0
    return offset


def get_plate_dims_in(phase):
    """Look up the plate width/height (in inches) for the plate used in a given phase."""
    plate = PHASE_TO_PLATE.get(phase)
    if plate is None:
        raise ValueError(f"Unknown phase {phase}; cannot map to a plate for dimension lookup.")
    return PLATE_DIMS_IN_BY_PLATE[plate]


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

phase_4_gain_scaling = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
MATRIX_PHASE_5 = phase_4_matrix * phase_4_gain_scaling  # Phase 5 warm-up: OR6-7-8000
MATRIX_PHASE_6 = phase_1_2_3_matrix * phase_1_2_3_gain_scaling  # Phase 6 warm-up: BP400600


# Unit conversions
FT_TO_IN = 12.0