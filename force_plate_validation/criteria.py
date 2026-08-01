"""Pass/fail acceptance criteria functions."""
import numpy as np
from .config import LBF_TO_N


def apply_noise_criteria(force_noise_df, noise_threshold_N=0.5):
    """Apply baseline noise acceptance criteria."""
    out = force_noise_df.copy()
    out["Peak-to-Peak (N)"] = (
        out["Peak-to-Peak"] * LBF_TO_N 
        if 'lbf' in out['Units'].iloc[0] 
        else out["Peak-to-Peak"]
    )
    out["Pass"] = out["Peak-to-Peak (N)"] <= (2 * noise_threshold_N)
    return out


def apply_drift_criteria(drift_df, applied_load_lbs=50.0, abs_threshold_n=1.0, pct_threshold=0.005):
    """Apply drift acceptance criteria."""
    true_load_n = applied_load_lbs * LBF_TO_N
    threshold_n = min(abs_threshold_n, pct_threshold * true_load_n)
    out = drift_df.copy()
    out["Threshold (N)"] = threshold_n
    out["Pass"] = out["Drift (N)"].abs() <= threshold_n
    return out


def apply_weight_and_crosstalk_criteria(corner_df, applied_load_lbs=50.0,
                                        weight_pct=0.005, crosstalk_pct=0.002):
    """Apply weight measurement and crosstalk acceptance criteria."""
    true_load_n = applied_load_lbs * LBF_TO_N
    weight_tol_n = weight_pct * true_load_n
    crosstalk_tol_n = crosstalk_pct * true_load_n
    
    out = corner_df.copy()
    out["True Fz (N)"] = true_load_n
    out["Weight Tolerance (N)"] = weight_tol_n
    out["Weight Pass"] = (out["Fz (N)"].abs() - true_load_n).abs() <= weight_tol_n
    out["Crosstalk Tolerance (N)"] = crosstalk_tol_n
    out["Fx Pass"] = out["Fx (N)"].abs() <= crosstalk_tol_n
    out["Fy Pass"] = out["Fy (N)"].abs() <= crosstalk_tol_n
    out["Pass"] = out["Weight Pass"] & out["Fx Pass"] & out["Fy Pass"]
    return out


def apply_cop_criteria(corner_df, true_cop_by_phase_corner, tolerance_mm=3.0):
    """Apply center of pressure acceptance criteria."""
    def _lookup(row, axis_idx):
        corner_map = true_cop_by_phase_corner.get(row["Phase"], {})
        return corner_map.get(row["Corner"], (np.nan, np.nan))[axis_idx]
    
    out = corner_df.copy()
    out["True CoP X (mm)"] = out.apply(_lookup, axis=1, axis_idx=0)
    out["True CoP Y (mm)"] = out.apply(_lookup, axis=1, axis_idx=1)
    err = np.sqrt(
        (out["CoP X (mm)"] - out["True CoP X (mm)"]) ** 2
        + (out["CoP Y (mm)"] - out["True CoP Y (mm)"]) ** 2
    )
    out["CoP Error (mm)"] = err
    out["Pass"] = err <= tolerance_mm
    return out