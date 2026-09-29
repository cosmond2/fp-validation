"""Test runner functions that iterate over files and call compute functions."""
import os
import re
from pathlib import Path

import numpy as np
from .file_io import get_files_by_phase
from .calibration import load_force_file, get_reader_and_matrix
from .compute import (
    compute_static_noise, compute_drift, compute_corner_loading
)
from .visualization import plot_interactive_timeseries, plot_fft


def _iter_test_files(directories, keyword, exclude_keywords=None):
    """Helper to get test files with optional exclusions."""
    files = get_files_by_phase(directories, phase='all', keyword=keyword)
    if exclude_keywords:
        files = [
            f for f in files
            if not any(k in f['basename'].lower() for k in exclude_keywords)
        ]
    return files


def _build_output_path(out_dir, base_name, suffix):
    """Create a safe HTML output path for a generated plot."""
    safe_base = re.sub(r'[<>:"/\\|?*]+', '_', base_name)
    output_path = Path(out_dir) / f"{safe_base}_{suffix}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def run_static_noise_test(directories, out_dir=None):
    """Run unloaded noise test across all files."""
    files = _iter_test_files(directories, keyword="static")
    
    raw_results, force_results = [], []
    for f in files:
        raw_df, force_df = load_force_file(f["filepath"], f["phase"])
        raw_noise, force_noise = compute_static_noise(raw_df, force_df)
        
        for df_ in (raw_noise, force_noise):
            df_.insert(0, "Phase", f["phase"])
            df_.insert(0, "File", f["basename"])
        
        raw_results.append(raw_noise)
        force_results.append(force_noise)
        
        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                raw_df, title=f"Raw signal: {base}", y_label="Voltage (V)",
                save_path=_build_output_path(out_dir, base, "raw_timeseries")
            )
            plot_fft(
                raw_df, channel='Ch1', title=f"FFT (Ch1): {base}",
                save_path=_build_output_path(out_dir, base, "fft")
            )
    
    return (
        pd.concat(raw_results, ignore_index=True),
        pd.concat(force_results, ignore_index=True)
    )


def run_drift_test(directories, out_dir=None, n_samples=1000):
    """Run drift test across all files."""
    files = _iter_test_files(directories, keyword=None)
    files = [f for f in files 
             if 'drift' in f['basename'].lower() or 'center' in f['basename'].lower()]
    
    results = []
    for f in files:
        _, force_df = load_force_file(f["filepath"], f["phase"])
        row = compute_drift(force_df, n_samples=n_samples)
        row.insert(0, "Phase", f["phase"])
        row.insert(0, "File", f["basename"])
        results.append(row)
        
        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                force_df, title=f"Drift: {base}", y_label="Force (lbf) / Moment",
                save_path=_build_output_path(out_dir, base, "drift")
            )
    
    return pd.concat(results, ignore_index=True)


def run_corner_test(directories, z_offset_in=0.0, out_dir=None):
    """Run corner loading test across all files."""
    files = _iter_test_files(
        directories, keyword=None,
        exclude_keywords=['static', 'drift']
    )
    files = [
        f for f in files
        if any(k in f['basename'].lower() 
              for k in ['corner', '_tl', '_tr', '_br', '_bl'])
    ]
    
    corner_labels = {'_tl': 'TL', '_tr': 'TR', '_br': 'BR', '_bl': 'BL'}
    
    results = []
    for f in files:
        _, force_df = load_force_file(f["filepath"], f["phase"])
        _, matrix, moment_units = get_reader_and_matrix(f["phase"])
        
        row = compute_corner_loading(force_df, moment_units, z_offset_in=z_offset_in)
        
        corner = next(
            (v for k, v in corner_labels.items() 
             if k in f['basename'].lower()), 
            "Unknown"
        )
        row.insert(0, "Corner", corner)
        row.insert(0, "Phase", f["phase"])
        row.insert(0, "File", f["basename"])
        results.append(row)
        
        if out_dir:
            base = os.path.splitext(f["basename"])[0]
            plot_interactive_timeseries(
                force_df, title=f"Corner: {base}", y_label="Force (lbf) / Moment",
                save_path=_build_output_path(out_dir, base, "corner")
            )
    
    return pd.concat(results, ignore_index=True)


# Import at bottom to avoid circular imports
import pandas as pd