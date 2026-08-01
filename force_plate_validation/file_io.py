"""File discovery and DAQ reading functions."""
import os
import glob
import pandas as pd


def determine_phase(filename):
    """Determine validation phase from filename."""
    filename = filename.lower()
    if "phase4" in filename:
        return 4
    elif "phase3" in filename:
        return 3
    elif "phase2" in filename:
        return 2
    elif "phase1" in filename:
        return 1
    raise NameError(
        f"File:{filename} does not contain valid phase1/2/3/4 for identification."
    )


def get_files_by_phase(directories, phase=None, keyword=None, extension="*.txt"):
    """Scan directories for files, determine phase, and filter."""
    files_info = []
    for directory in directories:
        search_path = os.path.join(directory, extension)
        for filepath in glob.glob(search_path):
            basename = os.path.basename(filepath)
            basename_lower = basename.lower()
            phase_id = determine_phase(basename)
            
            if phase not in (None, "all") and int(phase) != phase_id:
                continue
            if keyword and keyword.lower() not in basename_lower:
                continue
            
            files_info.append({
                "filepath": filepath,
                "directory": directory,
                "basename": basename,
                "phase": phase_id
            })
    
    files_info.sort(key=lambda x: (x["phase"], x["basename"]))
    
    if not files_info:
        raise FileNotFoundError(
            f"No files found for phase={phase}, keyword={keyword}."
        )
    
    return files_info


def read_ni_daq(file_path):
    """Read NI DAQ CSV file."""
    df = pd.read_csv(file_path)
    df.columns = ['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    return df


def read_powerlab_daq(file_path):
    """Read PowerLab DAQ tab-delimited file."""
    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=6,
        header=None,
        names=['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    )
    return df