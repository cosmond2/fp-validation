from force_plate_validation.file_io import get_files_by_phase
from force_plate_validation.calibration import load_force_file
from force_plate_validation.runners import run_static_noise_test, run_drift_test, run_corner_test

directories = [
    r"ni_6210_daq_txt_files",
    r"powerlab_1630_daq_txt_files",
]

# produce results and optional HTML plots under report_output/
raw_noise, force_noise = run_static_noise_test(directories, out_dir="report_output")
drift_results = run_drift_test(directories, out_dir="report_output")
corner_results = run_corner_test(directories, out_dir="report_output")
print("Done. Tables saved/plots written to report_output/")