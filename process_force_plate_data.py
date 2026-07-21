#%% Imports

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#%% Calibration Matrix Reference Placeholders

phase_1_2_3_gain_scaling = np.array([1,1,1,1,1,1]) # Data was recorded with channel gains = 4000, matching the calibration sheet. No scaling needed.

# Matrix for Phases 1, 2, and 3 (Original Force Plate configuration, lb,lb-in)
phase_1_2_3_matrix = np.array([
    [0.6521,0.0045,-0.0002,-0.0057,-0.0019,0.0020],
    [-0.0015,0.6525,-0.0117,-0.0082,-0.0034,-0.0077],
    [0.0010,-0.0052,2.5676,-0.0012,0.0012,0.0006],
    [-0.0168,0.0309,-0.0594,12.8865,-0.0469,-0.0248],
    [0.0314,0.0101,-0.0595,0.0163,10.1565,-0.0069],
    [0.0327,0.1287,-0.0279,0.0057,0.0677,5.4770]
], dtype=float)
MATRIX_PHASE_1_2_3 = phase_1_2_3_matrix * phase_1_2_3_gain_scaling


phase_4_gain_scaling = np.array([0.25,0.25,1,0.5,0.5,0.25]) # Data was recorded with channel gains = 4000, which does NOT match the calibration sheet (FxyzMxyz = 1000,1000,4000,2000,2000,1000). Scaling is needed.

# Matrix for Phase 4 (New OR6-7-800 Force Plate configuration, lb,lb-ft)
phase_4_matrix = np.array([
    [2.6952,0.0567,-0.0513,-0.0221,0.0360,-0.1051],
    [0.0145,2.6894,-0.0668,-0.0094,-0.0372,0.0441],
    [0.0536,0.0004,11.4268,-0.0847,0.0128,0.0498],
    [0.0037,0.0081,-0.0010,3.6987,0.0023,-0.0428],
    [0.0081,0.0071,-0.0011,0.0340,3.6909,-0.0164],
    [-0.0155,0.0034,0.0338,-0.0057,0.0091,1.7466]
], dtype=float)
MATRIX_PHASE_4 = phase_4_matrix * phase_4_gain_scaling

#%% Define helper functions for reading in txt files

def read_ni_daq(file_path):
    """
    Reads a raw text file from the NI DAQ system (Phase 3).
    Format: Comma-separated with a single-line header.
    """
    df = pd.read_csv(file_path)
    df.columns = ['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    return df

def read_powerlab_daq(file_path):
    """
    Reads a raw text file from the PowerLab/LabChart DAQ system (Phases 1, 2, 4).
    Format: Tab-separated with 6 metadata header lines to skip.
    """
    df = pd.read_csv(
        file_path, 
        sep='\t', 
        skiprows=6, 
        header=None,
        names=['Time', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    )
    return df

def convert_volt_to_force(df, matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"Calibration matrix must have shape (6, 6); got {matrix.shape}")

    voltage_data = df[['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']].values
    
    # ---------------------------------------------------------
    # PROPER HARDWARE SCALING
    # Replace these with the actual values from your amplifier/DAQ
    # ---------------------------------------------------------
    excitation_voltage = 10.0  # Usually 5V or 10V
    amplifier_gain = 4000.0
    
    # If your DAQ recorded in Volts, but the matrix expects microVolts/Volt:
    # you often need a multiplier of 1,000,000 to bridge the gap.
    electrical_scalar = 1_000_000 / (excitation_voltage * amplifier_gain)
    
    # Apply the scalar to the raw voltage before the matrix dot product
    scaled_voltage = voltage_data * electrical_scalar
    calibrated_data = np.dot(scaled_voltage, matrix.T)
    # ---------------------------------------------------------

    calibrated_df = pd.DataFrame(
        calibrated_data, 
        columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    )
    calibrated_df['Time'] = df['Time'].values
    return calibrated_df

#%% Plotting Helper Function

def plot_interactive_timeseries(df, title="Time Series Data", y_label="Amplitude", save_path=None):
    """
    Generates an interactive Plotly HTML graph.
    Saves to disk if save_path is provided, otherwise opens in the default browser.
    """
    if 'Time' not in df.columns:
        raise ValueError("DataFrame must contain a 'Time' column to plot.")
    
    signal_cols = [col for col in df.columns if col != 'Time']
    
    fig = go.Figure()
    
    for col in signal_cols:
        fig.add_trace(go.Scatter(
            x=df['Time'], 
            y=df[col], 
            mode='lines', 
            name=col
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Channels",
        template="plotly_white"
    )
    
    if save_path:
        # Saves the interactive plot as a standalone HTML file
        fig.write_html(save_path)
    else:
        # Opens in browser
        fig.show()

#%% 1.1, 2.1, 3.1, 4.1 - Static noise evaluation

def evaluate_static_noise(*systems, out_dir=None):
    """
    Accepts one or more (directory_path, matrix_123, matrix_4) tuples.
    Loops through files containing 'static'
    Selects the calibration matrix and DAQ reader based on the testing Phase.
    """
    print("==================================================")
    print("      STATIC NOISE EVALUATION RESULTS SUMMARY     ")
    print("==================================================")
    
    for directory_path, matrix_123, matrix_4 in systems:
        search_path = os.path.join(directory_path, "*.txt")
        files = glob.glob(search_path)
        
        for file in files:
            filename_lower = os.path.basename(file).lower()
            
            if 'static' in filename_lower:
                print(f"\nProcessing File: {os.path.basename(file)}")
                
                # --- PHASE MATRIX & DAQ SELECTION LOGIC ---
                if 'phase4' in filename_lower:
                    active_matrix = matrix_4
                    raw_df = read_powerlab_daq(file) # Phase 4 uses LabChart
                else:
                    active_matrix = matrix_123
                    if 'phase3' in filename_lower:
                        raw_df = read_ni_daq(file)       # Phase 3 uses NI DAQ
                    else:
                        raw_df = read_powerlab_daq(file) # Phase 1 & 2 use LabChart
                # ------------------------------------------
                    
                df = convert_volt_to_force(raw_df, active_matrix)


                if out_dir:
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    save_path = os.path.join(out_dir, f"{base_name}_static.html")
                    plot_interactive_timeseries(
                        df, 
                        title=f"Static Noise: {base_name}", 
                        y_label="Force (lbf)", 
                        save_path=save_path
                    )

                print(f"\nProcessing File: {os.path.basename(file)}")
                
                for col in ['Fx', 'Fy', 'Fz']:
                    signal = df[col].values
                    
                    ptp_val = np.max(signal) - np.min(signal) 
                    rms_val = np.sqrt(np.mean((signal - np.mean(signal)) ** 2))
                    
                    ptp_pass = ptp_val <= 2*1.12404  # Window width of +/- 5 N (in lbf)
                    rms_pass = rms_val < 0.224809    # RMS noise < 1 N (in lbf)
                    
                    status = "PASS" if (ptp_pass and rms_pass) else "FAIL"
                    print(f"  Channel {col} -> P2P: {ptp_val:.3f} N | RMS: {rms_val:.3f} N | Status: {status}")

#%% 1.2, 2.2, 3.2, 4.2 - Center drift evaluation

def evaluate_center_drift(*systems, applied_load_lbs=50.0, out_dir=None):
    """
    Evaluates baseline drift over 5-minute centered loading trials.
    Accepts one or more (directory_path, matrix_123, matrix_4) tuples.
    """
    true_weight_n = applied_load_lbs * 4.44822
    drift_threshold = min(1.0, 0.005 * true_weight_n) # Less of 1 N or 0.5% of total load
    
    print("\n==================================================")
    print("      CENTER DRIFT EVALUATION RESULTS SUMMARY     ")
    print("==================================================")
    
    for directory_path, matrix_123, matrix_4 in systems:
        search_path = os.path.join(directory_path, "*.txt")
        files = glob.glob(search_path)
        
        for file in files:
            filename_lower = os.path.basename(file).lower()
            
            if 'drift' in filename_lower or 'center' in filename_lower:
                
                # --- PHASE MATRIX & DAQ SELECTION LOGIC ---
                if 'phase4' in filename_lower:
                    active_matrix = matrix_4
                    raw_df = read_powerlab_daq(file)
                else:
                    active_matrix = matrix_123
                    if 'phase3' in filename_lower:
                        raw_df = read_ni_daq(file)
                    else:
                        raw_df = read_powerlab_daq(file)
                # ------------------------------------------
                    
                df = convert_volt_to_force(raw_df, active_matrix)

                if out_dir:
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    save_path = os.path.join(out_dir, f"{base_name}_drift.html")
                    plot_interactive_timeseries(
                        df, 
                        title=f"Center Drift: {base_name}", 
                        y_label="Force (lbf) / Moment", 
                        save_path=save_path
                    )

                fz_signal = df['Fz'].values
                
                # ----------------------------------------------------------------
                initial_fz = np.mean(fz_signal[:1000]) 
                final_fz = np.mean(fz_signal[-1000:])  
                total_drift = abs(final_fz - initial_fz)
                # ----------------------------------------------------------------
                
                status = "PASS" if total_drift < drift_threshold else "FAIL"
                print(f"File: {os.path.basename(file)}")
                print(f"  Fz Drift: {total_drift:.3f} N (Threshold: {drift_threshold:.3f} N) | Status: {status}")

#%% 1.3, 2.3, 3.3, 4.3 - Corner loading accuracy and crosstalk

def evaluate_corners_and_crosstalk(*systems, applied_load_lbs=50.0, out_dir=None):
    """
    Evaluates multi-axis crosstalk (<0.2%), scaling accuracy (<0.5%), 
    and Center of Pressure (COP) during eccentric corner loads.
    Accepts one or more (directory_path, matrix_123, matrix_4) tuples.
    """
    true_fz_lbs = applied_load_lbs
    true_fz_n = applied_load_lbs * 4.44822
    
    fz_lower_limit = true_fz_n * 0.995
    fz_upper_limit = true_fz_n * 1.005
    max_allowed_crosstalk = 0.002 * true_fz_n
    
    # Distance from the sensor's mechanical origin to the top surface.
    # Set this to 0 if your calibration matrix already projects the origin to the surface.
    # (AMTI plates often have a vertical offset, e.g., ~1.5 inches).
    z_offset_inches = 0.0  
    
    print("\n==================================================")
    print("    CORNER LOADING, CROSSTALK & COP RESULTS       ")
    print("==================================================")
    
    for directory_path, matrix_123, matrix_4 in systems:
        search_path = os.path.join(directory_path, "*.txt")
        files = glob.glob(search_path)
        
        for file in files:
            filename_lower = os.path.basename(file).lower()
            
            if any(corner in filename_lower for corner in ['corner', '_tl', '_tr', '_br', '_bl']) and 'static' not in filename_lower and 'drift' not in filename_lower:
                
                # --- PHASE MATRIX, DAQ, & UNIT SELECTION LOGIC ---
                if 'phase4' in filename_lower:
                    active_matrix = matrix_4
                    raw_df = read_powerlab_daq(file)
                    moment_units = 'lb-ft'
                else:
                    active_matrix = matrix_123
                    moment_units = 'lb-in'
                    if 'phase3' in filename_lower:
                        raw_df = read_ni_daq(file)
                    else:
                        raw_df = read_powerlab_daq(file)
                # ------------------------------------------
                    
                df = convert_volt_to_force(raw_df, active_matrix)

                if out_dir:
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    save_path = os.path.join(out_dir, f"{base_name}_corner_crosstalk.html")
                    plot_interactive_timeseries(
                        df, 
                        title=f"Corner Loading: {base_name}", 
                        y_label=f"Force (lbf) / Moment ({moment_units})", 
                        save_path=save_path
                    )

                # Extract steady-state mean values (in lbs and lb-in / lb-ft)
                measured_fx_lbs = np.mean(df['Fx'].values)
                measured_fy_lbs = np.mean(df['Fy'].values)
                measured_fz_lbs = np.mean(df['Fz'].values)
                measured_mx = np.mean(df['Mx'].values)
                measured_my = np.mean(df['My'].values)
                
                # Convert Forces to Newtons for the pass/fail checks
                measured_fx_n = measured_fx_lbs * 4.44822
                measured_fy_n = measured_fy_lbs * 4.44822
                measured_fz_n = measured_fz_lbs * 4.44822
                
                # --- CENTER OF PRESSURE (COP) MATH ---
                # Adjust moments if they are in lb-ft to match the lb-in system for math
                if moment_units == 'lb-ft':
                    measured_mx *= 12.0
                    measured_my *= 12.0 
                
                # COP Equations: COPx = (-My + Fx * Z_off) / Fz
                #                COPy = (Mx + Fy * Z_off) / Fz
                cop_x_inches = (-measured_my + (measured_fx_lbs * z_offset_inches)) / measured_fz_lbs
                cop_y_inches = (measured_mx + (measured_fy_lbs * z_offset_inches)) / measured_fz_lbs
                
                # Convert COP to millimeters (1 inch = 25.4 mm)
                cop_x_mm = cop_x_inches * 25.4
                cop_y_mm = cop_y_inches * 25.4
                
                # --- PASS / FAIL EVALUATION ---
                fz_pass = fz_lower_limit <= abs(measured_fz_n) <= fz_upper_limit
                crosstalk_pass = (abs(measured_fx_n) <= max_allowed_crosstalk) and (abs(measured_fy_n) <= max_allowed_crosstalk)
                
                status = "PASS" if (fz_pass and crosstalk_pass) else "FAIL"
                
                print(f"\nFile: {os.path.basename(file)} -> Status: {status}")
                print(f"  Fz (Vertical): {abs(measured_fz_n):.2f} N (Target: {true_fz_n:.2f} N)")
                print(f"  Fx (Cross): {measured_fx_n:.3f} N | Fy (Cross): {measured_fy_n:.3f} N (Max Allowed: {max_allowed_crosstalk:.3f} N)")
                print(f"  COP Location: X = {cop_x_mm:.2f} mm | Y = {cop_y_mm:.2f} mm")
                print(f"  -> Action: Verify X/Y matches physical {filename_lower.split('_')[-1].split('.')[0].upper()} corner within 3 mm.")


#%% Execution Block 
if __name__ == "__main__":
    # Define directory paths where the experimental text files are saved
    data_dir_1 = r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\ni_6210_daq_txt_files"
    data_dir_2 = r"G:\mkersh\Studies\77EDSTissueFunction\Raw Data\carle_force_plate_validation_dataset\powerlab_1630_daq_txt_files"
    
    # Create a directory for the diagnostic plots
    plot_out_dir = r"G:\mkersh\Studies\77EDSTissueFunction\Processed Data\carle_force_plate_validation_dataset\diagnostic_plots"
    os.makedirs(plot_out_dir, exist_ok=True)

    systems = [
        (data_dir_1, MATRIX_PHASE_1_2_3, MATRIX_PHASE_4),
        (data_dir_2, MATRIX_PHASE_1_2_3, MATRIX_PHASE_4),
    ]
    
    # Pass the output directory into your evaluation functions
    evaluate_static_noise(*systems, out_dir=plot_out_dir)
    #evaluate_center_drift(*systems, out_dir=plot_out_dir)
    #evaluate_corners_and_crosstalk(*systems, out_dir=plot_out_dir)
# %%
