# force_plate_validation

Python package supporting the formal validation study of 6-DOF force plates
(study `77EDSTissueFunction`). It converts raw amplifier/DAQ voltage files
into calibrated force/moment data, computes validation metrics, applies
report acceptance criteria, and generates diagnostic plots — all traceable
back to specific report sections.

## Study context

Four hardware configurations are validated, isolating one variable at a time:

| Phase | Force Plate | Amplifier | DAQ | Isolates |
|---|---|---|---|---|
| 1 | BP400600 | MSA6 SN6893 | PowerLab | Baseline |
| 2 | BP400600 | MSA6 SN7526 | PowerLab | Amplifier |
| 3 | BP400600 | MSA6 SN7526 | NI-6210 | DAQ |
| 4 | OR6-7-8000 | MSA6 SN7526 | PowerLab | Force plate |

Three core test types run against each phase: **unloaded noise**, **static
load drift**, and **corner loading accuracy / center of pressure (CoP)**.

Raw data lives under `G:\mkersh\Studies\77EDSTissueFunction`.

## Architecture

The package is organized in three deliberate layers so that computation,
aggregation, and acceptance logic can be audited independently:

```
raw files
   │
   ▼
file_io.py        discover files, read raw DAQ formats
   │
   ▼
calibration.py     convert raw voltage → calibrated force/moment
   │
   ▼
compute.py         PURE functions: one file in → one tidy DataFrame out
   │
   ▼
runners.py         loop over files, call compute functions, stack results
   │
   ▼
criteria.py         apply Table 4 pass/fail thresholds to runner output
   │
   ▼
visualization.py    time series, FFT, CoP scatter, component-comparison plots
```

`quickstart.py` re-exports everything needed for interactive notebook work
in one `from force_plate_validation.quickstart import *`.

### Why this separation matters

- **`compute.py` functions are pure** — they take an already-loaded
  DataFrame and return a tidy result, with no file I/O or looping. This
  makes them independently testable and reusable outside the runner loop
  (e.g. for the single-file EXAMPLES cell in `notebook_analysis.py`).
- **`runners.py` handles iteration only** — file discovery, calling the
  matching `compute.py` function per file, and concatenating results. No
  pass/fail logic lives here.
- **`criteria.py` is the only place Table 4 thresholds appear.** Compute
  and runner outputs are criteria-agnostic; a threshold change never
  requires touching the computation layer.

## Module reference

### `config.py`
Calibration matrices (`MATRIX_PHASE_1_2_3`, `MATRIX_PHASE_4`), unit
conversions (`LBF_TO_N`, `IN_TO_MM`, `FT_TO_IN`), force plate physical
dimensions (`DIMS_BP400600`, `DIMS_OR67800`, `BASE_PLATE_DIA`), phase
labels, and shared test constants (`APPLIED_LOAD_LBS`,
`DRIFT_RECORDING_SECONDS`, `DEFAULT_FS`, `DEFAULT_CUTOFF_HZ`).

### `file_io.py`
- `determine_phase(filename)` — infers phase 1–4 from filename (`raise`s if
  no phase token is found — stricter than the older monolithic script,
  which silently defaulted unmatched files to Phase 1).
- `get_files_by_phase(directories, phase, keyword, extension)` — scans one
  or more directories, filters by phase/keyword, returns a sorted list of
  file-info dicts (`filepath`, `directory`, `basename`, `phase`).
- `read_ni_daq(file_path)` / `read_powerlab_daq(file_path)` — format-specific
  raw readers, both normalized to `Time, Ch1..Ch6` columns.

### `calibration.py`
- `get_reader_and_matrix(phase)` — single source of truth mapping phase →
  (reader function, calibration matrix, moment units).
- `convert_volt_to_force(df, matrix, phase)` — applies excitation/gain
  scaling and the calibration matrix; tags the result with
  `df.attrs['moment_units']` so downstream code doesn't need to
  re-derive units from phase number.
- `convert_volt_to_mv(df)` — raw-channel V → mV, used for noise stats in
  the criteria's native (voltage) domain.
- `load_force_file(filepath, phase)` — convenience wrapper: read + convert
  in one call, returns `(raw_df, force_df)`.

### `signal_processing.py`
- `butterworth_filter(signal, fs, cutoff_hz, order, btype)` — zero-phase
  low-pass filter; accepts ndarray, Series, or DataFrame.
- `compute_fft_noise_signature(signal, fs, high_freq_hz, line_band_hz)` —
  returns line-noise (50 Hz ± band) and high-frequency power share, used to
  distinguish electrical interference from broadband noise.

### `compute.py` (pure per-file metrics)
- `compute_channel_stats(df, columns, units, data_source)` — generic
  mean/std/RMS/peak-to-peak table, the building block for every other
  stats function here.
- `compute_static_noise(raw_df, force_df)` — unloaded noise metrics in
  both raw mV and calibrated force domains.
- `compute_drift(force_df, n_samples)` — initial/final/delta Fz in both
  lbf and N (fixes the earlier lbf/N mislabeling bug — N columns are
  explicit conversions, never a relabeled lbf value).
- `compute_corner_loading(force_df, moment_units, z_offset_in)` — Fx/Fy/Fz
  and CoP X/Y for a single corner file.
- `compute_phase3_center_drift_channel_metrics(...)` — diagnostic
  supporting the Phase 3 investigation: raw + filtered stats normalized to
  the median, to help isolate a single misbehaving channel.
- `summarize_loaded_fft_noise(...)` — per-channel line-noise/high-freq FFT
  share, raw vs. filtered, for the loaded-case FFT overlay diagnostic.
- `compare_center_corner_fx_fy(center_df, corner_df, applied_load_lbs)` —
  checks whether Fx/Fy stay within crosstalk tolerance between center and
  corner loading, for the Phase 4 plate-uniformity investigation.

### `runners.py` (file looping + aggregation)
- `run_static_noise_test(directories, out_dir)` — all `*static*` files →
  stacked raw + force noise tables, optional diagnostic plots.
- `run_drift_test(directories, out_dir, n_samples)` — all `*drift*`/`*center*`
  files → stacked drift table.
- `run_corner_test(directories, z_offset_in, out_dir)` — all corner files
  (matched on `corner`/`_tl`/`_tr`/`_br`/`_bl`) → stacked corner-loading
  table with a `Corner` label column.

Each runner inserts `File` and `Phase` as leading columns on every result
so the aggregated frame stays traceable to source data.

### `criteria.py` (Table 4 pass/fail)
- `apply_noise_criteria(force_noise_df, noise_threshold_N)`
- `apply_drift_criteria(drift_df, applied_load_lbs, abs_threshold_n, pct_threshold)`
- `apply_weight_and_crosstalk_criteria(corner_df, applied_load_lbs, weight_pct, crosstalk_pct)`
- `apply_cop_criteria(corner_df, true_cop_by_phase_corner, tolerance_mm)` —
  looks up ground-truth CoP per `(Phase, Corner)`, since the two force
  plates have different footprints and therefore different true corner
  positions.

> **Known gap:** `apply_noise_criteria` currently thresholds the
> *calibrated force* noise table in N. Table 4's ±2.5 mV noise criterion
> is defined in the raw amplifier voltage domain — there is not yet a
> criteria function that applies it to `raw_noise` (mV) output from
> `compute_static_noise`. Add before relying on noise pass/fail.

### `visualization.py`
- `plot_interactive_timeseries(df, title, y_label, save_path)`
- `plot_fft(df, channel, fs, title, save_path)`
- `plot_cop_scatter(cop_x, cop_y, title, save_path)`
- `plot_component_comparison(metric_df, phase_a, phase_b, value_col, group_col, ...)` —
  grouped bar chart for the pairwise component-isolation comparisons
  (amplifier-only, DAQ-only, plate-only).

All plotting functions render inline (`fig.show()`) by default, or write
standalone HTML if `save_path` is given.

### `report.py`
- `generate_true_cop_by_corner()` — physical ground-truth CoP per phase and
  corner, derived from plate dimensions and the pneumatic base diameter.
  **Currently only Phases 1 and 4 are populated**; Phases 2/3 (which share
  BP400600 geometry with Phase 1) are stubbed with a `# ... repeat` comment
  and need to be filled in before `apply_cop_criteria` is run for those
  phases.
- `run_report()` — placeholder entry point for full report generation;
  not yet implemented.

### `quickstart.py`
Single import point for notebook work. Also prints a categorized function
reference (`_print_welcome()`) on import, so `notebook_analysis.py` gets a
menu of available tools as soon as it loads `quickstart`.

## Typical workflow

```python
from force_plate_validation.quickstart import *

directories = [
    r"G:\...\ni_6210_daq_txt_files",
    r"G:\...\powerlab_1630_daq_txt_files",
]

# Discover files
files = get_files_by_phase(directories, phase='all', keyword='static')

# Single-file inspection
raw_df, force_df = load_force_file(files[0]['filepath'], files[0]['phase'])
plot_interactive_timeseries(raw_df, title="Raw signal")
plot_fft(raw_df, channel='Ch3')

# Full test run across all phases
raw_noise, force_noise = run_static_noise_test(directories)
drift_results = run_drift_test(directories)
corner_results = run_corner_test(directories)

# Apply Table 4 criteria
drift_pass = apply_drift_criteria(drift_results)
corner_pass = apply_weight_and_crosstalk_criteria(corner_results)
cop_pass = apply_cop_criteria(corner_results, generate_true_cop_by_corner())
```

## Report-section traceability

`notebook_analysis.py` cells are organized with `#%%` markers matching
report sections directly, so a reviewer can jump from a report table to the
exact cell that produced it:

| Section | Content |
|---|---|
| 3.4 | File inventory / dataset overview |
| 4.1 / 5.1 | Unloaded noise |
| 4.2 / 5.2 | Static load drift |
| 4.3 / 5.3 | Corner loading & CoP |
| 5.4–5.6 | Component-isolation comparisons (amplifier / DAQ / plate, pairwise) |
| 6.1 | Loaded vs. unloaded FFT overlay |
| 6.2 | Drift time-series warm-up diagnostics |
| 6.3 | Phase 3 raw channel comparison vs. Phase 1 |
| 6.4 | Spatial Fz uniformity map (corner + center) |

## Unit conventions

- `compute.py` and `runners.py` keep force in **lbf** internally (matching
  the calibration matrix output) and add explicit **N** columns via
  `LBF_TO_N` — never relabel, always convert.
- Moments are **lbf-in** for Phases 1–3, **lbf-ft** for Phase 4 (OR6-7-8000
  uses foot-based moment arms); `force_df.attrs['moment_units']` carries
  this per-file so downstream code doesn't hardcode it.
- CoP is reported in **mm** (`IN_TO_MM` conversion from inches).
- Raw amplifier signal is reported in **mV** for noise criteria, since
  Table 4's noise threshold is defined in the voltage domain, not the
  calibrated force domain.

## Known issues / open items

1. `apply_noise_criteria` needs a raw-voltage-domain counterpart (see
   `criteria.py` note above).
2. `compute_static_noise` in `compute.py` hardcodes moment units as
   `lbf-in`, mislabeling Phase 4 (`lbf-ft`). Not currently called by any
   runner or notebook cell, but exported via `quickstart`/`__init__` — fix
   before calling it directly.
3. `generate_true_cop_by_corner()` in `report.py` needs Phase 2/3 entries
   filled in (identical to Phase 1 geometry).
4. `report.py::run_report()` is a stub.
5. `determine_phase` now raises `NameError` on an unrecognized filename
   (stricter than the original script's silent Phase-1 fallback) — expect
   this to surface any mis-named files early rather than silently
   mis-categorizing them.

## Requirements

`numpy`, `pandas`, `scipy` (fft, signal), `plotly` (`graph_objects`,
`express`), `matplotlib`.
