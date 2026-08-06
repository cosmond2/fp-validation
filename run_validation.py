from force_plate_validation.compute import generate_spatial_sensitivity_table
from force_plate_validation.runners import run_static_noise_test, run_drift_test, run_corner_test


def main():
    directories = [
        r"ni_6210_daq_txt_files",
        r"powerlab_1630_daq_txt_files",
    ]

    # produce results and optional HTML plots under report_output/
    raw_noise, force_noise = run_static_noise_test(directories, out_dir="report_output")
    drift_results = run_drift_test(directories, out_dir="report_output")
    corner_results = run_corner_test(directories, out_dir="report_output")

    print("\nStatic noise results (raw signal):")
    print(raw_noise.to_string(index=False))

    print("\nStatic noise results (force signal):")
    print(force_noise.to_string(index=False))

    static_unloaded_noise_summary = (
        force_noise.loc[force_noise["Channel"].isin(["Fx", "Fy", "Fz", "Mx", "My", "Mz"])]
        .pivot_table(
            index="Phase",
            columns="Channel",
            values="Peak-to-Peak",
            aggfunc="mean",
        )
        .loc[:, ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]]
    )

    print("\nStatic unloaded Peak-to-Peak summary by phase")
    print(static_unloaded_noise_summary)

    print("\nDrift results:")
    print(drift_results.to_string(index=False))

    print("\nCorner loading results:")
    print(corner_results.to_string(index=False))

    print("\nSpatial sensitivity results:")
    print(generate_spatial_sensitivity_table(corner_results, drift_results).to_string(index=False))

    print("\nDone. Tables saved/plots written to report_output/")


if __name__ == "__main__":
    main()