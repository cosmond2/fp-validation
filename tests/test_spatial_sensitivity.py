import pandas as pd

from force_plate_validation.compute import generate_spatial_sensitivity_table


def test_generate_spatial_sensitivity_table_includes_fx_fy_fz():
    corner_df = pd.DataFrame([
        {"Phase": 1, "Fx (N)": 1.0, "Fy (N)": 2.0, "Fz (N)": 3.0},
        {"Phase": 1, "Fx (N)": 1.5, "Fy (N)": 2.5, "Fz (N)": 3.5},
        {"Phase": 2, "Fx (N)": 4.0, "Fy (N)": 5.0, "Fz (N)": 6.0},
    ])
    drift_df = pd.DataFrame([
        {"Phase": 1, "Final Fx (N)": 1.2, "Final Fy (N)": 2.2, "Final Fz (N)": 3.2},
        {"Phase": 2, "Final Fx (N)": 4.2, "Final Fy (N)": 5.2, "Final Fz (N)": 6.2},
    ])

    table = generate_spatial_sensitivity_table(corner_df, drift_df)

    assert list(table.columns) == [
        "Phase",
        "Min Fx (N)", "Max Fx (N)", "Range Fx (N)",
        "Min Fy (N)", "Max Fy (N)", "Range Fy (N)",
        "Min Fz (N)", "Max Fz (N)", "Range Fz (N)",
    ]
    assert table.loc[table["Phase"] == 1, "Range Fx (N)"].item() == 0.5
    assert table.loc[table["Phase"] == 1, "Range Fy (N)"].item() == 0.5
    assert table.loc[table["Phase"] == 1, "Range Fz (N)"].item() == 0.5
