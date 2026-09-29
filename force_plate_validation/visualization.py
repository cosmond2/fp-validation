"""Plotting and visualization functions."""
import re
from pathlib import Path

import plotly.graph_objects as go
import numpy as np
from scipy.fft import fft, fftfreq


def _write_html_safely(fig, save_path):
    """Write a Plotly figure to HTML, creating the directory and sanitizing the filename."""
    if not save_path:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid Windows-invalid characters in filenames.
    safe_name = re.sub(r'[<>:"/\\|?*]+', '_', path.name)
    if safe_name != path.name:
        path = path.with_name(safe_name)

    try:
        fig.write_html(str(path))
    except OSError:
        html_str = fig.to_html(include_plotlyjs="cdn")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html_str)


def plot_interactive_timeseries(df, title="Time Series Data", y_label="Amplitude", save_path=None):
    """Plot interactive time series."""
    if 'Time' not in df.columns:
        raise ValueError("DataFrame must contain a 'Time' column.")
    
    signal_cols = [col for col in df.columns if col != 'Time']
    fig = go.Figure()
    for col in signal_cols:
        fig.add_trace(go.Scatter(x=df['Time'], y=df[col], mode='lines', name=col))
    
    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title=y_label,
        hovermode="x unified", legend_title="Channels", template="plotly_white"
    )
    
    if save_path:
        _write_html_safely(fig, save_path)
    else:
        fig.show()


def plot_fft(df, channel='Ch1', fs=1000.0, title=None, save_path=None):
    """Plot FFT power spectrum."""
    signal = df[channel].values - np.mean(df[channel].values)
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    power = np.abs(yf[:n//2])**2 / n
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=power, mode='lines', name='Power Spectrum'))
    fig.update_layout(
        title=title or f"FFT of {channel}",
        xaxis_title="Frequency (Hz)", yaxis_title="Power", template="plotly_white"
    )
    
    if save_path:
        _write_html_safely(fig, save_path)
    else:
        fig.show()


def plot_cop_scatter(cop_x, cop_y, title="Center of Pressure", save_path=None):
    """Plot center of pressure scatter."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cop_x, y=cop_y, mode='markers',
        marker=dict(size=3, opacity=0.4)
    ))
    fig.update_layout(
        title=title, xaxis_title="CoP X (mm)", yaxis_title="CoP Y (mm)",
        template="plotly_white", yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    if save_path:
        _write_html_safely(fig, save_path)
    else:
        fig.show()


def plot_component_comparison(metric_df, phase_a, phase_b, value_col, group_col,
                              label_a, label_b, title, y_label, save_path=None):
    """Grouped bar chart comparing two phases."""
    a = metric_df[metric_df["Phase"] == phase_a].set_index(group_col)[value_col]
    b = metric_df[metric_df["Phase"] == phase_b].set_index(group_col)[value_col]
    categories = sorted(set(a.index) | set(b.index))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name=label_a, x=categories, 
                        y=[a.get(c, np.nan) for c in categories]))
    fig.add_trace(go.Bar(name=label_b, x=categories, 
                        y=[b.get(c, np.nan) for c in categories]))
    fig.update_layout(
        barmode="group", title=title, yaxis_title=y_label, template="plotly_white"
    )
    
    if save_path:
        _write_html_safely(fig, save_path)
    else:
        fig.show()
    
    return fig