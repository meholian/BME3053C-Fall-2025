"""Streamlit lab for exploring Fourier transforms with educational annotations."""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import get_window
import streamlit as st
from streamlit_plotly_events import plotly_events


MAX_COMPONENTS = 7
BASE_DURATION_RANGE = (0.5, 2.0)
BASE_FS_RANGE = (200, 2000)

PRESET_LIBRARY: Dict[str, Dict[str, object]] = {
    "Square Wave": {
        "description": (
            "Approximates a 20 Hz square wave via odd harmonics. Square waves contain \n"
            "strong odd-multiple peaks, so their spectrum decays slowly with frequency."
        ),
        "components": [
            {"amplitude": 1.0, "frequency": 20.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 1 / 3, "frequency": 60.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 1 / 5, "frequency": 100.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 1 / 7, "frequency": 140.0, "phase": 0.0, "waveform": "sine", "active": True},
        ],
    },
    "Triangle Wave": {
        "description": (
            "Triangle waves also use odd harmonics, yet amplitudes fall with the square of the harmonic index,\n"
            "which makes the spectrum roll off more quickly than a square wave."
        ),
        "components": [
            {"amplitude": 1.0, "frequency": 15.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 1 / 9, "frequency": 45.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 1 / 25, "frequency": 75.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 1 / 49, "frequency": 105.0, "phase": 0.0, "waveform": "cosine", "active": True},
        ],
    },
    "Sawtooth Wave": {
        "description": (
            "Sawtooth waves include both even and odd harmonics with amplitudes that fall as 1/f. \n"
            "That slow decay produces a denser spectral comb."
        ),
        "components": [
            {"amplitude": 1.0, "frequency": 10.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.5, "frequency": 20.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.33, "frequency": 30.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.25, "frequency": 40.0, "phase": 0.0, "waveform": "sine", "active": True},
        ],
    },
    "Gaussian Pulse": {
        "description": (
            "A Gaussian pulse is localized in time and therefore spreads broadly in frequency.\n"
            "We approximate it with multiple phased sinusoids to mimic that localized bump."
        ),
        "components": [
            {"amplitude": 0.8, "frequency": 5.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 0.8, "frequency": 15.0, "phase": math.pi / 4, "waveform": "cosine", "active": True},
            {"amplitude": 0.6, "frequency": 25.0, "phase": math.pi / 2, "waveform": "cosine", "active": True},
            {"amplitude": 0.5, "frequency": 35.0, "phase": math.pi, "waveform": "cosine", "active": True},
        ],
    },
    "Step": {
        "description": (
            "Steps contain a DC offset plus many low-frequency components. Their spectrum peaks at zero\n"
            "frequency and then drops roughly as 1/f."
        ),
        "components": [
            {"amplitude": 1.0, "frequency": 0.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 0.6, "frequency": 5.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.4, "frequency": 15.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.25, "frequency": 25.0, "phase": 0.0, "waveform": "sine", "active": True},
        ],
    },
    "Impulse / Spike": {
        "description": (
            "A spike is extremely short in time, so its Fourier transform is almost flat.\n"
            "We imitate that with many high-frequency sinusoids of equal weight."
        ),
        "components": [
            {"amplitude": 0.4, "frequency": 60.0, "phase": 0.0, "waveform": "cosine", "active": True},
            {"amplitude": 0.4, "frequency": 120.0, "phase": math.pi / 4, "waveform": "cosine", "active": True},
            {"amplitude": 0.4, "frequency": 180.0, "phase": math.pi / 2, "waveform": "cosine", "active": True},
            {"amplitude": 0.4, "frequency": 240.0, "phase": 3 * math.pi / 4, "waveform": "cosine", "active": True},
        ],
    },
    "Chirp": {
        "description": (
            "Chirps sweep frequency with time. We approximate one by mixing two components whose phases\n"
            "increase quadratically to mimic the sweep."
        ),
        "components": [
            {"amplitude": 0.8, "frequency": 5.0, "phase": 0.0, "waveform": "sine", "active": True},
            {"amplitude": 0.8, "frequency": 40.0, "phase": math.pi / 2, "waveform": "sine", "active": True},
            {"amplitude": 0.6, "frequency": 80.0, "phase": math.pi, "waveform": "sine", "active": True},
        ],
    },
}

WINDOW_SUMMARY = {
    "rectangular": {
        "text": "Rectangular windows keep every sample equally but cause strong leakage.",
        "main_lobe": "~2 bins",
        "sidelobe": "~-13 dB",
    },
    "hann": {
        "text": "Hann is a go-to for general spectral analysis with moderate leakage control.",
        "main_lobe": "~4 bins",
        "sidelobe": "~-31 dB",
    },
    "hamming": {
        "text": "Hamming offers slightly narrower main lobes but higher sidelobes than Hann.",
        "main_lobe": "~4 bins",
        "sidelobe": "~-43 dB",
    },
    "blackman": {
        "text": "Blackman widens the main lobe yet pushes sidelobes far down, ideal for weak tones.",
        "main_lobe": "~6 bins",
        "sidelobe": "~-58 dB",
    },
    "kaiser": {
        "text": "Kaiser lets you dial trade-offs via beta; higher beta lowers sidelobes but widens the lobe.",
        "main_lobe": "variable",
        "sidelobe": "variable",
    },
}


def init_state() -> None:
    if "components" not in st.session_state:
        st.session_state.components = [
            {
                "amplitude": 1.0,
                "frequency": 10.0,
                "phase": 0.0,
                "waveform": "sine",
                "active": True,
            },
            {
                "amplitude": 0.5,
                "frequency": 30.0,
                "phase": math.pi / 4,
                "waveform": "cosine",
                "active": True,
            },
        ]
    if "base_duration" not in st.session_state:
        st.session_state.base_duration = 1.0
    if "base_fs" not in st.session_state:
        st.session_state.base_fs = 500
    if "highlight_freq" not in st.session_state:
        st.session_state.highlight_freq = None
    if "highlight_time" not in st.session_state:
        st.session_state.highlight_time = None
    if "selected_band" not in st.session_state:
        st.session_state.selected_band = (10.0, 20.0)
    if "misalign" not in st.session_state:
        st.session_state.misalign = False


def generate_time_vector(duration: float, fs: float) -> np.ndarray:
    n = max(128, int(duration * fs))
    return np.linspace(0.0, duration, n, endpoint=False)


def compute_window(name: str, length: int, beta: float = 14.0) -> np.ndarray:
    window_name = name.lower()
    if window_name == "rectangular":
        return np.ones(length)
    if window_name == "kaiser":
        return get_window((window_name, beta), length)
    return get_window(window_name, length)


def build_signal(
    components: List[Dict[str, float]],
    t: np.ndarray,
    misalign: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    base_signal = np.zeros_like(t)
    component_signals = []
    bin_offset = 0.0
    if misalign:
        bin_width = 1.0 / (t[1] - t[0]) / len(t)
        bin_offset = 0.37 * bin_width
    for idx, comp in enumerate(components):
        if not comp.get("active", True):
            component_signals.append(np.zeros_like(t))
            continue
        amp = comp["amplitude"]
        freq = comp["frequency"] + (bin_offset if misalign and idx == 0 else 0.0)
        phase = comp["phase"]
        if comp["waveform"] == "cosine":
            sig = amp * np.cos(2 * np.pi * freq * t + phase)
        else:
            sig = amp * np.sin(2 * np.pi * freq * t + phase)
        component_signals.append(sig)
        base_signal += sig
    return base_signal, component_signals


def render_component_expression(components: List[Dict[str, float]]) -> str:
    pieces = []
    for idx, comp in enumerate(components, start=1):
        if not comp.get("active", True):
            continue
        amp = comp["amplitude"]
        freq = comp["frequency"]
        phase = comp["phase"]
        wf = "\\sin" if comp["waveform"] == "sine" else "\\cos"
        pieces.append(
            f"{amp:.2f} \\cdot {wf}(2\\pi\\cdot{freq:.1f}t + {phase:.2f})"
        )
    if not pieces:
        return "x(t) = 0"
    return "x(t) = " + " + ".join(pieces)


def fft_spectrum(
    x: np.ndarray,
    fs: float,
    window: Optional[np.ndarray] = None,
    zero_pad_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    data = x.copy()
    if window is not None:
        data *= window
    n = len(data)
    pad = max(0, int((zero_pad_factor - 1) * n))
    if pad > 0:
        data = np.pad(data, (0, pad))
        n = len(data)
    spectrum = fft(data)
    freqs = fftfreq(n, d=1 / fs)
    return freqs, spectrum


def reconstruct_band(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    band: Tuple[float, float],
) -> np.ndarray:
    f_low, f_high = band
    mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    filtered = np.zeros_like(spectrum, dtype=complex)
    filtered[mask] = spectrum[mask]
    return ifft(filtered).real


def explanatory_caption(text: str, container=None) -> None:
    target = container if container is not None else st
    target.caption(text)


def component_block(
    index: int,
    component: Dict[str, float],
    container,
) -> Optional[int]:
    with container.expander(f"Component {index + 1}", expanded=True):
        component["active"] = container.checkbox(
            "Include this sinusoid",
            value=component.get("active", True),
            key=f"comp_active_{index}",
        )
        explanatory_caption(
            "Toggle components on/off to hear what happens when a single tone disappears.",
            container=container,
        )
        component["amplitude"] = container.slider(
            "Amplitude",
            min_value=0.0,
            max_value=3.0,
            value=float(component["amplitude"]),
            key=f"amp_{index}",
        )
        explanatory_caption(
            "Amplitude controls the height/energy of this wave. Larger values create taller peaks and stronger spectral lines.",
            container=container,
        )
        component["frequency"] = container.slider(
            "Frequency (Hz)",
            min_value=0.0,
            max_value=250.0,
            value=float(component["frequency"]),
            key=f"freq_{index}",
        )
        explanatory_caption(
            "Frequency sets how many cycles fit into a second and thus where this component lands in the spectrum.",
            container=container,
        )
        component["phase"] = container.slider(
            "Phase (rad)",
            min_value=0.0,
            max_value=float(2 * math.pi),
            value=float(component["phase"]),
            key=f"phase_{index}",
        )
        explanatory_caption(
            "Phase shifts the waveform left/right in time. It determines how components align or cancel when added.",
            container=container,
        )
        component["waveform"] = container.radio(
            "Basis",
            options=["sine", "cosine"],
            index=0 if component["waveform"] == "sine" else 1,
            horizontal=True,
            key=f"wave_{index}",
        )
        explanatory_caption(
            "Pick sine vs cosine to illustrate how the orthogonal basis functions combine into the full signal.",
            container=container,
        )
        if container.button("Remove this component", key=f"remove_{index}"):
            return index
    return None


def sidebar_controls() -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    st.sidebar.header("Signal Builder")
    st.sidebar.write(
        "Assemble a signal term-by-term. Every control offers a short rationale so the lab feels guided."
    )
    st.sidebar.slider(
        "Master duration (s)",
        min_value=float(BASE_DURATION_RANGE[0]),
        max_value=float(BASE_DURATION_RANGE[1]),
        value=float(st.session_state.base_duration),
        key="base_duration",
    )
    explanatory_caption(
        "Longer durations give you finer frequency spacing because the FFT sees more of the waveform.",
        container=st.sidebar,
    )
    st.sidebar.slider(
        "Reference sampling rate (Hz)",
        min_value=BASE_FS_RANGE[0],
        max_value=BASE_FS_RANGE[1],
        value=int(st.session_state.base_fs),
        step=50,
        key="base_fs",
    )
    explanatory_caption(
        "This high-rate reference creates an almost-continuous signal that the rest of the tabs build upon.",
        container=st.sidebar,
    )
    components = st.session_state.components
    removal_queue = None
    for idx, comp in enumerate(components):
        removal_queue = component_block(idx, comp, st.sidebar)
        if removal_queue is not None:
            break
    if removal_queue is not None:
        components.pop(removal_queue)
    if len(components) < MAX_COMPONENTS and st.sidebar.button("Add sinusoid"):
        components.append(
            {
                "amplitude": 0.3,
                "frequency": 50.0,
                "phase": 0.0,
                "waveform": "sine",
                "active": True,
            }
        )
    st.sidebar.caption("Use 'Add sinusoid' to reach five or more components for rich spectra.")
    explanatory_caption(
        "You can layer up to seven components—enough to craft classic textbook signals.",
        container=st.sidebar,
    )
    preset_name = st.sidebar.selectbox(
        "Quick-add preset",
        list(PRESET_LIBRARY.keys()),
        help="Load classic signals instantly—then keep editing them.",
    )
    explanatory_caption(
        "Each preset demonstrates a different spectral signature so you can compare theory vs reality.",
        container=st.sidebar,
    )
    st.sidebar.info(PRESET_LIBRARY[preset_name]["description"])
    if st.sidebar.button("Load preset into components"):
        st.session_state.components = [copy.deepcopy(c) for c in PRESET_LIBRARY[preset_name]["components"]]
        st.session_state.preset_loaded = preset_name
        st.rerun()
    st.sidebar.caption("Presets provide annotated launch points so you can immediately see signature spectra.")
    latex_expr = render_component_expression(st.session_state.components)
    st.sidebar.latex(latex_expr)
    st.sidebar.success(
        "Symbols recap: amplitude multiplies each basis function, f identifies its spectral slot, and phase offsets the timing."
    )
    t = generate_time_vector(st.session_state.base_duration, st.session_state.base_fs)
    signal, component_waves = build_signal(
        st.session_state.components,
        t,
        misalign=st.session_state.misalign,
    )
    return t, signal, component_waves


def plot_time_domain(
    t: np.ndarray,
    signal: np.ndarray,
    component_waves: List[np.ndarray],
    highlight_idx: Optional[int] = None,
    sampled_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    overlay: Optional[np.ndarray] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            name="Sum",
            line=dict(color="#2E86AB", width=3),
            hovertemplate="t=%{x:.4f}s<br>amp=%{y:.3f}",
        )
    )
    for idx, comp_wave in enumerate(component_waves):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=comp_wave,
                name=f"Component {idx + 1}",
                opacity=0.3 if highlight_idx is not None and highlight_idx != idx else 0.6,
                line=dict(dash="dot" if highlight_idx == idx else "dash"),
                showlegend=False,
            )
        )
    if sampled_points is not None:
        fig.add_trace(
            go.Scatter(
                x=sampled_points[0],
                y=sampled_points[1],
                mode="markers",
                marker=dict(color="#C70039", size=8),
                name="Samples",
            )
        )
    if overlay is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=overlay,
                name="Band reconstruction",
                line=dict(color="#FF5733", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_frequency_domain(
    signal: np.ndarray,
    fs: float,
    window_name: str = "rectangular",
    beta: float = 14.0,
    zero_pad_factor: float = 1.0,
    highlight_freq: Optional[float] = None,
    highlight_band: Optional[Tuple[float, float]] = None,
    alias_freq: Optional[float] = None,
) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
    window = compute_window(window_name, len(signal), beta)
    freqs, spectrum = fft_spectrum(signal, fs, window, zero_pad_factor)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    mask = freqs >= 0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Magnitude", "Phase"))
    fig.add_trace(
        go.Scatter(
            x=freqs[mask],
            y=magnitude[mask],
            name="Magnitude",
            hovertemplate="f=%{x:.2f} Hz<br>|X|=%{y:.3f}",
        ),
        row=1,
        col=1,
    )
    if highlight_freq is not None:
        fig.add_vline(
            x=highlight_freq,
            line=dict(color="#C70039", width=2, dash="dash"),
            annotation_text="Highlighted component",
            annotation_position="top right",
        )
    if highlight_band is not None:
        fig.add_vrect(
            x0=highlight_band[0],
            x1=highlight_band[1],
            fillcolor="rgba(255,165,0,0.2)",
            layer="below",
            line_width=0,
        )
    if alias_freq is not None:
        fig.add_vline(
            x=alias_freq,
            line=dict(color="#8E44AD", width=2),
            annotation_text="Aliased location",
            annotation_position="top left",
        )
    fig.add_trace(
        go.Scatter(
            x=freqs[mask],
            y=phase[mask],
            name="Phase",
            hovertemplate="f=%{x:.2f} Hz<br>phase=%{y:.3f} rad",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="|X(f)|",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="angle",
        template="plotly_white",
    )
    return fig, freqs, spectrum, magnitude


def linked_view_section(
    t: np.ndarray,
    signal: np.ndarray,
    component_waves: List[np.ndarray],
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    freq_col, time_col = st.columns(2)
    window_name = "hann"
    freq_fig, freqs, spectrum, magnitude = plot_frequency_domain(
        signal,
        fs,
        window_name=window_name,
        highlight_freq=st.session_state.highlight_freq,
    )
    if st.session_state.highlight_time is not None:
        span = st.session_state.base_duration / 20
        time_mask = (t >= st.session_state.highlight_time - span) & (
            t <= st.session_state.highlight_time + span
        )
        if np.any(time_mask):
            localized = signal[time_mask]
            local_window = np.hanning(localized.size) if localized.size > 1 else np.ones_like(localized)
            local_freqs, local_spec = fft_spectrum(
                localized * local_window,
                fs,
                window=None,
                zero_pad_factor=2.0,
            )
            positive = local_freqs >= 0
            freq_fig.add_trace(
                go.Scatter(
                    x=local_freqs[positive],
                    y=np.abs(local_spec)[positive],
                    name="Localized spectrum",
                    line=dict(color="#FF8C00", dash="dot"),
                    opacity=0.6,
                    hovertemplate="Local f=%{x:.2f} Hz<br>|X|=%{y:.3f}",
                ),
                row=1,
                col=1,
            )
    with freq_col:
        st.info(
            "Hovering on a frequency component helps visualize which part of the time signal that tone contributes to."
        )
        freq_events = plotly_events(
            freq_fig,
            hover_event=True,
            select_event=True,
            override_height=400,
            key="freq_plot",
        )
        st.caption("Magnitude (top) and phase (bottom) update instantly as you tweak components.")
    highlight_idx = None
    if st.session_state.highlight_freq is not None:
        component_freqs = [comp["frequency"] for comp in st.session_state.components]
        if component_freqs:
            diffs = np.abs(np.array(component_freqs) - st.session_state.highlight_freq)
            highlight_idx = int(np.argmin(diffs))
    time_fig = plot_time_domain(
        t,
        signal,
        component_waves,
        highlight_idx=highlight_idx,
    )
    with time_col:
        st.info(
            "Sharp events contain many frequencies. Hover to see their spectral footprint highlighted on the right."
        )
        time_events = plotly_events(
            time_fig,
            hover_event=True,
            override_height=400,
            key="time_plot",
        )
        st.caption("Time plot overlays each component so you can see which waveform your hover is emphasizing.")
    if freq_events:
        event = freq_events[0]
        if isinstance(event, dict) and "x" in event:
            st.session_state.highlight_freq = float(event["x"])
    if time_events:
        event = time_events[0]
        if isinstance(event, dict) and "x" in event:
            st.session_state.highlight_time = float(event["x"])
    freqs_raw, spectrum_raw = fft_spectrum(signal, fs)
    return freqs_raw, spectrum_raw


def sampling_aliasing_tab(
    tab, t, signal, component_waves
):
    with tab:
        st.info(
            "Purpose: experiment with sampling knobs to see Nyquist, aliasing, and spectral resolution in action.\n"
            "Key concepts: sampling frequency, total duration, discrete FFT bins, zero padding, and alias read-outs.\n"
            "Controls: sliders below change fs, duration, and sample count, updating plots immediately."
        )
        fs = st.slider(
            "Sampling rate fs (Hz)",
            min_value=50,
            max_value=500,
            value=200,
            key="sampling_fs",
        )
        explanatory_caption(
            "fs must exceed twice the highest signal frequency to avoid aliasing (Nyquist criterion)."
        )
        duration = st.slider(
            "Total duration T (s)",
            min_value=0.25,
            max_value=2.0,
            value=1.0,
            key="sampling_duration",
        )
        explanatory_caption(
            "Longer durations create finer FFT spacing because frequency bins equal 1/T."
        )
        num_samples = st.slider(
            "Number of samples N",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            key="sampling_n",
        )
        explanatory_caption(
            "N determines how many discrete points you store. Holding fs fixed while increasing N extends duration."
        )
        watch_frequency = st.slider(
            "Frequency to monitor for aliasing (Hz)",
            min_value=0.0,
            max_value=400.0,
            value=150.0,
            key="alias_watch",
        )
        explanatory_caption(
            "Set this above fs/2 to force aliasing; the readout shows where it folds back."
        )
        zero_pad = st.checkbox("Apply zero padding", value=True, key="zero_pad_toggle")
        explanatory_caption(
            "Zero padding interpolates the FFT display for smooth curves but adds no new spectral info."
        )
        pad_factor = st.slider(
            "Zero padding factor",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            key="pad_factor",
        )
        explanatory_caption(
            "Higher factors only densify the plotted spectrum; they do not improve true resolution."
        )
        sampled_t = np.linspace(0, duration, num_samples, endpoint=False)
        sampled_signal = np.interp(sampled_t, t, signal)
        sampled_components = [np.interp(sampled_t, t, comp) for comp in component_waves]
        alias_freq = None
        nyquist = fs / 2
        if watch_frequency > nyquist:
            alias_freq = abs(watch_frequency - math.ceil(watch_frequency / fs) * fs)
        freq_fig, _, _, _ = plot_frequency_domain(
            sampled_signal,
            fs,
            window_name="hann",
            zero_pad_factor=pad_factor if zero_pad else 1.0,
            alias_freq=alias_freq,
        )
        time_fig = plot_time_domain(
            sampled_t,
            sampled_signal,
            sampled_components,
            sampled_points=(sampled_t, sampled_signal),
        )
        st.plotly_chart(time_fig, use_container_width=True)
        st.caption(
            "Red dots show discrete samples. Notice how undersampling distorts the apparent waveform."
        )
        st.plotly_chart(freq_fig, use_container_width=True)
        st.caption(
            "Aliased peaks are annotated when the monitored tone exceeds Nyquist."
        )
        st.warning(
            f"Nyquist = {nyquist:.1f} Hz. If the tone {watch_frequency:.1f} Hz exceeds it, the alias lands at {alias_freq or watch_frequency:.1f} Hz."
        )


def windowing_tab(tab, t, signal, component_waves):
    with tab:
        st.info(
            "Purpose: explore how different windows tame spectral leakage.\n"
            "Key concepts: window shapes, main-lobe width, sidelobe levels, and misaligned frequencies.\n"
            "Controls: pick a window, adjust the Kaiser beta when available, and toggle deliberate misalignment."
        )
        window_name = st.selectbox(
            "Window type",
            options=["rectangular", "hann", "hamming", "blackman", "kaiser"],
            index=1,
            key="window_type",
        )
        explanatory_caption(WINDOW_SUMMARY[window_name]["text"])
        beta = st.slider(
            "Kaiser beta (only used when Kaiser selected)",
            min_value=1.0,
            max_value=20.0,
            value=10.0,
            key="kaiser_beta",
        )
        explanatory_caption(
            "Beta trades main-lobe width for sidelobe suppression; high beta mimics Blackman-like leakage control."
        )
        st.session_state.misalign = st.checkbox(
            "Misalign frequency to bin",
            value=st.session_state.misalign,
            key="misalign_toggle",
        )
        explanatory_caption(
            "When tones fall between FFT bins, their energy smears outward—this toggle injects that scenario."
        )
        window_values = compute_window(window_name, len(signal), beta)
        st.plotly_chart(
            go.Figure(
                data=[
                    go.Scatter(x=t, y=window_values, name="Window", line=dict(color="#16A085"))
                ],
                layout=go.Layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Weight",
                    template="plotly_white",
                ),
            ),
            use_container_width=True,
        )
        st.caption("Windowing tapers the ends to reduce discontinuities that cause leakage.")
        freq_fig, _, _, _ = plot_frequency_domain(
            signal * window_values,
            st.session_state.base_fs,
            window_name="rectangular",
        )
        st.plotly_chart(freq_fig, use_container_width=True)
        st.success(
            f"Main-lobe width estimate: {WINDOW_SUMMARY[window_name]['main_lobe']} | Peak sidelobe: {WINDOW_SUMMARY[window_name]['sidelobe']}"
        )


def band_reconstruction_tab(tab, t, signal, fs, freqs, spectrum):
    with tab:
        st.info(
            "Purpose: treat the spectrum like a surgical tool—select a band, reconstruct it, and inspect the partial signal.\n"
            "Key concepts: ideal filtering, inverse FFT reconstruction, and how bandwidth controls waveform detail.\n"
            "Controls: brush the magnitude plot to set a passband, then compare the overlayed reconstruction in time."
        )
        st.write(
            "Step 1: drag-select a band on the spectrum. Step 2: compare the resulting partial signal with the original."
        )
        freq_fig, _, _, _ = plot_frequency_domain(
            signal,
            fs,
            highlight_band=st.session_state.selected_band,
        )
        events = plotly_events(
            freq_fig,
            select_event=True,
            override_height=400,
            key="band_selector",
        )
        if events:
            first_event = events[0]
            band_range: Optional[Tuple[float, float]] = None
            if isinstance(first_event, dict) and "range" in first_event and "x" in first_event["range"]:
                x0, x1 = first_event["range"]["x"]
                band_range = (float(min(x0, x1)), float(max(x0, x1)))
            else:
                xs = [float(ev["x"]) for ev in events if isinstance(ev, dict) and "x" in ev]
                if xs:
                    band_range = (min(xs), max(xs))
            if band_range:
                st.session_state.selected_band = band_range
        band = st.session_state.selected_band
        reconstructed = reconstruct_band(spectrum, freqs, band)
        time_fig = plot_time_domain(
            t,
            signal,
            [],
            overlay=reconstructed,
        )
        st.plotly_chart(time_fig, use_container_width=True)
        st.info(
            "Band selection behaves like an ideal filter. Perfectly isolating frequencies is rarely possible with physical hardware,"
            " but the FFT lets you experiment digitally."
        )


def main() -> None:
    st.set_page_config(
        page_title="Fourier Explorer Lab",
        layout="wide",
    )
    st.title("Interactive Fourier Transform Laboratory")
    st.write(
        "Every control is annotated so you not only see plots move—you understand why."
    )
    init_state()
    t, signal, component_waves = sidebar_controls()
    tabs = st.tabs(
        [
            "Signal Builder",
            "Sampling & Aliasing",
            "Windowing & Leakage",
            "Band Reconstruction",
        ]
    )
    with tabs[0]:
        st.info(
            "Purpose: craft signals, inspect their math representation, and link time vs frequency intuition.\n"
            "Key concepts: linear superposition, basis functions, and dual time/frequency viewpoints.\n"
            "Controls: tweak sidebar sliders, hover on the plots, and watch the analytic expression update live."
        )
        st.success(
            "Use the sidebar to sculpt the analytic expression below, then study how each term changes the plots."
        )
        latex_expr = render_component_expression(st.session_state.components)
        st.latex(latex_expr)
        freqs, spectrum = linked_view_section(
            t,
            signal,
            component_waves,
            st.session_state.base_fs,
        )
    sampling_aliasing_tab(tabs[1], t, signal, component_waves)
    windowing_tab(tabs[2], t, signal, component_waves)
    band_reconstruction_tab(
        tabs[3],
        t,
        signal,
        st.session_state.base_fs,
        freqs,
        spectrum,
    )


if __name__ == "__main__":
    main()
