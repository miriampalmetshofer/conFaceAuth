"""Interactive trust-score timelines built from authentication sessions.

Each session contributes one trust-over-time line. Time comes from each sample's
``timestamp_ms`` shown in seconds. The all-videos timeline additionally draws the
mean trust trajectory and unauthorized-user lockout-time markers for the paper.
"""
from typing import Iterator

import numpy as np
import plotly.graph_objects as go

from face_auth.domain import Device, Scenario
from evaluation.shared.models import AuthenticationMetrics, FrameResult, RunConfig, Session
from evaluation.shared.plots.segments import (
    add_segment_backgrounds,
    segment_boundaries,
    segment_header_annotations,
)


def _session_samples(session: Session) -> Iterator[FrameResult]:
    """Yield a session's samples in timeline order across all its segments."""
    yield from session.genuine.samples
    if session.black is not None:
        yield from session.black.samples
    yield from session.imposter.samples


def _show_hide_buttons(num_traces: int) -> list[dict]:
    """Create Show All / Hide All buttons for a plotly figure."""
    return [
        dict(label="Show All", method="update", args=[{"visible": [True] * num_traces}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * num_traces}]),
    ]


def _add_session_traces(fig: go.Figure, sessions: list[Session]) -> None:
    """Add one trust-over-time trace per session, each an auto-assigned distinct color."""
    for session in sessions:
        samples = list(_session_samples(session))
        customdata = [
            [
                f"{s.similarity:.4f}" if s.similarity is not None else "N/A",
                "Yes" if s.face_detected else "No"
            ]
            for s in samples
        ]

        fig.add_trace(go.Scatter(
            x=[s.timestamp_ms / 1000.0 for s in samples],
            y=[s.trust_score for s in samples],
            mode='lines',
            name=session.csv_path.stem,
            line=dict(width=2),
            customdata=customdata,
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Time: %{x:.2f}s<br>'
                'Trust: %{y:.4f}<br>'
                'Similarity: %{customdata[0]}<br>'
                'Face: %{customdata[1]}'
                '<extra></extra>'
            ),
            visible=True
        ))


def _base_timeline_figure(
        sessions: list[Session],
        title: str,
        threshold: float,
        segments: dict,
) -> go.Figure:
    """Create a trust timeline figure with one trace per session."""
    fig = go.Figure()

    add_segment_backgrounds(fig, segments)
    _add_session_traces(fig, sessions)

    fig.add_hline(y=threshold, line_dash="dash", line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Trust Score",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=600,
        margin=dict(t=100),
        template='plotly_white',
        font=dict(size=18),
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=_show_hide_buttons(len(sessions)),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top"
            )
        ]
    )

    return fig


def _mean_trust_over_time(sessions: list[Session]) -> tuple[list[float], list[float]]:
    """Return (times_seconds, mean_trust), the trust score averaged over all sessions
    per whole second."""
    trust_by_second: dict[int, list[float]] = {}
    for session in sessions:
        for sample in _session_samples(session):
            second = round(sample.timestamp_ms / 1000.0)
            trust_by_second.setdefault(second, []).append(sample.trust_score)
    seconds = sorted(trust_by_second.keys())
    return (
        [float(s) for s in seconds],
        [float(np.mean(trust_by_second[s])) for s in seconds],
    )


def _add_lockout_markers(
        fig: go.Figure,
        metrics: AuthenticationMetrics,
        threshold: float,
        segments: dict,
) -> None:
    """Add mean and P90 unauthorized-user lockout-time markers on the threshold line."""
    imposter_start = segments['imposter'][0]

    if metrics.imposter_lockout_time.mean is not None:
        fig.add_trace(go.Scatter(
            x=[imposter_start + metrics.imposter_lockout_time.mean],
            y=[threshold],
            mode='markers',
            name='Mean ULT',
            marker=dict(size=15, color='rgba(142, 68, 173, 0.9)', symbol='diamond',
                        line=dict(width=2, color='white')),
            showlegend=True,
            hovertemplate=f'Mean ULT: {metrics.imposter_lockout_time.mean:.1f}s<extra></extra>'
        ))

    if metrics.imposter_lockout_time.p90 is not None:
        fig.add_trace(go.Scatter(
            x=[imposter_start + metrics.imposter_lockout_time.p90],
            y=[threshold],
            mode='markers',
            name='P90 ULT',
            marker=dict(size=15, color='rgba(231, 76, 60, 0.9)', symbol='diamond',
                        line=dict(width=2, color='white')),
            showlegend=True,
            hovertemplate=f'P90 ULT: {metrics.imposter_lockout_time.p90:.1f}s<extra></extra>'
        ))


def _move_lockout_markers_to_front(fig: go.Figure) -> None:
    """Render lockout markers above the threshold lines."""
    marker_names = {"Mean ULT", "P90 ULT"}
    markers = [trace for trace in fig.data if trace.name in marker_names]
    others = [trace for trace in fig.data if trace.name not in marker_names]
    fig.data = tuple(others + markers)


def create_trust_timeline_all_videos(
        sessions: list[Session],
        config: RunConfig,
        study_name: str,
        metrics: AuthenticationMetrics,
) -> go.Figure:
    """Create the paper trust-score timeline across all sessions.

    Draws every session faintly, the mean trust trajectory, the threshold, and the
    unauthorized-user lockout-time markers.
    """
    segments = segment_boundaries(config)
    threshold = config.threshold

    title = f"{study_name} - Trust Score Timeline (All Videos)<br><sub>Threshold: {threshold}</sub>"
    fig = _base_timeline_figure(sessions, title, threshold, segments)
    fig.update_layout(title=None)

    # Hide individual session legend entries, drop the show/hide buttons, and remove
    # only the threshold hline shape while keeping the vrect segment backgrounds.
    fig.update_traces(showlegend=False)
    fig.layout.updatemenus = ()
    fig.layout.shapes = [s for s in fig.layout.shapes if not (s.type == 'line' and s.xref == 'paper')]

    mean_times, mean_trust = _mean_trust_over_time(sessions)
    x_end = mean_times[-1] if mean_times else segments['imposter'][1]
    if mean_times:
        fig.add_trace(go.Scatter(
            x=mean_times,
            y=mean_trust,
            mode='lines',
            name='Mean Trust',
            line=dict(color='black', width=3),
            showlegend=True,
            hovertemplate='Mean Trust: %{y:.4f}<br>Time: %{x:.2f}s<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=[0, x_end],
        y=[threshold, threshold],
        mode='lines',
        name='Threshold',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=True,
        hovertemplate=f'Threshold: {threshold}<extra></extra>'
    ))

    # Invisible trace so the black transition appears in the legend next to Threshold.
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Black Frames',
        marker=dict(size=10, color='rgba(149, 165, 166, 0.3)', symbol='square'),
        showlegend=True,
        hoverinfo='skip'
    ))

    _add_lockout_markers(fig, metrics, threshold, segments)
    _move_lockout_markers_to_front(fig)

    fig.update_layout(
        annotations=segment_header_annotations(segments),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.085,
            xanchor='right',
            x=1.0,
            font=dict(size=22, family='Latin Modern Roman, serif', color='black')
        ),
        margin=dict(l=80, r=20, t=78, b=60),
        font=dict(size=22, family='Latin Modern Roman, serif', color='black'),
        xaxis=dict(title_font=dict(size=22, family='Latin Modern Roman, serif', color='black'), tickfont=dict(size=18, family='Latin Modern Roman, serif', color='black')),
        yaxis=dict(title_font=dict(size=22, family='Latin Modern Roman, serif', color='black'), tickfont=dict(size=18, family='Latin Modern Roman, serif', color='black'))
    )

    return fig


def create_trust_timeline_by_device(
        sessions: list[Session],
        config: RunConfig,
        devices: list[Device],
        study_name: str,
) -> list[go.Figure]:
    """Create one interactive trust-score timeline per device."""
    segments = segment_boundaries(config)
    figures = []

    for device in devices:
        device_sessions = [s for s in sessions if s.device == device]
        if not device_sessions:
            raise ValueError(f"No sessions found for device: {device.value}")

        title = f"{study_name} - {device.value.upper()}<br><sub>Threshold: {config.threshold}</sub>"
        figures.append(
            _base_timeline_figure(device_sessions, title, config.threshold, segments)
        )

    return figures


_SCENARIO_COLORS = {
    Scenario.EASY: '#2ecc71',
    Scenario.ANGLE: '#3498db',
    Scenario.LIGHTING: '#f39c12',
}


def create_trust_timeline_by_scenario(
        sessions: list[Session],
        config: RunConfig,
        scenarios: list[Scenario],
        study_name: str,
) -> list[go.Figure]:
    """Create one interactive trust-score timeline per scenario."""
    segments = segment_boundaries(config)
    figures = []

    for scenario in scenarios:
        scenario_sessions = [s for s in sessions if s.genuine_identity.scenario == scenario]
        if not scenario_sessions:
            raise ValueError(f"No sessions found for scenario: {scenario.value}")

        title = f"{study_name} - {scenario.value.upper()}<br><sub>Threshold: {config.threshold}</sub>"
        figures.append(
            _base_timeline_figure(scenario_sessions, title, config.threshold, segments)
        )

    return figures


def create_aggregated_trust_timeline_by_scenario(
        sessions: list[Session],
        config: RunConfig,
        scenarios: list[Scenario],
        study_name: str,
) -> go.Figure:
    """Create a single timeline with one mean-trust line per scenario."""
    segments = segment_boundaries(config)
    fig = go.Figure()
    add_segment_backgrounds(fig, segments)

    for scenario in scenarios:
        scenario_sessions = [s for s in sessions if s.genuine_identity.scenario == scenario]
        if not scenario_sessions:
            raise ValueError(f"No sessions found for scenario: {scenario.value}")

        times, mean_trust = _mean_trust_over_time(scenario_sessions)
        fig.add_trace(go.Scatter(
            x=times,
            y=mean_trust,
            mode='lines',
            name=scenario.value.upper(),
            line=dict(color=_SCENARIO_COLORS.get(scenario, '#95a5a6'), width=3),
            hovertemplate=(
                f'<b>{scenario.value.upper()}</b><br>'
                'Time: %{x:.0f}s<br>'
                'Mean Trust: %{y:.4f}'
                '<extra></extra>'
            ),
        ))

    fig.add_hline(
        y=config.threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold ({config.threshold})",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"{study_name} - Aggregated Trust Score by Scenario<br><sub>Threshold: {config.threshold}</sub>",
        xaxis_title="Time (seconds)",
        yaxis_title="Trust Score",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=600,
        margin=dict(t=100),
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig
