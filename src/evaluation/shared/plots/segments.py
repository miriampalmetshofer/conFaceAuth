"""Composed-video segment boundaries and their plotly backgrounds.

A composed video runs a genuine segment, then a black transition, then an imposter
segment. Boundaries are taken from the run's configured segment durations.
"""
import plotly.graph_objects as go

from evaluation.shared.models import RunConfig


def segment_boundaries(config: RunConfig) -> dict[str, tuple[float, float]]:
    """Return (start, end) in seconds for each composed-video segment."""
    genuine_end = config.genuine_user_seconds
    black_end = genuine_end + config.black_screen_seconds
    imposter_end = black_end + config.impostor_seconds
    return {
        'genuine': (0.0, genuine_end),
        'black': (genuine_end, black_end),
        'imposter': (black_end, imposter_end),
    }


def segment_header_annotations(segments: dict) -> list[dict]:
    """Create paper figure segment headers between the legend and the plot area."""
    if not segments:
        return []

    return [
        dict(
            x=(segments['genuine'][0] + segments['genuine'][1]) / 2,
            y=1.005,
            xref='x',
            yref='paper',
            text='<b>Genuine User</b>',
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=20, color='black')
        ),
        dict(
            x=(segments['imposter'][0] + segments['imposter'][1]) / 2,
            y=1.005,
            xref='x',
            yref='paper',
            text='<b>Unauthorized User</b>',
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=20, color='black')
        )
    ]


def add_segment_backgrounds(fig: go.Figure, segments: dict) -> None:
    """Add colored background regions and boundary lines for the video segments."""
    if not segments:
        return

    colors = {
        'genuine': 'rgba(46, 204, 113, 0.1)',   # Green
        'black': 'rgba(149, 165, 166, 0.1)',     # Gray
        'imposter': 'rgba(231, 76, 60, 0.1)'     # Red
    }

    line_colors = {
        'genuine': 'rgba(46, 204, 113, 0.6)',
        'black': 'rgba(149, 165, 166, 0.6)',
        'imposter': 'rgba(231, 76, 60, 0.6)'
    }

    for segment_name, (start, end) in segments.items():
        if start >= end:
            continue
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=colors.get(segment_name, 'rgba(0, 0, 0, 0.05)'),
            layer="below",
            line_width=0
        )

    boundaries: list[tuple[float, str]] = []
    seen_positions: set[float] = set()
    for segment_name, (start, end) in segments.items():
        if start > 0 and start not in seen_positions:
            boundaries.append((start, segment_name))
            seen_positions.add(start)
        if end not in seen_positions:
            boundaries.append((end, segment_name))
            seen_positions.add(end)

    for position, segment_name in boundaries:
        fig.add_vline(
            x=position,
            line_color=line_colors.get(segment_name, 'rgba(0, 0, 0, 0.3)'),
            line_width=2,
            line_dash="dash",
            layer="below"
        )
