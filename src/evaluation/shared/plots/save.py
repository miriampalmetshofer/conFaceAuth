"""Saving plotly and matplotlib figures to disk."""
from pathlib import Path

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from evaluation.shared.font_config import (
    apply_plotly_font,
    plotly_font_style_tag,
    plotly_html,
)


def save_html(fig: go.Figure | list[go.Figure], output_path: Path, filename: str) -> Path:
    """Save plotly figure(s) as HTML, stacking multiple figures vertically."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    if isinstance(fig, list):
        html_content = (
            '<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            f'{plotly_font_style_tag()}</head><body>\n'
        )

        for i, single_fig in enumerate(fig):
            apply_plotly_font(single_fig)
            html_content += f'<div id="plot_{i}"></div>\n'
            html_content += f'<script>Plotly.newPlot("plot_{i}", {single_fig.to_json()});</script>\n'

        html_content += '</body></html>'

        filepath.write_text(html_content)
    else:
        apply_plotly_font(fig)
        filepath.write_text(plotly_html(fig))

    return filepath


def save_png(fig: plt.Figure, output_path: Path, filename: str) -> Path:
    """Save a matplotlib figure as PNG."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def save_plotly_png(fig: go.Figure, output_path: Path, filename: str, width: int = 1200, height: int = 600) -> Path:
    """Save a plotly figure as PNG using kaleido."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    apply_plotly_font(fig)

    try:
        import plotly.io as pio
        pio.kaleido.scope.mathjax = None
        fig.write_image(str(filepath), width=width, height=height, scale=2, engine="kaleido")
    except Exception as e:
        print(f"Warning: Could not export PNG with Latin Modern Roman font. Error: {e}")
        print("The font may not be available to Kaleido. Attempting fallback...")
        fig.write_image(str(filepath), width=width, height=height, scale=2)

    return filepath
