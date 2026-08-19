"""Static PNG summary figure for a set of authentication sessions.

Combines a trust-score distribution, per-device error rates, and locked/unlocked
proportions for the genuine and imposter segments into one overview figure.
"""
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from evaluation.shared.models import DeviceMetrics, RunConfig, Segment, Session


# Rate metrics shown in the per-device bar chart: (label, color, value accessor).
_PLOTTED_RATES = [
    ("FRR", '#e74c3c', lambda m: m.false_reject_rate),
    ("FAR", '#f39c12', lambda m: m.false_accept_rate),
    ("GT", '#27ae60', lambda m: m.mean_genuine_trust),
]


def plot_trust_distribution(ax: plt.Axes, sessions: list[Session], threshold: float) -> None:
    """Plot a trust-score distribution histogram for genuine and imposter segments."""
    genuine_trust = [s.trust_score for session in sessions for s in session.genuine.samples]
    imposter_trust = [s.trust_score for session in sessions for s in session.imposter.samples]

    ax.hist([genuine_trust, imposter_trust], bins=30, label=['Genuine', 'Imposter'],
            color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Trust Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Trust Score Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_error_rates_by_device(ax: plt.Axes, device_metrics: list[DeviceMetrics]) -> None:
    """Plot an authentication-rates bar chart by device."""
    devices = [dm.device for dm in device_metrics]

    x = np.arange(len(devices))
    width = 0.2
    offset_start = -(len(_PLOTTED_RATES) - 1) / 2  # Center the bars

    for i, (label, color, value_of) in enumerate(_PLOTTED_RATES):
        values = [value_of(dm.metrics) for dm in device_metrics]
        ax.bar(x + (offset_start + i) * width, values, width, label=label, color=color, alpha=0.8)

    ax.set_ylabel('Rate (%)')
    ax.set_xlabel('Device')
    ax.set_title('Authentication Rates by Device')
    ax.set_xticks(x)
    ax.set_xticklabels([d.value.upper() for d in devices])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])


def plot_state_pie_chart(
    ax: plt.Axes,
    sessions: list[Session],
    segment_of: Callable[[Session], Segment],
    title: str,
) -> None:
    """Plot a locked/unlocked proportion pie chart for a chosen segment of each session."""
    unlocked = 0
    locked = 0
    for session in sessions:
        for sample in segment_of(session).samples:
            if sample.is_authenticated:
                unlocked += 1
            else:
                locked += 1

    ax.pie([unlocked, locked], labels=['Unlocked', 'Locked'], autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax.set_title(title)


def create_summary_visualization(
        sessions: list[Session],
        config: RunConfig,
        device_metrics: list[DeviceMetrics],
        title: str,
) -> plt.Figure:
    """Create a comprehensive PNG summary figure."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    plot_trust_distribution(fig.add_subplot(gs[0, 0]), sessions, config.threshold)
    plot_error_rates_by_device(fig.add_subplot(gs[0, 1:]), device_metrics)
    plot_state_pie_chart(fig.add_subplot(gs[1, 0]), sessions, lambda s: s.genuine,
                         "Genuine Segments\n(Should be Unlocked)")
    plot_state_pie_chart(fig.add_subplot(gs[1, 1]), sessions, lambda s: s.imposter,
                         "Imposter Segments\n(Should be Locked)")
    fig.add_subplot(gs[1, 2]).axis('off')

    fig.suptitle(title, fontsize=20, y=0.995)

    return fig
