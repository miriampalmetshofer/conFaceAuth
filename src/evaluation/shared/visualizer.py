"""Interactive visualization utilities using Plotly."""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

from evaluation.shared.models import GroupedMetrics, SegmentAnalysis


__all__ = [
    'create_risk_score_timeline',
    'create_scenario_aggregated_timeline',
    'save_interactive_plot',
    'create_grouped_comparison_plot',
    'create_device_breakdown_plot',
    'create_segment_comparison_plot',
    'create_state_distribution_pie',
    'create_risk_score_histogram',
    'create_distance_histogram'
]


def create_scenario_aggregated_timeline(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Create aggregated risk score timeline by scenario (average across all videos per scenario).

    Args:
        df: Results dataframe with 'scenario' column
        threshold: Authentication threshold

    Returns:
        Plotly figure with three lines (one per scenario: easy, angle, lighting)
    """
    if 'scenario' not in df.columns:
        raise ValueError("DataFrame must contain 'scenario' column for scenario aggregation")

    fig = go.Figure()

    # Define scenario colors
    scenario_colors = {
        'easy': '#2ecc71',      # Green
        'angle': '#f39c12',     # Orange
        'lighting': '#9b59b6'   # Purple
    }

    # Get unique scenarios
    scenarios = df['scenario'].unique()

    for scenario in sorted(scenarios):
        scenario_data = df[df['scenario'] == scenario].copy()

        # Group by frame and calculate mean risk score
        aggregated = scenario_data.groupby('frame').agg({
            'risk_score': 'mean',
            'distance': 'mean',
            'segment_type': 'first'  # Assuming same frame has same segment_type
        }).reset_index()

        # Create hover text
        hover_text = [
            f"<b>Scenario: {scenario.title()}</b><br>"
            f"Frame: {row['frame']}<br>"
            f"Avg Risk Score: {row['risk_score']:.4f}<br>"
            f"Avg Distance: {row['distance']:.4f}<br>"
            f"Segment: {row.get('segment_type', 'N/A').title()}"
            for _, row in aggregated.iterrows()
        ]

        # Add line trace
        fig.add_trace(go.Scatter(
            x=aggregated['frame'],
            y=aggregated['risk_score'],
            mode='lines',
            name=f'{scenario.title()} Scenario',
            line=dict(width=3, color=scenario_colors.get(scenario, '#95a5a6')),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            showlegend=True
        ))

    # Add threshold line
    fig.add_trace(go.Scatter(
        x=[df['frame'].min(), df['frame'].max()],
        y=[threshold, threshold],
        mode='lines',
        name='Threshold',
        line=dict(color='black', width=3, dash='dash'),
        hovertemplate=f'Threshold: {threshold}<extra></extra>',
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Average Risk Score Over Time by Scenario<br><sub>Averaged across all videos per scenario | Click legend to show/hide | Hover for details</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Frame',
        yaxis_title='Average Risk Score',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Gray",
            borderwidth=2,
            font=dict(size=11)
        ),
        template='plotly_white',
        height=700,
        margin=dict(r=250, t=100)
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Add range slider
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        type='linear'
    )

    return fig


def create_risk_score_timeline(df: pd.DataFrame, threshold: float,
                               segment_boundaries: dict = None) -> go.Figure:
    """Create interactive line plot of risk scores over time with hover-based video info.

    Args:
        df: Results dataframe
        threshold: Authentication threshold
        segment_boundaries: Optional dict with segment boundaries for coloring

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add segment boundary lines if segment_type column exists
    if 'segment_type' in df.columns:
        # Define segment colors for lines
        segment_line_colors = {
            'genuine': '#2ecc71',  # Green
            'black': '#95a5a6',     # Gray
            'imposter': '#e74c3c'   # Red
        }

        # Get unique videos to process each separately
        videos = df['video_path'].unique()

        for video_path in videos:
            video_data = df[df['video_path'] == video_path].copy()
            video_data = video_data.sort_values('frame')

            # Find segment transitions and draw vertical lines
            current_segment = None

            for idx, row in video_data.iterrows():
                if row['segment_type'] != current_segment:
                    # Draw vertical line at segment transition
                    if current_segment is not None:
                        # Get color for the new segment
                        line_color = segment_line_colors.get(row['segment_type'], '#95a5a6')

                        fig.add_vline(
                            x=row['frame'],
                            line_dash="dash",
                            line_color=line_color,
                            line_width=2,
                            opacity=0.7,
                            layer="below"
                        )

                    # Update current segment
                    current_segment = row['segment_type']

    # Get unique videos
    videos = df['video_path'].unique()

    # Color palette
    colors = px.colors.qualitative.Plotly

    for idx, video_path in enumerate(videos):
        video_data = df[df['video_path'] == video_path].copy()
        video_name = Path(video_path).stem

        # Create hover text with detailed information
        hover_text = [
            f"<b>{video_name}</b><br>"
            f"Frame: {row['frame']}<br>"
            f"Risk Score: {row['risk_score']:.4f}<br>"
            f"Distance: {row['distance']:.4f}<br>"
            f"State: {row['predicted_state']}<br>"
            f"Face Detected: {row['face_detected']}"
                    + (f"<br>Device: {row.get('device', 'N/A')}" if 'device' in video_data.columns else "")
            + (f"<br>Segment: {row.get('segment_type', 'N/A').title()}" if 'segment_type' in video_data.columns else "")
            for _, row in video_data.iterrows()
        ]

        # Add line trace (hidden by default, but clickable in legend)
        fig.add_trace(go.Scatter(
            x=video_data['frame'],
            y=video_data['risk_score'],
            mode='lines',
            name=video_name,
            line=dict(width=2, color=colors[idx % len(colors)]),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            showlegend=True,
            visible='legendonly'  # Hidden by default, but appears in legend
        ))

    # Add threshold line
    fig.add_trace(go.Scatter(
        x=[df['frame'].min(), df['frame'].max()],
        y=[threshold, threshold],
        mode='lines',
        name='Threshold',
        line=dict(color='black', width=3, dash='dash'),
        hovertemplate=f'Threshold: {threshold}<extra></extra>',
        showlegend=True
    ))

    # Add dummy traces for segment legend
    if 'segment_type' in df.columns:
        segment_info = [
            ('Genuine Segment', 'rgba(46, 204, 113, 0.5)'),
            ('Black Screen Segment', 'rgba(149, 165, 166, 0.5)'),
            ('Imposter Segment', 'rgba(231, 76, 60, 0.5)')
        ]

        for name, color in segment_info:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=name,
                marker=dict(size=10, color=color, symbol='square'),
                showlegend=True,
                hoverinfo='skip'
            ))

    # Count number of video traces (exclude threshold and segment legends)
    num_video_traces = len(videos)

    # Update layout for better interactivity
    fig.update_layout(
        title={
            'text': 'Risk Score Over Time by Video<br><sub>Videos hidden by default - Click legend items to show individual videos | Use buttons to show/hide all | Hover for details</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Frame',
        yaxis_title='Risk Score',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Gray",
            borderwidth=2,
            font=dict(size=11),
            itemclick="toggle",  # Click to toggle visibility
            itemdoubleclick="toggleothers"  # Double-click to isolate
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True] * num_video_traces + [True] + ([True] * 3 if 'segment_type' in df.columns else [])}],
                        label="Show All Videos",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": ['legendonly'] * num_video_traces + [True] + ([True] * 3 if 'segment_type' in df.columns else [])}],
                        label="Hide All Videos",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top"
            )
        ],
        template='plotly_white',
        height=700,
        margin=dict(r=250, t=100)  # Extra margin for legend and title
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Add range slider for easier navigation
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        type='linear'
    )

    return fig


def create_segment_comparison_plot(genuine_metrics: dict, imposter_metrics: dict) -> go.Figure:
    """Create comparison bar chart for genuine vs imposter segments.

    Args:
        genuine_metrics: Metrics dictionary for genuine segments
        imposter_metrics: Metrics dictionary for imposter segments

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    categories = ['Accept Rate (%)', 'Reject Rate (%)', 'Avg Risk Score', 'Avg Distance']

    genuine_values = [
        genuine_metrics.get('true_accept_rate', 0),
        genuine_metrics.get('false_reject_rate', 0),
        genuine_metrics.get('avg_risk_score', 0) * 100,  # Scale for visibility
        genuine_metrics.get('avg_distance', 0) * 100     # Scale for visibility
    ]

    imposter_values = [
        imposter_metrics.get('false_accept_rate', 0),
        imposter_metrics.get('true_reject_rate', 0),
        imposter_metrics.get('avg_risk_score', 0) * 100,
        imposter_metrics.get('avg_distance', 0) * 100
    ]

    fig.add_trace(go.Bar(
        name='Genuine User',
        x=categories,
        y=genuine_values,
        marker_color='green',
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Imposter',
        x=categories,
        y=imposter_values,
        marker_color='red',
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Genuine vs Imposter Segment Performance',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        template='plotly_white',
        height=500,
        hovermode='x'
    )

    return fig


def create_state_distribution_pie(df: pd.DataFrame, segment_type: str = None) -> go.Figure:
    """Create interactive pie chart for authentication state distribution.

    Args:
        df: Results dataframe
        segment_type: Optional filter for segment type ('genuine', 'imposter', 'black')

    Returns:
        Plotly figure
    """
    if segment_type and 'segment_type' in df.columns:
        data = df[df['segment_type'] == segment_type]
        title = f'Authentication State Distribution - {segment_type.title()} Segment'
    else:
        data = df
        title = 'Overall Authentication State Distribution'

    state_counts = data['predicted_state'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=state_counts.index,
        values=state_counts.values,
        hole=0.3,
        marker=dict(colors=['#2ecc71', '#e74c3c']),  # Green for unlocked, red for locked
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )

    return fig


def create_risk_score_histogram(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Create interactive histogram of risk score distribution.

    Args:
        df: Results dataframe
        threshold: Authentication threshold

    Returns:
        Plotly figure
    """
    # Separate by segment type if available
    if 'segment_type' in df.columns:
        genuine_data = df[df['segment_type'] == 'genuine']['risk_score']
        imposter_data = df[df['segment_type'] == 'imposter']['risk_score']

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=genuine_data,
            name='Genuine User',
            opacity=0.7,
            marker_color='green',
            nbinsx=50,
            hovertemplate='Risk Score: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))

        fig.add_trace(go.Histogram(
            x=imposter_data,
            name='Imposter',
            opacity=0.7,
            marker_color='red',
            nbinsx=50,
            hovertemplate='Risk Score: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(barmode='overlay')
    else:
        fig = go.Figure(data=[go.Histogram(
            x=df['risk_score'],
            nbinsx=50,
            marker_color='steelblue',
            hovertemplate='Risk Score: %{x:.4f}<br>Count: %{y}<extra></extra>'
        )])

    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"Threshold: {threshold}",
        annotation_position="top"
    )

    fig.update_layout(
        title='Risk Score Distribution',
        xaxis_title='Risk Score',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500,
        hovermode='x'
    )

    return fig


def create_distance_histogram(df: pd.DataFrame) -> go.Figure:
    """Create interactive histogram of distance distribution.

    Args:
        df: Results dataframe

    Returns:
        Plotly figure
    """
    # Separate by segment type if available
    if 'segment_type' in df.columns:
        genuine_data = df[df['segment_type'] == 'genuine']['distance']
        imposter_data = df[df['segment_type'] == 'imposter']['distance']

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=genuine_data,
            name='Genuine User',
            opacity=0.7,
            marker_color='green',
            nbinsx=50,
            hovertemplate='Distance: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))

        fig.add_trace(go.Histogram(
            x=imposter_data,
            name='Imposter',
            opacity=0.7,
            marker_color='red',
            nbinsx=50,
            hovertemplate='Distance: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(barmode='overlay')
    else:
        fig = go.Figure(data=[go.Histogram(
            x=df['distance'],
            nbinsx=50,
            marker_color='green',
            hovertemplate='Distance: %{x:.4f}<br>Count: %{y}<extra></extra>'
        )])

    fig.update_layout(
        title='Face Distance Distribution',
        xaxis_title='Distance',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500,
        hovermode='x'
    )

    return fig


def save_interactive_plot(fig: go.Figure, output_path: Path, filename: str) -> Path:
    """Save Plotly figure as interactive HTML.

    Args:
        fig: Plotly figure
        output_path: Output directory
        filename: Filename (should end with .html)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not filename.endswith('.html'):
        filename = filename.replace('.png', '.html')
        if not filename.endswith('.html'):
            filename += '.html'

    output_file = output_path / filename
    fig.write_html(output_file)

    print(f"Saved interactive plot to {output_file}")
    return output_file


def create_grouped_comparison_plot(grouped_metrics: GroupedMetrics) -> go.Figure:
    """Create comparison plot for grouped metrics (device, scenario, environment, etc.).

    Args:
        grouped_metrics: GroupedMetrics dataclass

    Returns:
        Plotly figure
    """
    groups = list(grouped_metrics.groups.keys())

    # Extract metrics for plotting
    genuine_tar = []
    genuine_frr = []
    imposter_trr = []
    imposter_far = []

    for group_name in groups:
        analysis = grouped_metrics.groups[group_name]
        genuine_tar.append(analysis.genuine.true_accept_rate)
        genuine_frr.append(analysis.genuine.false_reject_rate)
        imposter_trr.append(analysis.imposter.true_reject_rate)
        imposter_far.append(analysis.imposter.false_accept_rate)

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='True Accept Rate',
        x=groups,
        y=genuine_tar,
        marker_color='#2ecc71',
        hovertemplate='<b>%{x}</b><br>TAR: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='False Reject Rate',
        x=groups,
        y=genuine_frr,
        marker_color='#f39c12',
        hovertemplate='<b>%{x}</b><br>FRR: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='True Reject Rate',
        x=groups,
        y=imposter_trr,
        marker_color='#3498db',
        hovertemplate='<b>%{x}</b><br>TRR: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='False Accept Rate',
        x=groups,
        y=imposter_far,
        marker_color='#e74c3c',
        hovertemplate='<b>%{x}</b><br>FAR: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=f'Authentication Performance by {grouped_metrics.dimension_name.title()}',
        xaxis_title=grouped_metrics.dimension_name.title(),
        yaxis_title='Rate (%)',
        barmode='group',
        template='plotly_white',
        height=500,
        hovermode='x',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_device_breakdown_plot(device_metrics: GroupedMetrics) -> go.Figure:
    """Create device-specific breakdown visualization.

    Args:
        device_metrics: GroupedMetrics dataclass with device metrics

    Returns:
        Plotly figure with subplots for each device
    """
    devices = list(device_metrics.groups.keys())
    num_devices = len(devices)

    # Create subplots - one row per device
    fig = make_subplots(
        rows=num_devices,
        cols=2,
        subplot_titles=[f'{dev} - Genuine' for dev in devices] + [f'{dev} - Imposter' for dev in devices],
        specs=[[{'type': 'bar'}, {'type': 'bar'}] for _ in range(num_devices)],
        vertical_spacing=0.15 / num_devices if num_devices > 1 else 0.15
    )

    for idx, device in enumerate(devices, start=1):
        analysis = device_metrics.groups[device]

        genuine = analysis.genuine
        fig.add_trace(
            go.Bar(
                x=['TAR', 'FRR'],
                y=[genuine.true_accept_rate, genuine.false_reject_rate],
                marker_color=['#2ecc71', '#f39c12'],
                name=f'{device} Genuine',
                showlegend=False,
                hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
            ),
            row=idx,
            col=1
        )

        imposter = analysis.imposter
        fig.add_trace(
            go.Bar(
                x=['TRR', 'FAR'],
                y=[imposter.true_reject_rate, imposter.false_accept_rate],
                marker_color=['#3498db', '#e74c3c'],
                name=f'{device} Imposter',
                showlegend=False,
                hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
            ),
            row=idx,
            col=2
        )

    fig.update_layout(
        title='Device-Specific Performance Breakdown',
        template='plotly_white',
        height=300 * num_devices,
        showlegend=False
    )

    fig.update_yaxes(title_text='Rate (%)', range=[0, 105])

    return fig
