from dash import html
import plotly.graph_objects as go
import numpy as np
from src.Dataset import Dataset
from typing import List

def get_indices_from_customdata(customdata):
    """Extract indices from customdata, supporting both 1D and 2D formats"""
    customdata_array = np.array(customdata)
    if customdata_array.ndim == 1:
        return customdata_array  # dev_pascal format (1D array of indices)
    else:
        return customdata_array[:, -1].astype(int)  # dev_joppe format (2D array, last column is indices)

def create_focus_box():
    """Return an empty container; the hover callback injects card content here."""
    return html.Div(id='focus-box', className='focus-box')

def _calculate_zoom_range(embeddings_2d: np.ndarray, indices: List[int]):
    """
    Calculates the x and y ranges to zoom into a subset of points.
    Simple zoom without any aspect ratio manipulation or shearing.
    """
    if not indices or len(indices) == 0:
        return None, None

    target_embeddings = embeddings_2d[indices]

    if target_embeddings.shape[0] == 0:
        return None, None
        
    if target_embeddings.shape[0] == 1:
        x_center, y_center = target_embeddings[0, 0], target_embeddings[0, 1]
        padding = 1.0
        
        x_range = [x_center - padding, x_center + padding]
        y_range = [y_center - padding, y_center + padding]
        return x_range, y_range

    x_min, x_max = np.min(target_embeddings[:, 0]), np.max(target_embeddings[:, 0])
    y_min, y_max = np.min(target_embeddings[:, 1]), np.max(target_embeddings[:, 1])
    
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2
    
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]
    
    return x_range, y_range

def _generate_topic_annotations(topic_ids: List[int]) -> List[dict]:
    """
    Generates a list of annotation dictionaries for the given topic IDs.
    These are the "child" topic labels.
    """
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    new_annotations = []

    for topic_id in topic_ids:
        if topic_id == -1:
            continue
        
        topic_mask = (df['topic'] == topic_id)
        if not topic_mask.any():
            continue

        centroid = np.mean(embeddings_2d[topic_mask], axis=0)
        label = df[topic_mask]['topic_name'].iloc[0]
        
        new_annotations.append(dict(
            x=centroid[0], 
            y=centroid[1], 
            text=f"<b>{label}</b>", 
            showarrow=False, 
            font=dict(size=14, color='black'),
            bgcolor='rgba(255, 255, 255, 0.7)',
            borderpad=4
        ))
        
    return new_annotations


def focus_on_topics(fig: go.Figure, ordered_topic_ids: List[int], count: int) -> go.Figure:
    """
    Adjusts the figure to zoom in on the top 'count' topics and highlights them.
    It also updates the annotations to show only the child labels for the focused topics.
    Simple zoom without distortion.
    """
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    topics_to_focus_on = ordered_topic_ids[:count]

    if not topics_to_focus_on:
        return fig

    focus_indices = df[df['topic'].isin(topics_to_focus_on)].index.tolist()
    if not focus_indices:
        return fig

    x_range, y_range = _calculate_zoom_range(embeddings_2d, focus_indices)
    
    for trace in fig.data:
        if trace.mode != 'markers' or trace.customdata is None:
            continue

        trace_indices = get_indices_from_customdata(trace.customdata)
        points_in_trace_df = df.loc[trace_indices]
        is_focus_topic_mask = points_in_trace_df['topic'].isin(topics_to_focus_on).values

        new_sizes = np.full(len(trace_indices), 4)
        new_sizes[is_focus_topic_mask] = 10
        trace.marker.size = new_sizes.tolist()

        original_colors = np.array(trace.marker.color)
        new_colors = np.full(len(trace_indices), 'rgba(200, 200, 200, 0.4)', dtype=object)
        if original_colors.size == 1:
            original_colors = np.repeat(original_colors, len(trace_indices))
        new_colors[is_focus_topic_mask] = original_colors[is_focus_topic_mask]
        trace.marker.color = new_colors.tolist()

    new_annotations = _generate_topic_annotations(topics_to_focus_on)
    
    fig.layout.annotations = new_annotations
    if x_range and y_range:
        fig.layout.xaxis.range = x_range
        fig.layout.yaxis.range = y_range

    return fig


def focus_on_segments(fig: go.Figure, ordered_segment_indices: List[int], count: int) -> go.Figure:
    """
    Adjusts the figure to zoom in on the top 'count' segments and highlights them.
    It also updates annotations to show the child labels for the topics these segments belong to.
    Simple zoom without distortion.
    """
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    segments_to_focus_on = ordered_segment_indices[:count]

    if not segments_to_focus_on:
        return fig

    x_range, y_range = _calculate_zoom_range(embeddings_2d, segments_to_focus_on)
        
    for trace in fig.data:
        if trace.mode != 'markers' or trace.customdata is None:
            continue
            
        trace_indices = get_indices_from_customdata(trace.customdata)
        is_focus_segment_mask = np.isin(trace_indices, segments_to_focus_on)

        new_sizes = np.full(len(trace_indices), 4)
        new_sizes[is_focus_segment_mask] = 12
        trace.marker.size = new_sizes.tolist()

        new_colors = np.full(len(trace_indices), 'rgba(200, 200, 200, 0.4)', dtype=object)
        new_colors[is_focus_segment_mask] = 'red'
        trace.marker.color = new_colors.tolist()

    focused_segments_df = df.loc[segments_to_focus_on]
    unique_topics_in_focus = focused_segments_df['topic'].unique().tolist()
    
    new_annotations = _generate_topic_annotations(unique_topics_in_focus)
    
    fig.layout.annotations = new_annotations
    if x_range and y_range:
        fig.layout.xaxis.range = x_range
        fig.layout.yaxis.range = y_range

    return fig