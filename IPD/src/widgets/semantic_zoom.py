import plotly.graph_objects as go
import numpy as np
from src.Dataset import Dataset
from typing import List

def _calculate_zoom_range(embeddings_2d: np.ndarray, indices: List[int], padding_factor: float = 0.1):
    """
    Calculates the x and y ranges to zoom into a subset of points.
    """
    if not indices or len(indices) == 0:
        return None, None

    target_embeddings = embeddings_2d[indices]

    if target_embeddings.shape[0] == 0:
        return None, None
        
    if target_embeddings.shape[0] == 1:
        x_center, y_center = target_embeddings[0, 0], target_embeddings[0, 1]
        x_span = np.max(embeddings_2d[:, 0]) - np.min(embeddings_2d[:, 0])
        y_span = np.max(embeddings_2d[:, 1]) - np.min(embeddings_2d[:, 1])
        x_padding, y_padding = x_span * 0.05, y_span * 0.05
        x_range = [x_center - x_padding, x_center + x_padding]
        y_range = [y_center - y_padding, y_center + y_padding]
        return x_range, y_range

    x_min, x_max = np.min(target_embeddings[:, 0]), np.max(target_embeddings[:, 0])
    y_min, y_max = np.min(target_embeddings[:, 1]), np.max(target_embeddings[:, 1])
    x_padding = (x_max - x_min) * padding_factor
    y_padding = (y_max - y_min) * padding_factor
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
    
    # Apply same outlier filtering as in scatterplot_explorer for consistency
    if len(embeddings_2d) > 0:
        center = np.mean(embeddings_2d, axis=0)
        distances = np.linalg.norm(embeddings_2d - center, axis=1)
        threshold = np.mean(distances) + 3 * np.std(distances)
        non_outlier_mask = distances < threshold
        df = df[non_outlier_mask]
        embeddings_2d = embeddings_2d[non_outlier_mask]
    
    new_annotations = []

    for topic_id in topic_ids:
        if topic_id == -1:
            continue
        
        topic_mask = (df['topic'] == topic_id)
        if not topic_mask.any():
            continue

        # Safety check: ensure we have points for this topic
        topic_embeddings = embeddings_2d[topic_mask]
        topic_data = df[topic_mask]
        
        if len(topic_embeddings) == 0 or len(topic_data) == 0:
            continue
            
        centroid = np.mean(topic_embeddings, axis=0)
        
        # Safety check: ensure topic_name column exists and has data
        if 'topic_name' in topic_data.columns and not topic_data['topic_name'].empty:
            label = topic_data['topic_name'].iloc[0]
        else:
            label = f"Topic {topic_id}"
        
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
    """
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    
    # Apply same outlier filtering as in scatterplot_explorer for consistency
    if len(embeddings_2d) > 0:
        center = np.mean(embeddings_2d, axis=0)
        distances = np.linalg.norm(embeddings_2d - center, axis=1)
        threshold = np.mean(distances) + 3 * np.std(distances)
        non_outlier_mask = distances < threshold
        df = df[non_outlier_mask]
        embeddings_2d = embeddings_2d[non_outlier_mask]
    
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

        trace_indices = np.array(trace.customdata)
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

    fig.update_layout(
        xaxis_range=x_range,
        yaxis_range=y_range,
        annotations=new_annotations
    )

    return fig


def focus_on_segments(fig: go.Figure, ordered_segment_indices: List[int], count: int) -> go.Figure:
    """
    Adjusts the figure to zoom in on the top 'count' segments and highlights them.
    It also updates annotations to show the child labels for the topics these segments belong to.
    """
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    
    # Apply same outlier filtering as in scatterplot_explorer for consistency
    if len(embeddings_2d) > 0:
        center = np.mean(embeddings_2d, axis=0)
        distances = np.linalg.norm(embeddings_2d - center, axis=1)
        threshold = np.mean(distances) + 3 * np.std(distances)
        non_outlier_mask = distances < threshold
        df = df[non_outlier_mask]
        embeddings_2d = embeddings_2d[non_outlier_mask]
    
    segments_to_focus_on = ordered_segment_indices[:count]

    if not segments_to_focus_on:
        return fig

    x_range, y_range = _calculate_zoom_range(embeddings_2d, segments_to_focus_on)
        
    for trace in fig.data:
        if trace.mode != 'markers' or trace.customdata is None:
            continue
            
        trace_indices = np.array(trace.customdata)
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

    fig.update_layout(
        xaxis_range=x_range,
        yaxis_range=y_range,
        annotations=new_annotations
    )

    return fig