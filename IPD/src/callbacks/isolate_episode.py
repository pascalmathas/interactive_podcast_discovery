from dash import callback, Output, Input, State, no_update
from src.Dataset import Dataset
from src.widgets.scatterplot_explorer import create_scatter_figure
import numpy as np


def _calculate_annotations_for_view(relayout_data, df, embeddings_2d):
    """A helper to calculate annotations based on a specific view, adapted from semantic_zoom."""
    if not relayout_data or 'xaxis.range[0]' not in relayout_data:
        from src.widgets.scatterplot_explorer import _get_spatially_distributed_parent_annotations
        return _get_spatially_distributed_parent_annotations(df, embeddings_2d)

    x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
    y_range = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]

    visible_mask = (
        (embeddings_2d[:, 0] >= x_range[0]) & (embeddings_2d[:, 0] <= x_range[1]) &
        (embeddings_2d[:, 1] >= y_range[0]) & (embeddings_2d[:, 1] <= y_range[1])
    )
    visible_points_count = np.sum(visible_mask)
    visible_df = df[visible_mask]
    
    DATAPOINT_THRESHOLD = 2500

    new_annotations = []
    if visible_points_count < DATAPOINT_THRESHOLD:
        for topic_id in visible_df['topic'].unique():
            if topic_id == -1: continue
            topic_mask_in_view = (visible_df['topic'] == topic_id)
            if topic_mask_in_view.any():
                centroid = np.mean(embeddings_2d[visible_mask][topic_mask_in_view], axis=0)
                label = visible_df[topic_mask_in_view]['topic_name'].iloc[0]
                new_annotations.append(dict(x=centroid[0], y=centroid[1], text=label, showarrow=False, font=dict(size=14, color='black')))
    else:
        if 'parent_topic_name' in visible_df.columns:
            for parent_name in visible_df['parent_topic_name'].unique():
                parent_mask_in_view = (visible_df['parent_topic_name'] == parent_name)
                if parent_mask_in_view.any():
                    centroid = np.mean(embeddings_2d[visible_mask][parent_mask_in_view], axis=0)
                    new_annotations.append(dict(x=centroid[0], y=centroid[1], text=f"<b>{parent_name}</b>", showarrow=False, align='center', font=dict(size=16, color='black')))
    
    return new_annotations


@callback(
    [
        Output('main-plot', 'figure', allow_duplicate=True),
        Output('isolation-mode-store', 'data', allow_duplicate=True),
        Output('isolate-podcast-button', 'children', allow_duplicate=True),
    ],
    Input('isolate-podcast-button', 'n_clicks'),
    [
        State('isolated-episode-store', 'data'),
        State('main-plot', 'relayoutData'),
        State('isolation-mode-store', 'data')
    ],
    prevent_initial_call=True
)
def toggle_episode_isolation(n_clicks, episode_to_isolate, relayout_data, is_in_isolation_mode):
    if not n_clicks:
        return no_update, no_update, no_update

    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()

    if is_in_isolation_mode:
        fig = create_scatter_figure(
            df,
            embeddings_2d,
            Dataset.get_topic_model()
        )
        new_button_text = "Isolate Episode"
        new_isolation_mode = False
        
        new_annotations = _calculate_annotations_for_view(relayout_data, df, embeddings_2d)
        fig.update_layout(annotations=new_annotations)
    
    else:
        if not episode_to_isolate:
            return no_update, no_update, no_update
        
        fig = create_scatter_figure(
            df,
            embeddings_2d,
            Dataset.get_topic_model(),
            isolated_episode_details=episode_to_isolate
        )
        new_button_text = "Stop Isolating"
        new_isolation_mode = True

    if relayout_data and 'xaxis.range[0]' in relayout_data:
        fig.update_layout(
            xaxis_range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']],
            yaxis_range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]
        )

    return fig, new_isolation_mode, new_button_text