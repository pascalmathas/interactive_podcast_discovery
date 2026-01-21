from dash import callback, Output, Input, no_update, State, Patch, html
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
        Output('isolated-episode-store', 'data', allow_duplicate=True),
        Output('isolate-podcast-button-container', 'style', allow_duplicate=True),
        Output('focus-mode-store', 'data', allow_duplicate=True),
        Output('close-focus-button', 'style', allow_duplicate=True)
    ],
    Input('reset-view-button', 'n_clicks'),
    [
        State('chat-input', 'value'),
        State('isolated-episode-store', 'data')
    ],
    prevent_initial_call=True
)
def reset_plot_view(n_clicks, search_value, isolated_episode_details):
    """
    Resets the plot's zoom, pan, and semantic zoom level to the initial
    state, while preserving any active search filters or episode isolation styling.
    This does not affect the current selection or the app's isolation state.
    """
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    fig_with_reset_view = create_scatter_figure(
        Dataset.get(),
        Dataset.get_embeddings_2d(),
        Dataset.get_topic_model(),
        query=search_value,
        isolated_episode_details=isolated_episode_details
    )

    return fig_with_reset_view, no_update, no_update, False, {'display': 'none'}

@callback(
    [
        Output('main-plot', 'figure', allow_duplicate=True),
        Output('chat-input', 'value'),
        Output('selected-points-store', 'data', allow_duplicate=True),
        Output('panel-collapsed-store', 'data', allow_duplicate=True),
        Output('isolated-episode-store', 'data', allow_duplicate=True),
        Output('isolate-podcast-button-container', 'style', allow_duplicate=True),
        Output('isolation-mode-store', 'data', allow_duplicate=True),
        Output('conversation-store', 'data', allow_duplicate=True),
        Output('focus-mode-store', 'data', allow_duplicate=True),
        Output('close-focus-button', 'style', allow_duplicate=True),
        Output('deselect-button', 'style', allow_duplicate=True),
        Output('chat-history-output', 'children', allow_duplicate=True)
    ],
    Input('reset-app-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_full_app(n_clicks):
    """
    Resets the entire application to its initial state, as if the
    page was refreshed.
    """
    if not n_clicks:
        return (no_update,) * 12

    initial_fig = create_scatter_figure(
        Dataset.get(),
        Dataset.get_embeddings_2d(),
        Dataset.get_topic_model()
    )

    return initial_fig, "", [], False, None, {'display': 'none'}, False, [], False, {'display': 'none'}, {'display': 'none'}, html.Div()

@callback(
    [
        Output('focus-mode-store', 'data', allow_duplicate=True),
        Output('main-plot', 'figure', allow_duplicate=True),
        Output('close-focus-button', 'style', allow_duplicate=True)
    ],
    Input('close-focus-button', 'n_clicks'),
    [
        State('main-plot', 'figure'),
        State('main-plot', 'relayoutData'),
    ],
    prevent_initial_call=True
)
def close_focus_mode(n_clicks, current_fig_dict, relayout_data):
    """
    Closes the focus mode by resetting the plot to its normal state
    and hiding the close focus button.
    """
    if not n_clicks:
        return no_update, no_update, no_update

    patched_figure = Patch()
    
    clean_fig = create_scatter_figure(
        Dataset.get(),
        Dataset.get_embeddings_2d(),
        Dataset.get_topic_model()
    )
    clean_fig_dict = clean_fig.to_dict()
    
    for i, trace in enumerate(current_fig_dict['data']):
        if i < len(clean_fig_dict['data']) and 'marker' in trace and 'marker' in clean_fig_dict['data'][i]:
             if 'color' in clean_fig_dict['data'][i]['marker']:
                 patched_figure['data'][i]['marker']['color'] = clean_fig_dict['data'][i]['marker']['color']
             if 'size' in clean_fig_dict['data'][i]['marker']:
                 patched_figure['data'][i]['marker']['size'] = clean_fig_dict['data'][i]['marker']['size']
    
    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    new_annotations = _calculate_annotations_for_view(relayout_data, df, embeddings_2d)
    patched_figure['layout']['annotations'] = new_annotations

    return False, patched_figure, {'display': 'none'}