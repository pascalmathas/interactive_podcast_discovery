import numpy as np
from dash import callback, Output, Input, State, no_update, Patch
from src.Dataset import Dataset

def get_indices_from_customdata(customdata):
    """Extract indices from customdata, supporting both 1D and 2D formats"""
    customdata_array = np.array(customdata)
    if customdata_array.ndim == 1:
        return customdata_array  # dev_pascal format (1D array of indices)
    else:
        return customdata_array[:, -1].astype(int)  # dev_joppe format (2D array, last column is indices)

DATAPOINT_THRESHOLD = 2500
GLYPH_THRESHOLD = 650

PODCAST_SYMBOLS = {
    "Machine_Learning_Street_Talk_MLST_2025-06-20-2023-02-22": "square",
    "No_Priors_Artificial_Intelligence_Technology_Startups": "triangle-up",
    "Gradient_Dissent_Conversations_on_AI": "diamond",
    "Lex Fridman": "triangle-left",
    "Felix McLean": "triangle-right",
    "The_TWIML_AI_Podcast_formerly_This_Week_in_Machine_Learning_Artificial_Intelligence": "star",
    "Practical_AI": "hexagon",
    "NVIDIA_AI_Podcast": "bowtie",
    "This_Day_in_AI_Podcast": "hourglass",
    "The_Gradient_Perspectives_on_AI": "cross",
    "Brain_Inspired": "x",
}
DEFAULT_SYMBOL = "pentagon"

DEFAULT_MARKER_SIZE = 7
ISOLATED_MARKER_SIZE = 14
FADED_MARKER_SIZE = 4

MIN_GLYPH_SIZE = 9
MAX_GLYPH_SIZE = 33

@callback(
    Output('main-plot', 'figure', allow_duplicate=True),
    Input('main-plot', 'relayoutData'),
    [
        State('main-plot', 'figure'),
        State('isolated-episode-store', 'data'),
        State('isolation-mode-store', 'data'),
        State('focus-mode-store', 'data')
    ],
    prevent_initial_call=True
)
def semantic_zoom(relayout_data, fig, isolated_episode_details, is_isolation_mode, is_focus_mode):
    if is_focus_mode:
        return no_update

    if not relayout_data or 'xaxis.range[0]' not in relayout_data:
        return no_update

    df = Dataset.get()
    embeddings_2d = Dataset.get_embeddings_2d()
    
    # Check if data is available
    if df is None or embeddings_2d is None:
        return no_update

    x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
    y_range = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]

    visible_mask = (
        (embeddings_2d[:, 0] >= x_range[0]) & (embeddings_2d[:, 0] <= x_range[1]) &
        (embeddings_2d[:, 1] >= y_range[0]) & (embeddings_2d[:, 1] <= y_range[1])
    )
    visible_points_count = np.sum(visible_mask)
    
    patched_figure = Patch()
    visible_df = df[visible_mask]
    is_fading_active = is_isolation_mode and isolated_episode_details

    if is_fading_active:
        print(f"Points in view: {visible_points_count}. ZOOM LEVEL: ANY (ISOLATION MODE)")
        
        new_annotations = []
        isolated_visible_mask = (visible_df['podcast_title'] == isolated_episode_details['podcast_title']) & \
                                (visible_df['episode_title'] == isolated_episode_details['episode_title'])
        
        if np.any(isolated_visible_mask):
            visible_embeddings_2d = embeddings_2d[visible_mask]
            isolated_visible_embeddings_2d = visible_embeddings_2d[isolated_visible_mask]
            isolated_visible_df = visible_df[isolated_visible_mask]
            
            for topic_id in isolated_visible_df['topic'].unique():
                if topic_id == -1: continue
                
                topic_mask_in_isolated_view = (isolated_visible_df['topic'] == topic_id)
                if topic_mask_in_isolated_view.any():
                    centroid = np.mean(isolated_visible_embeddings_2d[topic_mask_in_isolated_view], axis=0)
                    label = isolated_visible_df[topic_mask_in_isolated_view]['topic_name'].iloc[0]
                    new_annotations.append(dict(
                        x=centroid[0], 
                        y=centroid[1], 
                        text=f"<b>{label}</b>", 
                        showarrow=False, 
                        font=dict(size=14, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        borderpad=4
                    ))
        patched_figure['layout']['annotations'] = new_annotations
        
        for i, trace in enumerate(fig['data']):
            patched_figure['data'][i]['marker']['symbol'] = 'circle'
            if trace.get('customdata'):
                indices = get_indices_from_customdata(trace['customdata'])
                # Filter valid indices to prevent IndexError
                valid_indices = [idx for idx in indices if idx in df.index]
                if not valid_indices:
                    continue
                trace_df = df.loc[valid_indices]
                is_isolated_mask = (trace_df['podcast_title'] == isolated_episode_details['podcast_title']) & \
                                   (trace_df['episode_title'] == isolated_episode_details['episode_title'])
                final_sizes = np.where(is_isolated_mask, ISOLATED_MARKER_SIZE, FADED_MARKER_SIZE)
                patched_figure['data'][i]['marker']['size'] = final_sizes.tolist()
        
        patched_figure['layout']['showlegend'] = False
        
    else:
        new_annotations = []
        
        if visible_points_count < GLYPH_THRESHOLD:
            print(f"Points in view: {visible_points_count}. ZOOM LEVEL: HIGH (Glyphs)")
            min_len, max_len = visible_df['character_length'].min(), visible_df['character_length'].max()
            range_len = max_len - min_len if max_len > min_len else 1

            for i, trace in enumerate(fig['data']):
                raw_podcast_name = trace.get('legendgroup')
                if not raw_podcast_name: continue
                patched_figure['data'][i]['marker']['symbol'] = PODCAST_SYMBOLS.get(raw_podcast_name, DEFAULT_SYMBOL)
                if trace.get('customdata'):
                    indices = get_indices_from_customdata(trace['customdata'])
                    # Filter valid indices to prevent IndexError
                    valid_indices = [idx for idx in indices if idx in df.index]
                    if not valid_indices:
                        continue
                    trace_df = df.loc[valid_indices]
                    scaled_sizes = MIN_GLYPH_SIZE + ((trace_df['character_length'] - min_len) / range_len * (MAX_GLYPH_SIZE - MIN_GLYPH_SIZE))
                    patched_figure['data'][i]['marker']['size'] = scaled_sizes.tolist()
            
            patched_figure['layout']['showlegend'] = True
            patched_figure['layout']['legend'] = dict(
                yanchor="top", y=0.9, xanchor="right", x=0.99, font=dict(size=14),
                title=dict(text='<b>Podcasts</b><br><sup>Size indicates segment length</sup>', font=dict(size=12)),
                bgcolor='rgba(255, 255, 255, 1)', bordercolor='Black', borderwidth=1
            )
        else:
            print(f"Points in view: {visible_points_count}. ZOOM LEVEL: LOW/MEDIUM (Circles)")
            for i, trace in enumerate(fig['data']):
                patched_figure['data'][i]['marker']['symbol'] = 'circle'
                patched_figure['data'][i]['marker']['size'] = DEFAULT_MARKER_SIZE
            patched_figure['layout']['showlegend'] = False
            
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
        
        patched_figure['layout']['annotations'] = new_annotations
        
    return patched_figure