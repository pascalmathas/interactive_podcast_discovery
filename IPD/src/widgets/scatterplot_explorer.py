import plotly.graph_objects as go
import numpy as np
from dash import dcc, html
import plotly.express as px

from src.widgets.selection_display import PODCAST_NAME_MAP

SCATTER_LAYOUT = dict(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    hovermode='closest',
    margin=dict(l=0, r=0, t=30, b=0)
)

def _get_spatially_distributed_parent_annotations(df_to_plot, embeddings_to_plot):
    """
    Calculates positions for parent topic annotations based on the center of mass of their
    constituent points, sizes them by topic prevalence, and filters them to avoid overlaps.
    """
    if 'parent_topic_name' not in df_to_plot.columns or df_to_plot.empty:
        return []

    # Identify all unique parent topics present in the current view
    unique_parent_names = df_to_plot['parent_topic_name'].dropna().unique()
    
    potential_annotations = []
    for parent_name in unique_parent_names:
        # Create a mask for all points belonging to the current parent topic
        parent_mask = df_to_plot['parent_topic_name'] == parent_name
        parent_embeddings = embeddings_to_plot[parent_mask]
        
        if parent_embeddings.shape[0] > 0:
            # Calculate the center of mass for this parent topic for accurate positioning
            center_of_mass = np.mean(parent_embeddings, axis=0)
            num_points = parent_embeddings.shape[0]

            potential_annotations.append({
                'x': center_of_mass[0],
                'y': center_of_mass[1],
                'text': f"<b>{parent_name}</b>",
                'showarrow': False, 'align': 'center',
                # Font size will be added dynamically below
                'num_points': num_points
            })

    if not potential_annotations:
        return []

    # Dynamically size annotations based on the number of points in the topic
    all_num_points = np.array([p['num_points'] for p in potential_annotations])
    # Using log scale for better visual differentiation between topic sizes
    log_points = np.log1p(all_num_points)
    min_log, max_log = np.min(log_points), np.max(log_points)
    
    min_font, max_font = 14/2, 32/2

    for anno in potential_annotations:
        log_num = np.log1p(anno['num_points'])
        if max_log == min_log:
            scaled_size = (min_font + max_font) / 2
        else:
            scale = (log_num - min_log) / (max_log - min_log)
            scaled_size = min_font + scale * (max_font - min_font)
        
        anno['font'] = {'size': scaled_size, 'color': 'black'}

    # Spatially distribute the annotations to avoid clutter

    # Sort by number of points to prioritize showing larger topics
    potential_annotations.sort(key=lambda p: p.pop('num_points'), reverse=True)

    # Determine a dynamic distance threshold based on the plot's scale
    x_span = np.max(embeddings_to_plot[:, 0]) - np.min(embeddings_to_plot[:, 0])
    y_span = np.max(embeddings_to_plot[:, 1]) - np.min(embeddings_to_plot[:, 1])
    # Set threshold to 13% of the smaller dimension of the visible plot area.
    # A smaller factor (e.g., 0.13) allows more, closer labels. A larger factor (e.g., 0.2) shows fewer, more spread-out labels.
    min_dist_sq = (min(x_span, y_span) * 0.13)**2
    
    final_annotations = []
    if potential_annotations:
        final_annotations.append(potential_annotations[0])  # Always include the largest topic
    
    for annotation in potential_annotations[1:]:
        is_far_enough = True
        for final_annotation in final_annotations:
            dist_sq = (annotation['x'] - final_annotation['x'])**2 + (annotation['y'] - final_annotation['y'])**2
            if dist_sq < min_dist_sq:
                is_far_enough = False
                break
        if is_far_enough:
            final_annotations.append(annotation)

    return final_annotations


def create_scatter_figure(df_docs, embeddings_2d, topic_model, query=None, isolated_episode_details=None):
    fig = go.Figure()

    if len(embeddings_2d) > 0:
        center = np.mean(embeddings_2d, axis=0)
        distances = np.linalg.norm(embeddings_2d - center, axis=1)
        threshold = np.mean(distances) + 3 * np.std(distances)
        non_outlier_mask = distances < threshold
        df_docs = df_docs[non_outlier_mask].copy()
        embeddings_2d = embeddings_2d[non_outlier_mask]

    all_topic_ids = sorted([t for t in df_docs['topic'].unique() if t != -1])
    centroids = {tid: np.mean(embeddings_2d[df_docs['topic'] == tid], axis=0) for tid in all_topic_ids if (df_docs['topic'] == tid).any()}
    
    core_topic_ids = list(centroids.keys())
    if len(centroids) > 5:
        centroid_points = np.array(list(centroids.values()))
        overall_center = np.mean(centroid_points, axis=0)
        distances = np.linalg.norm(centroid_points - overall_center, axis=1)
        distance_threshold = np.mean(distances) + 1.5 * np.std(distances)
        core_topic_ids = [tid for i, tid in enumerate(core_topic_ids) if distances[i] <= distance_threshold]

    topics_to_plot = core_topic_ids
    if query:
        from src.Dataset import Dataset
        search_engine = Dataset.get_search_engine()
        relevant_topics_from_search = search_engine.filter_topics_by_relevance(query)
        topics_to_plot = list(set(core_topic_ids) & set(relevant_topics_from_search))
        if not topics_to_plot: return create_empty_figure()

    final_plot_mask = df_docs['topic'].isin(topics_to_plot)
    df_to_plot = df_docs[final_plot_mask].copy()
    embeddings_to_plot = embeddings_2d[final_plot_mask]

    if df_to_plot.empty: return create_empty_figure()

    unique_topics_in_df = sorted(df_docs['topic'].unique())
    colors = px.colors.qualitative.Plotly
    color_map = {topic_id: colors[i % len(colors)] for i, topic_id in enumerate(unique_topics_in_df)}
    
    if isolated_episode_details:
        is_isolated_episode_mask = (df_to_plot['podcast_title'] == isolated_episode_details['podcast_title']) & \
                                   (df_to_plot['episode_title'] == isolated_episode_details['episode_title'])
        
        default_colors = df_to_plot['topic'].map(color_map).fillna('grey')
        
        df_to_plot['point_color'] = np.where(is_isolated_episode_mask, default_colors, 'rgba(200, 200, 200, 0.3)')
        
        isolated_points = df_to_plot[is_isolated_episode_mask]
        isolated_embeddings = embeddings_to_plot[is_isolated_episode_mask]
        
        if len(isolated_points) > 1:
            sort_indices = np.argsort(isolated_points.index)
            sorted_embeddings = isolated_embeddings[sort_indices]
            
            fig.add_trace(go.Scattergl(
                x=sorted_embeddings[:, 0],
                y=sorted_embeddings[:, 1],
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.5)', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
                name='Episode Connection'
            ))
    else:
        df_to_plot['point_color'] = df_to_plot['topic'].map(color_map).fillna('grey')

    for podcast_name in sorted(df_to_plot['podcast_title'].unique()):
        display_name = PODCAST_NAME_MAP.get(podcast_name, str(podcast_name).replace("_", " ").title())
        podcast_mask = df_to_plot['podcast_title'] == podcast_name
        plot_df_podcast = df_to_plot[podcast_mask]
        
        point_colors = plot_df_podcast['point_color']

        custom_data_for_hover = np.stack((
            plot_df_podcast['podcast_title'],
            plot_df_podcast['parent_topic_name'] if 'parent_topic_name' in plot_df_podcast.columns else ['N/A'] * len(plot_df_podcast),
            plot_df_podcast['topic_name'],
            plot_df_podcast['title'],
            plot_df_podcast.index.to_series().astype(str)
        ), axis=-1)
        
        fig.add_trace(go.Scattergl(
            x=embeddings_to_plot[podcast_mask, 0],
            y=embeddings_to_plot[podcast_mask, 1],
            mode='markers',
            legendgroup=podcast_name,
            showlegend=False,
            name=display_name,
            customdata=custom_data_for_hover,
            hoverinfo='none',
            marker=dict(
                size=7,
                color=point_colors,
                symbol='circle',
                line=dict(width=0, color='rgba(0,0,0,0)')  # Initialize for hover modifications
            ),
            text=plot_df_podcast['title']
        ))

        fig.add_trace(go.Scattergl(
            x=[None], y=[None],
            mode='markers',
            name=display_name,
            legendgroup=podcast_name,
            showlegend=True,
            marker=dict(color='black', symbol='circle')
        ))

    parent_annotations = _get_spatially_distributed_parent_annotations(df_to_plot, embeddings_to_plot)

    final_layout = SCATTER_LAYOUT.copy()
    x_min, x_max = np.min(embeddings_to_plot[:, 0]), np.max(embeddings_to_plot[:, 0])
    y_min, y_max = np.min(embeddings_to_plot[:, 1]), np.max(embeddings_to_plot[:, 1])
    x_padding, y_padding = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    final_layout['xaxis']['range'] = [x_min - x_padding, x_max + x_padding]
    final_layout['yaxis']['range'] = [y_min - y_padding, y_max + y_padding]
    final_layout['annotations'] = parent_annotations
    final_layout['showlegend'] = False

    fig.update_layout(**final_layout)
    
    fig.update_yaxes(scaleanchor="x", scaleratio=0.5)
    
    return fig

def create_scatterplot_explorer():
    from src.Dataset import Dataset
    return html.Div([
        dcc.Store(id='hover-throttle-store', data={'last_topic': None, 'last_time': 0}),
        dcc.Graph(
            id='main-plot',
            figure=create_scatter_figure(
                Dataset.get(),
                Dataset.get_embeddings_2d(),
                Dataset.get_topic_model()
            ),
            className='main-plot',
            clear_on_unhover=True,
            config={'modeBarButtonsToAdd': ['drawrect', 'eraseshape']}
        )
    ])


def create_empty_figure():
    fig = go.Figure()
    fig.update_layout(**SCATTER_LAYOUT, annotations=[dict(text="No results found.", showarrow=False)])
    fig.update_yaxes(scaleanchor="x", scaleratio=0.5)
    return fig