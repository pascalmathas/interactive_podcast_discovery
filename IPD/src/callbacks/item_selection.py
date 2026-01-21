from dash import callback, Output, Input, State, callback_context, no_update, html
import dash_bootstrap_components as dbc
from src.widgets.selection_display import (
    display_single_point_details, 
    display_selected_points
)
from src.story_generator import StoryGenerator
from src.Dataset import Dataset

story_generator = StoryGenerator()

@callback(
    Output('tabs-container', 'children'),
    Output('tab-content', 'children', allow_duplicate=True),
    Output('selected-points-store', 'data'),
    Output('panel-collapsed-store', 'data', allow_duplicate=True),
    Output('isolate-podcast-button-container', 'style'),
    Output('isolate-podcast-button', 'children'),
    Output('isolated-episode-store', 'data'),
    Output('isolation-mode-store', 'data', allow_duplicate=True),
    Output('deselect-button', 'style'),
    Input('main-plot', 'selectedData'),
    Input('main-plot', 'clickData'),
    prevent_initial_call=True
)
def setup_panel_on_selection(selected_data, click_data):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 9
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[1]
    
    points_from_plot = []
    if triggered_id == 'selectedData' and selected_data and 'points' in selected_data:
        points_from_plot = selected_data['points']
    elif triggered_id == 'clickData' and click_data and 'points' in click_data:
        points_from_plot = click_data['points']
    
    isolate_button_style = {'display': 'none'}
    isolate_button_text = "Isolate Episode"
    isolated_episode_to_store = None
    deselect_button_style = {'display': 'none'}
    
    if not points_from_plot:
        placeholder_tabs = dbc.Tabs([
            dbc.Tab(label="Select points to see details", tab_id="placeholder-tab", disabled=True)
        ], id="panel-tabs", active_tab="placeholder-tab", className="nav-pills", style={'display': 'none'})
        
        return placeholder_tabs, html.Div(), [], False, isolate_button_style, isolate_button_text, None, False, deselect_button_style
    
    deselect_button_style = {'display': 'inline-block'}
    
    num_points = len(points_from_plot)
    df = Dataset.get()
    
    points_to_store = []

    if num_points == 1:
        custom_data = points_from_plot[0].get('customdata')
        idx = int(custom_data[-1])
        row = df.iloc[idx]
        point_data = {key: row.get(key) for key in row.keys()}
        point_data['customdata'] = idx
        points_to_store = [point_data]
        initial_content = display_single_point_details(point_data)
        
        episode_title = row.get("episode_title", "Episode")
        podcast_title = row.get("podcast_title")
        
        isolate_button_style = {'display': 'inline-block'}
        
        isolated_episode_to_store = {
            'podcast_title': podcast_title,
            'episode_title': episode_title
        }
    else:
        for i, point in enumerate(points_from_plot):
            if i >= 25:
                break
            
            custom_data = point.get('customdata')
            if custom_data is None:
                continue
            idx = int(custom_data[-1])
            
            row = df.iloc[idx]
            summary = row.get('summary', 'Summary not available.')

            point_data = {key: row.get(key) for key in row.keys()}
            point_data['customdata'] = idx
            point_data['summary'] = summary
            points_to_store.append(point_data)

        if num_points > 25:
            for point in points_from_plot[25:]:
                custom_data = point.get('customdata')
                if custom_data is None:
                    continue
                idx = int(custom_data[-1])
                row = df.iloc[idx]
                points_to_store.append({
                    'customdata': idx,
                    'episode_title': row.get('episode_title', 'N/A'),
                    'podcast_title': row.get('podcast_title', 'N/A'),
                })

        initial_content = display_selected_points(points_to_store, expanded_index=None)

    overview_label = "Description" if num_points == 1 else "Overview"
    summary_label = "Link Summary" if num_points > 1 else "Summary"

    tabs = dbc.Tabs([
        dbc.Tab(label=overview_label, tab_id="overview-tab"),
        dbc.Tab(label=summary_label, tab_id="summary-tab"),
    ], id="panel-tabs", active_tab="overview-tab", className="nav-pills")

    return tabs, initial_content, points_to_store, False, isolate_button_style, isolate_button_text, isolated_episode_to_store, False, deselect_button_style