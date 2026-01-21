from dash import callback, Output, Input, State, html, no_update
from src.widgets.selection_display import display_selected_points, display_single_point_details, extract_points_data
from src.callbacks.llm_summary import generate_summary_display, generate_loading_display
import time


@callback(
    Output('tab-content', 'children'),
    Output('side-panel-sub-1', 'className'),
    Output('side-panel-sub-1', 'children'),
    Output('summary-tab-activated', 'data'),
    Input('panel-tabs', 'active_tab'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def switch_tab(active_tab, points):
    
    if not points:
        return no_update, no_update, no_update, no_update

    num_points = len(points)
    
    if active_tab == "overview-tab":
        if num_points == 1:
            content = display_single_point_details(points[0])
        else:
            content = display_selected_points(points, expanded_index=None)
        return content, 'side-panel-sub-1', html.Div(), no_update

    elif active_tab == "summary-tab":
        selected_points_data = extract_points_data(points)
        
        loading_component = generate_loading_display(selected_points_data)
        
        tab_activation_timestamp = time.time()
        
        return loading_component, 'side-panel-sub-1', html.Div(), tab_activation_timestamp

    return html.Div(), 'side-panel-sub-1', html.Div(), no_update


@callback(
    Output('tab-content', 'children', allow_duplicate=True),
    Output('side-panel-sub-1', 'children', allow_duplicate=True),
    Input('summary-result-store', 'data'),
    State('panel-tabs', 'active_tab'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def update_summary_content(result_data, active_tab, points):
    """
    Updates the summary content when background processing is complete.
    """
    if not result_data or not result_data.get('completed') or active_tab != 'summary-tab':
        return no_update, no_update
    
    if not points:
        return no_update, no_update
    
    if 'error' in result_data:
        error_content = html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                html.Span(f"Error generating summary: {result_data['error']}")
            ], className="text-warning p-3")
        ])
        return error_content, no_update
    
    summary_component = result_data.get('summary')
    if summary_component:
        return summary_component, no_update
    
    return no_update, no_update