from dash import callback, Output, Input, State, no_update, callback_context, html

@callback(
    Output('panel-collapsed-store', 'data'),
    Input('hide-panel-button', 'n_clicks'),
    Input('show-panel-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_panel_collapse_state(hide_clicks, show_clicks):
    """Updates the store based on which toggle button was clicked."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'hide-panel-button':
        return True
    elif triggered_id == 'show-panel-button':
        return False
    
    return no_update

@callback(
    Output('selection-overlay', 'className'),
    Output('hide-panel-button', 'style'),
    Output('show-panel-button', 'style'),
    Input('selected-points-store', 'data'),
    Input('panel-collapsed-store', 'data')
)
def control_panel_visibility(selection_data, is_collapsed):
    """
    This is the master callback for panel visibility.
    It determines if the panel should be open or closed and which
    button (hide or show) should be visible next to the search bar.
    """
    has_selection = bool(selection_data)
    hide_style = {'display': 'none'}
    show_style = {'display': 'none'}
    overlay_class = 'selection-overlay'

    if has_selection:
        if is_collapsed:
            show_style = {'display': 'flex'}
        else:
            overlay_class = 'selection-overlay visible'
            hide_style = {'display': 'flex'}
    
    return overlay_class, hide_style, show_style