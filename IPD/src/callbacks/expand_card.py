from dash import callback, Output, Input, State, callback_context, no_update, ALL
from src.widgets.selection_display import display_selected_points
import json


@callback(
    Output('tab-content', 'children', allow_duplicate=True),
    Input({'type': 'podcast-card', 'index': ALL}, 'n_clicks'),
    State('selected-points-store', 'data'),
    State({'type': 'podcast-card-collapse', 'index': ALL}, 'is_open'),
    prevent_initial_call=True
)
def toggle_card_expansion(n_clicks, points_in_store, collapse_states):
    """
    Toggles the expansion of a clicked card in the overview.
    If a collapsed card is clicked, it expands.
    If an expanded card is clicked, it collapses.
    """
    ctx = callback_context
    if not ctx.triggered_id or not any(n_clicks):
        return no_update
    
    # Using ctx.triggered_id is the robust way for pattern-matching callbacks
    if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "podcast-card":
        clicked_index = ctx.triggered_id['index']
    else:
        return no_update
    
    currently_expanded_index = None
    if collapse_states:
        try:
            currently_expanded_index = collapse_states.index(True)
        except ValueError:
            currently_expanded_index = None

    if clicked_index == currently_expanded_index:
        new_expanded_index = None
    else:
        new_expanded_index = clicked_index
        
    if points_in_store:
        return display_selected_points(points_in_store, expanded_index=new_expanded_index)
    
    return no_update