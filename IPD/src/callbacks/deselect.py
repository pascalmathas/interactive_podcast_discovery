from dash import callback, Output, Input, no_update, Patch, State, html

@callback(
    [
        Output('selected-points-store', 'data', allow_duplicate=True),
        Output('panel-collapsed-store', 'data', allow_duplicate=True),
        Output('deselect-button', 'style', allow_duplicate=True),
        Output('main-plot', 'figure', allow_duplicate=True),
        Output('isolate-podcast-button-container', 'style', allow_duplicate=True),
        Output('conversation-store', 'data', allow_duplicate=True),
        Output('chat-history-output', 'children', allow_duplicate=True)
    ],
    Input('deselect-button', 'n_clicks'),
    State('main-plot', 'figure'),
    prevent_initial_call=True
)
def deselect_points(n_clicks, current_figure):
    """
    Clears the current selection, closes the side panel, resets the chat, and
    removes all selection visuals from the plot, without affecting the plot view.
    """
    if not n_clicks or not current_figure:
        return (no_update,) * 7

    patched_figure = Patch()
    
    patched_figure['layout']['selections'] = []
    
    for i in range(len(current_figure['data'])):
        patched_figure['data'][i]['selectedpoints'] = None

    return (
        [],
        True,
        {'display': 'none'},
        patched_figure,
        {'display': 'none'},
        [],
        None
    )