import dash_bootstrap_components as dbc
from dash import html

def create_panel_tabs():
    
    initial_tabs = dbc.Tabs([
        dbc.Tab(label="Select points to see details", tab_id="placeholder-tab", disabled=True)
    ], id="panel-tabs", active_tab="placeholder-tab", className="nav-pills", style={'display': 'none'})
    
    panel_content = html.Div([
        
        html.Div(id='chat-history-output', className='mb-2'),
        html.Div([
            html.Div([initial_tabs], id='tabs-container'),
            html.Div(id='tab-content', className='tab-content')
        ], id='selection-panel-content'),
        
        html.Div(id='chat-panel-content', style={'display': 'none'}) 
    ])
    
    return html.Div([
        html.Div([
            panel_content
        ], className='side-panel-main'),
        html.Div(id='side-panel-sub-1',
                 className='side-panel-sub-1')
    ], id='selection-overlay', className='selection-overlay')