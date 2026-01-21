import dash_bootstrap_components as dbc
from dash import dcc, html

def create_llm_chat():
    return html.Div([
        dbc.InputGroup([
            dcc.Input(
                id='chat-input',
                placeholder='Ask about podcast content...',
                debounce=True,
                className='form-control'
            ),
            dbc.Button('Send', id='send-button', color='primary')
        ]),
        
        html.Div(id='chat-history-container', className='chat-history-container', style={'display': 'none'}),
        dcc.Store(id='conversation-store', data=[])
    ], className='search-wrapper')