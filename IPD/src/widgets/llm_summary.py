
from dash import html


def summary_display(story_text, title="Generated Content"):
    
    if not story_text:
        return html.Div(
            "No story available", 
            className='no-points-message'
        )
    
    return html.Div([
        html.H5(title, className='mb-3'),
        html.Div([
            html.P(story_text, className='story-wrapper')
        ], className='story-container')
    ])