import sys
from dash import Dash, html, dcc
from src import config
from src.Dataset import Dataset
import dash_bootstrap_components as dbc

from src.widgets.scatterplot_explorer import create_scatterplot_explorer
from src.widgets.llm_chat import create_llm_chat
from src.widgets.panel_tabs import create_panel_tabs
from src.topic_labeler import GPT3TopicLabeler
from src.widgets.focus import create_focus_box
from src.callbacks import item_selection
from src.callbacks import panel_tabs
from src.callbacks import expand_card
from src.callbacks import llm_chats
from src.callbacks import llm_summary
from src.callbacks import reset
from src.callbacks import semantic_zoom
from src.callbacks import panel_visibility
from src.callbacks import isolate_episode
from src.callbacks import deselect
from src.callbacks import semantic_hover

from src.data_pipeline.embedding.generate_embeddings import run_bertopic_pipeline
import os

def run_ui():
    external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
    app = Dash(__name__, external_stylesheets=external_stylesheets,
               suppress_callback_exceptions=True)
    
    reset_buttons_widget = html.Div([
        dbc.Button("Reset View", id="reset-view-button", color="secondary", className="me-2"),
        dbc.Button("Reset App", id="reset-app-button", color="danger"),
        dbc.Button("Close Focus", id="close-focus-button", color="warning", className="ms-2", style={'display': 'none'}),
        dbc.Button("Deselect", id="deselect-button", color="info", className="ms-2", style={'display': 'none'}),
        html.Div(
            dbc.Button("Isolate Episode", id="isolate-podcast-button", color="info", className="ms-3"),
            id="isolate-podcast-button-container",
            style={'display': 'none'}
        )
    ], className="reset-buttons-container")

    llm_chat_widget = create_llm_chat()
    panel_tabs_widget = create_panel_tabs()
    scatterplot_widget = create_scatterplot_explorer()
    focus_box_widget = create_focus_box()
    
    top_bar = html.Div([
        llm_chat_widget,
        dbc.Button(html.I(className="fas fa-chevron-left"), id="hide-panel-button", color="light", className="hide-panel-button", style={'display': 'none'}),
        dbc.Button(html.I(className="fas fa-chevron-right"), id="show-panel-button", color="light", className="show-panel-button", style={'display': 'none'}),
    ], className="top-bar")


    app.layout = html.Div([

        dcc.Store(id='selected-points-store', data=[]),
        dcc.Store(id='panel-collapsed-store', data=False),
        dcc.Store(id='isolated-episode-store', data=None),
        dcc.Store(id='isolation-mode-store', data=False),
        dcc.Store(id='focus-mode-store', data=False),
        
        # Background summary processing stores
        dcc.Store(id='summary-tab-activated', data=None),
        dcc.Store(id='summary-processing-trigger', data=None),
        dcc.Store(id='summary-result-store', data=None),
        
        dcc.Interval(
            id='summary-check-interval',
            interval=1000,  # Check every 1 second
            n_intervals=0,
            disabled=False
        ),
        
        reset_buttons_widget,

        top_bar,

        panel_tabs_widget,

        scatterplot_widget,
        
        focus_box_widget,

    ], className='app-container')

    app.run(debug=True, use_reloader=True)

def main():
    print("\nRunning BERTopic pipeline...")
    
    umap_params = {
        'n_neighbors': 20,
        'n_components': 10,
        'min_dist': 0.1,
        'metric': 'cosine',
        'random_state': 42
    }
    
    hdbscan_params = {
        'min_cluster_size': 10,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'prediction_data': True
    }
    
    bertopic_params = {
        'top_n_words': 10,
        'verbose': True,
    }
    
    from src.config import DATASET_SEGMENTED_DIR, BERTOPIC_DIR
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    run_bertopic_pipeline(
        data_path=DATASET_SEGMENTED_DIR,
        cache_path=BERTOPIC_DIR,
        openai_api_key=openai_api_key,
        embedding_model_name='all-MiniLM-L12-v2',
        force_retrain=False,  # set to True to force retraining, e.g. trying out different parameters
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        bertopic_params=bertopic_params,
        seed=42
    )
    print("\nBERTopic pipeline complete.")
    
    if not Dataset.files_exist():
        print('Missing required files for dataset:')
        print(f'Podcast segments: {config.PODCAST_SEGMENTS_PATH}')
        print(f'Embeddings: {config.EMBEDDINGS_PATH}')
        print(f'BERTopic model: {config.BERTOPIC_MODEL_PATH}')
        sys.exit(1)
    else:
        Dataset.load()
        print('Starting Dash')
        run_ui()

if __name__ == '__main__':
    main()