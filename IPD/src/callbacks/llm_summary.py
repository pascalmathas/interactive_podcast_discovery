from dash import callback, Output, Input, State, html, dcc, no_update
from src.story_generator import StoryGenerator
from src.widgets.llm_summary import summary_display
import threading
import time
import uuid

story_generator = StoryGenerator()

# Global dictionary to track processing jobs
processing_jobs = {}


def generate_summary_display(selected_points_data):
    """
    Generates a summary or linked story based on the number of selected points.
    """
    if not selected_points_data:
        return summary_display(None, title="No Content")
    
    num_points = len(selected_points_data)
    title = "Linked Story" if num_points > 1 else "Summary"
    
    story = story_generator.generate_content(selected_points_data)

    return summary_display(story, title=title)


def generate_loading_display(selected_points_data):
    """
    Generates a loading display while summary is being processed.
    """
    if not selected_points_data:
        return summary_display(None, title="No Content")
    
    num_points = len(selected_points_data)
    title = "Linked Story" if num_points > 1 else "Summary"
    
    loading_content = html.Div([
        dcc.Loading(
            children=[
                html.Div([
                    html.I(className="fas fa-robot me-2"),
                    html.Span("Generating summary with AI..." if num_points == 1 else "Weaving story from selected segments...")
                ], className="text-muted d-flex align-items-center justify-content-center p-4")
            ],
            type="default",
            style={"height": "100px"}
        )
    ])
    
    return summary_display(loading_content, title=title)


def process_summary_in_background(job_id, selected_points_data):
    """
    Function to run summary generation in a separate thread.
    """
    try:
        # Generate summary (this is the slow LLM call)
        summary_component = generate_summary_display(selected_points_data)
        
        # Store the result
        processing_jobs[job_id] = {
            'status': 'completed',
            'result': summary_component,
            'timestamp': time.time()
        }
    except Exception as e:
        processing_jobs[job_id] = {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }


@callback(
    Output('summary-processing-trigger', 'data'),
    Input('summary-tab-activated', 'data'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def trigger_background_summary(tab_activated, points):
    """
    Triggers background summary generation when summary tab is activated.
    """
    if not tab_activated or not points:
        return no_update
    
    # Generate unique ID for this processing request
    job_id = str(uuid.uuid4())
    
    # Extract points data for processing
    from src.widgets.selection_display import extract_points_data
    selected_points_data = extract_points_data(points)
    
    # Mark job as processing
    processing_jobs[job_id] = {
        'status': 'processing',
        'timestamp': time.time()
    }
    
    # Start background thread
    thread = threading.Thread(
        target=process_summary_in_background,
        args=(job_id, selected_points_data)
    )
    thread.daemon = True
    thread.start()
    
    return {
        'job_id': job_id,
        'timestamp': tab_activated
    }


@callback(
    Output('summary-result-store', 'data'),
    Input('summary-check-interval', 'n_intervals'),
    State('summary-processing-trigger', 'data'),
    prevent_initial_call=True
)
def check_summary_completion(n_intervals, trigger_data):
    """
    Periodically checks if background summary generation is complete.
    """
    if not trigger_data:
        return no_update
    
    job_id = trigger_data.get('job_id')
    if not job_id or job_id not in processing_jobs:
        return no_update
    
    job = processing_jobs[job_id]
    
    if job['status'] == 'completed':
        # Clean up completed job
        result = job['result']
        del processing_jobs[job_id]
        
        return {
            'job_id': job_id,
            'summary': result,
            'completed': True
        }
    elif job['status'] == 'error':
        # Clean up failed job
        error = job.get('error', 'Unknown error')
        del processing_jobs[job_id]
        
        return {
            'job_id': job_id,
            'error': error,
            'completed': True
        }
    
    # Still processing
    return no_update