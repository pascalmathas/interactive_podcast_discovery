import dash_bootstrap_components as dbc
from dash import html

PODCAST_NAME_MAP = {
    "Machine_Learning_Street_Talk_MLST_2025-06-20-2023-02-22": "MLST",
    "No_Priors_Artificial_Intelligence_Technology_Startups": "No Priors",
    "Gradient_Dissent_Conversations_on_AI": "Gradient Dissent",
    "Lex Fridman": "Lex Fridman Podcast",
    "Felix McLean": "Felix McLean",
    "The_TWIML_AI_Podcast_formerly_This_Week_in_Machine_Learning_Artificial_Intelligence": "This Week in Machine Learning",
    "Practical_AI": "Practical AI",
    "NVIDIA_AI_Podcast": "NVIDIA AI Podcast",
    "This_Day_in_AI_Podcast": "This Day in AI",
    "The_Gradient_Perspectives_on_AI": "The Gradient: Perspectives on AI",
    "Brain_Inspired": "Brain Inspired",
}

HOST_MAP = {
    "Machine_Learning_Street_Talk_MLST_2025-06-20-2023-02-22": "Thomas Dohmatob & Yannic Kilcher",
    "No_Priors_Artificial_Intelligence_Technology_Startups": "Sarah Guo & Elad Gil",
    "Gradient_Dissent_Conversations_on_AI": "Lukas Biewald",
    "Lex Fridman": "Lex Fridman",
    "Felix McLean": "Felix McLean",
    "The_TWIML_AI_Podcast_formerly_This_Week_in_Machine_Learning_Artificial_Intelligence": "Sam Charrington",
    "Practical_AI": "Chris Benson & Daniel Whitenack",
    "NVIDIA_AI_Podcast": "Noah Kravitz",
    "This_Day_in_AI_Podcast": "Jack Clark & Zack Kass",
    "The_Gradient_Perspectives_on_AI": "Daniel Bashir",
    "Brain_Inspired": "Paul Middlebrooks",
}

PODCAST_ICON_MAP = {
    "Machine_Learning_Street_Talk_MLST_2025-06-20-2023-02-22": "/assets/podcast_icons/mlst.jpeg",
    "No_Priors_Artificial_Intelligence_Technology_Startups": "/assets/podcast_icons/no_priors.avif",
    "Gradient_Dissent_Conversations_on_AI": "/assets/podcast_icons/gradient_dissent.jpeg",
    "Lex Fridman": "/assets/podcast_icons/fridman_icon.jpeg",
    "Felix McLean": "/assets/podcast_icons/fridman_icon.jpeg",
    "The_TWIML_AI_Podcast_formerly_This_Week_in_Machine_Learning_Artificial_Intelligence": "/assets/podcast_icons/tiwml.jpg",
    "Practical_AI": "/assets/podcast_icons/practical_ai.jpeg",
    "NVIDIA_AI_Podcast": "/assets/podcast_icons/nvidia.jpeg",
    "This_Day_in_AI_Podcast": "/assets/podcast_icons/this_day_in_ai.jpg",
    "The_Gradient_Perspectives_on_AI": "/assets/podcast_icons/the_gradient.jpeg",
    "Brain_Inspired": "/assets/podcast_icons/brain_inspired.jpeg",
}

def display_single_point_details(point_data):
    """Creates a detailed, formatted view for a single selected data point."""
    episode_title = point_data.get("episode_title", "Untitled Episode")
    segment_id_val = point_data.get("segment_id", "N/A")
    raw_podcast_title = point_data.get("podcast_title", "N/A")
    date = point_data.get("date", "N/A")
    topic_name = point_data.get("parent_topic_name", "N/A")
    segment_text = point_data.get("description", "No text available for this segment.")
    icon_src = PODCAST_ICON_MAP.get(raw_podcast_title, "")

    display_podcast_title = PODCAST_NAME_MAP.get(raw_podcast_title, str(raw_podcast_title).replace("_", " ").title())
    host_name = HOST_MAP.get(raw_podcast_title, "N/A")
    display_segment_id = segment_id_val + 1 if isinstance(segment_id_val, int) else segment_id_val

    header = html.Div([
        html.Img(src=icon_src, className="podcast-icon-large"),
        html.H4(episode_title, className="mb-0")
    ], className="single-point-header")

    return html.Div([
        header,
        dbc.Row([
            dbc.Col(html.Strong("Segment:"), width=3),
            dbc.Col(display_segment_id, width=9)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Strong("Podcast:"), width=3),
            dbc.Col(display_podcast_title, width=9)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Strong("Host:"), width=3),
            dbc.Col(host_name, width=9)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Strong("Date:"), width=3),
            dbc.Col(date, width=9)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Strong("Topic:"), width=3),
            dbc.Col(topic_name, width=9)
        ], className="mb-3"),
        html.Hr(),
        html.Strong("Full Segment Text:"),
        html.P(
            segment_text,
            className="mt-2",
            style={'whiteSpace': 'pre-wrap', 'maxHeight': '60vh', 'overflowY': 'auto'}
        )
    ])


def extract_points_data(points_data):
    """
    Pass-through function. The data stored in the app state is already
    in the format required by the StoryGenerator.
    """
    return points_data[:5]


def display_selected_points(points_data, expanded_index=None):
    selected_cards = []
    num_total_points = len(points_data)

    for i, point_data in enumerate(points_data):
        if i >= 25 and num_total_points > 25:
            selected_cards.append(
                html.P(f"... and {num_total_points - 25} more points", className='text-muted more-points-text')
            )
            break

        is_expanded = (expanded_index == i)

        raw_podcast_title = point_data.get('podcast_title', 'N/A')
        icon_src = PODCAST_ICON_MAP.get(raw_podcast_title, "")
        display_podcast_title = PODCAST_NAME_MAP.get(raw_podcast_title, str(raw_podcast_title).replace("_", " ").title())
        host_name = HOST_MAP.get(raw_podcast_title, "N/A")
        topic_name = point_data.get("topic_name", "N/A")
        segment_id_val = point_data.get("segment_id", "N/A")
        display_segment_id = segment_id_val + 1 if isinstance(segment_id_val, int) else segment_id_val

        unexpanded_content = dbc.Row([
            dbc.Col(html.Img(src=icon_src, className="podcast-icon-small"), width="auto"),
            dbc.Col([
                html.H6(point_data.get('episode_title', 'N/A'), className='podcast-title'),
                html.P(display_podcast_title, className='text-muted mb-0 topic-name'),
                dbc.Badge(topic_name, pill=True, color="primary", className="me-1")
            ], className="ps-2")
        ], align="center", className="g-0")

        expanded_content = dbc.Collapse([
            html.Hr(className='card-divider'),
            html.Div([
                dbc.Row([
                    dbc.Col(html.Strong("Segment:"), width=3),
                    dbc.Col(display_segment_id, width=9)
                ], className="mb-1"),
                dbc.Row([
                    dbc.Col(html.Strong("Podcast:"), width=3),
                    dbc.Col(display_podcast_title, width=9)
                ], className="mb-1"),
                dbc.Row([
                    dbc.Col(html.Strong("Host:"), width=3),
                    dbc.Col(host_name, width=9)
                ], className="mb-2"),
                html.Strong('Summary:', className='summary-label'),
                html.P(
                    point_data.get('summary', 'Summary not available.'),
                    className='summary-text'
                )
            ])
        ], is_open=is_expanded, id={'type': 'podcast-card-collapse', 'index': i})

        card_wrapper = html.Div([
            dbc.Card([
                unexpanded_content,
                expanded_content
            ], className='podcast-card')
        ], id={'type': 'podcast-card', 'index': i}, n_clicks=0)

        selected_cards.append(card_wrapper)

    return dbc.Stack(selected_cards)