import dash_bootstrap_components as dbc
from dash import callback, Input, Output, State, Patch, no_update, html
from src.widgets.selection_display import PODCAST_NAME_MAP, PODCAST_ICON_MAP
import time
import re
from src.config import HOVER_HIGHLIGHT_ENABLED

HOVER_THROTTLE_MS = 10  # Minimum time between hover updates
FOCUS_BOX_IMAGE_SIZE = 60
FOCUS_BOX_BORDER_RADIUS = 6
FOCUS_BOX_CARD_RADIUS = 10
FOCUS_BOX_CARD_PADDING = 4
FOCUS_BOX_CARD_MAX_WIDTH = 500
FOCUS_BOX_CARD_BLUR = 1

def get_luminance(color_str):
    """
    Calculate the relative luminance of a color for contrast calculations.
    Returns a value between 0 (darkest) and 1 (lightest).
    """
    def parse_color_component(component):
        component = float(component) / 255.0
        if component <= 0.03928:
            return component / 12.92
        else:
            return pow((component + 0.055) / 1.055, 2.4)
    
    # Extract RGB values from different color formats
    if color_str.startswith('rgba'):
        match = re.match(r'rgba\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*[\d.]+\)', color_str)
        if match:
            r, g, b = match.groups()
        else:
            return 0.5  # fallback
    elif color_str.startswith('rgb'):
        match = re.match(r'rgb\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', color_str)
        if match:
            r, g, b = match.groups()
        else:
            return 0.5  # fallback
    elif color_str.startswith('#'):
        if len(color_str) == 4:  # #RGB
            r, g, b = [int(color_str[i]*2, 16) for i in range(1, 4)]
        elif len(color_str) == 7:  # #RRGGBB
            r, g, b = [int(color_str[i:i+2], 16) for i in (1, 3, 5)]
        else:
            return 0.5  # fallback
    else:
        return 0.5  # fallback for unknown formats
    
    r_linear = parse_color_component(float(r))
    g_linear = parse_color_component(float(g))
    b_linear = parse_color_component(float(b))
    
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

def get_contrast_ratio(color1_luminance, color2_luminance):
    """Calculate contrast ratio between two luminance values."""
    lighter = max(color1_luminance, color2_luminance)
    darker = min(color1_luminance, color2_luminance)
    return (lighter + 0.05) / (darker + 0.05)

def get_accessible_text_color(background_color):
    """
    Determine the best text color (white or black) for AAA contrast (7:1 ratio).
    Returns the color that provides the highest contrast.
    """
    bg_luminance = get_luminance(background_color)
    white_luminance = 1.0
    black_luminance = 0.0
    
    white_contrast = get_contrast_ratio(bg_luminance, white_luminance)
    black_contrast = get_contrast_ratio(bg_luminance, black_luminance)
    
    # For AAA compliance, we need at least 7:1 contrast ratio
    # Return the color with the highest contrast
    if white_contrast >= black_contrast:
        return "#FFFFFF" if white_contrast >= 7.0 else "#000000"
    else:
        return "#000000" if black_contrast >= 7.0 else "#FFFFFF"

@callback(
    Output('main-plot', 'figure', allow_duplicate=True),
    Output('hover-throttle-store', 'data'),
    Input('main-plot', 'hoverData'),
    State('main-plot', 'figure'),
    State('isolation-mode-store', 'data'),
    State('focus-mode-store', 'data'),
    State('hover-throttle-store', 'data'),
    prevent_initial_call=True
)
def highlight_on_hover(hoverData, figure, is_in_isolation_mode, is_in_focus_mode, throttle_data):
    """
    Optimized hover highlighting using Patch() for partial updates and performance throttling.
    Throttling state is stored in dcc.Store (hover-throttle-store) for per-session safety.
    """
    if not HOVER_HIGHLIGHT_ENABLED:
        return no_update, throttle_data
    if not figure:
        return no_update, throttle_data
    if is_in_isolation_mode or is_in_focus_mode:
        return no_update, throttle_data
    current_time = time.time() * 1000  # ms
    last_hover_topic = throttle_data.get('last_topic') if throttle_data else None
    last_hover_time = throttle_data.get('last_time', 0) if throttle_data else 0
    hovered_topic = None
    if hoverData and hoverData.get('points'):
        point = hoverData['points'][0]
        if 'customdata' in point and len(point['customdata']) > 2:
            hovered_topic = point['customdata'][1]
    if (hovered_topic == last_hover_topic and current_time - last_hover_time < HOVER_THROTTLE_MS):
        return no_update, throttle_data
    new_throttle_data = {'last_topic': hovered_topic, 'last_time': current_time}
    patched_figure = Patch()
    for i, trace in enumerate(figure['data']):
        if trace.get('mode') != 'markers' or 'customdata' not in trace:
            continue
        if hovered_topic and hovered_topic != 'N/A':
            line_colors = []
            line_widths = []
            for cd in trace['customdata']:
                if len(cd) > 2 and cd[1] == hovered_topic:
                    line_colors.append('rgba(0,0,0,1)')
                    line_widths.append(2)
                else:
                    line_colors.append('rgba(0,0,0,0)')
                    line_widths.append(0)
        else:
            num_points = len(trace['customdata'])
            line_colors = ['rgba(0,0,0,0)'] * num_points
            line_widths = [0] * num_points
        patched_figure['data'][i]['marker']['line']['color'] = line_colors
        patched_figure['data'][i]['marker']['line']['width'] = line_widths
    return patched_figure, new_throttle_data

@callback(
    Output('focus-box', 'children'),
    Input('main-plot', 'hoverData'),
    State('main-plot', 'figure')
)
def update_focus_box(hoverData, figure):
    """Return a styled Dash-Bootstrap card with segment details and matching colour."""
    if hoverData is None or not hoverData.get('points'):
        return []

    try:
        point = hoverData['points'][0]
        custom_data = point.get('customdata')
        if not custom_data or len(custom_data) < 4:
            return []
        podcast_title, parent_topic, topic_name, segment_title, _ = custom_data
        podcast_display_name = PODCAST_NAME_MAP.get(
            podcast_title, podcast_title.replace("_", " ").title()
        )
        color = None
        for key in ('marker.color', 'color'):
            if key in point:
                color = point[key]
                break
        if color is None and figure:
            curve_i = point.get('curveNumber')
            point_i = point.get('pointIndex') or point.get('pointNumber')
            try:
                trace = figure['data'][curve_i]
                trace_colors = trace['marker']['color']
                if isinstance(trace_colors, (list, tuple)):
                    color = trace_colors[point_i]
                else:
                    color = trace_colors
            except Exception:
                pass
        if color is None:
            color = 'rgb(255,255,255)'
        def to_rgba(col, alpha=0.25):
            if col.startswith('rgba'):
                parts = col[5:-1].split(',')
                if len(parts) == 4:
                    parts[-1] = str(alpha)
                else:
                    parts.append(str(alpha))
                return f"rgba({','.join(parts)})"
            if col.startswith('rgb'):
                return col.replace('rgb', 'rgba').replace(')', f', {alpha})')
            if col.startswith('#') and len(col) in (7, 4):
                if len(col) == 4:
                    r, g, b = [int(col[i]*2, 16) for i in range(1,4)]
                else:
                    r, g, b = [int(col[i:i+2], 16) for i in (1,3,5)]
                return f"rgba({r}, {g}, {b}, {alpha})"
            return f"rgba(0,0,0,{alpha})"
        bg_color = to_rgba(str(color), alpha=0.8)
        text_color = get_accessible_text_color(bg_color)
        icon_src = PODCAST_ICON_MAP.get(podcast_title, "")
        card = dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Img(
                            src=icon_src,
                            style={
                                "width": f"{FOCUS_BOX_IMAGE_SIZE}px",
                                "height": f"{FOCUS_BOX_IMAGE_SIZE}px",
                                "objectFit": "cover",
                                "borderRadius": f"{FOCUS_BOX_BORDER_RADIUS}px",
                            },
                        ),
                        width="auto",
                        style={"margin-right": "10px"}
                    ),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(
                                html.H6(
                                    segment_title,
                                    className="mb-1",
                                    style={
                                        "fontSize": "15px",
                                        "fontWeight": "bold",
                                        "color": text_color,
                                    },
                                ),
                                width=12,
                            )
                        ]),
                        dbc.Row([
                            dbc.Col(
                                html.Small(
                                    "Podcast:",
                                    style={"fontWeight": "bold", "color": text_color},
                                ),
                                width=12,
                            ),
                            dbc.Col(
                                html.Small(
                                    podcast_display_name,
                                    className="d-block",
                                    style={"fontWeight": "normal", "color": text_color},
                                ),
                                width=12,
                            ),
                        ], className="d-block"),
                        dbc.Row([
                            dbc.Col(
                                html.Small(
                                    "Parent Topic:",
                                    className="d-block",
                                    style={"fontWeight": "bold", "color": text_color},
                                ),
                                width=12,
                            ),
                            dbc.Col(
                                html.Small(
                                    parent_topic,
                                    className="d-block",
                                    style={"fontWeight": "normal", "color": text_color},
                                ),
                                width=12,
                            ),
                        ]),
                        dbc.Row([
                            dbc.Col(
                                html.Small(
                                    "Topic:",
                                    className="d-block",
                                    style={"fontWeight": "bold", "color": text_color},
                                ),
                                width=12,
                            ),
                            dbc.Col(
                                html.Small(
                                    topic_name,
                                    className="d-block",
                                    style={"fontWeight": "normal", "color": text_color},
                                ),
                                width=12,
                            ),
                        ]),
                    ], className="focus-box-content"),
                ], className="g-1 align-items-center"),
            ]),
            style={
                "backgroundColor": bg_color,
                "borderRadius": f"{FOCUS_BOX_CARD_RADIUS}px",
                "backdropFilter": f"blur({FOCUS_BOX_CARD_BLUR}px)",
                "padding": f"{FOCUS_BOX_CARD_PADDING}px",
                "maxWidth": f"{FOCUS_BOX_CARD_MAX_WIDTH}px",
            },
            className="shadow",
        )
        return card
    except (KeyError, IndexError, TypeError):
        return [] 