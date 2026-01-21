import dash
from dash import dcc, html, Input, Output, State, callback
import os
import json
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.Dataset import Dataset
from src.widgets.focus import focus_on_segments, focus_on_topics
from src.widgets.selection_display import HOST_MAP


def search_episode_segments(query, top_k):
    """
    Called by LLM: returns top_k episode segments from dataset using cosine similarity with query.
    """
    
    print(f"using search_episode_segments, with query: {query}...")

    df = Dataset.get()
    embeddings = Dataset.get_embeddings()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    segments = []
    for idx in top_indices:
        row = df.iloc[idx]
        segments.append({
            "customdata": int(idx),
            "episode_id": int(row["episode_id"]),
            "segment_id": int(row["segment_id"]),
            "podcast_title": str(row["podcast_title"]),
            "episode_title": str(row["episode_title"]),
            "date": str(row["date"]),
            "topic_description_llm": str(row.get("topic_description_llm", "Description not available")),
            "first_5_words": str(row["first_5_words"]),
            "start_index": int(row["start_index"]),
            "end_index": int(row["end_index"]),
            "character_length": int(row["character_length"]),
            "original_text": str(row["original_text"]),
            "summary": str(row.get("summary", ""))
        })
    
    return {"segments": segments}


def search_topics(query, top_k):
    """
    Called by LLM: returns top_k topics from dataset using cosine similarity with query.
    """
    print(f"using search_topics, with query: {query}...")
    topic_model = Dataset.get_topic_model()
    similar_topics, similarities = topic_model.find_topics(query, top_n=top_k)
    
    results = []
    topic_info = topic_model.get_topic_info()
    
    for topic_id, similarity in zip(similar_topics, similarities):
        if topic_id == -1: continue
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        if not topic_row.empty:
            topic_name = topic_row['GPT3'].iloc[0] if 'GPT3' in topic_row.columns else topic_row['Name'].iloc[0]
            results.append({
                "title": f"Topic {topic_id}",
                "topic_name": topic_name,
                "similarity": float(similarity)
            })
    
    return {"topics": results}


search_episode_segments_function = {
    "name": "SEARCH_EPISODE_SEGMENTS",
    "description": "Returns the top_k most similar podcast segments based on the search query",
    "parameters": {
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "Search query for finding relevant podcast segments", },
            "top_k": { "type": "integer", "description": "Number of top similar segments to return", },
        }, "required": ["query", "top_k"],
    },
}

search_topics_function = {
    "name": "SEARCH_TOPICS",
    "description": "Returns the top_k most similar topics based on the search query",
    "parameters": {
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "Topic query for finding relevant segments", },
            "top_k": { "type": "integer", "description": "Number of top similar topics to return", },
        }, "required": ["query", "top_k"],
    },
}

tools = types.Tool(function_declarations=[search_episode_segments_function, search_topics_function])
config = types.GenerateContentConfig(tools=[tools])
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model = "gemini-2.5-flash"

CHATBOT_INSTRUCTIONS = """
You are Discovery, a helpful assistant for exploring a podcast segment dataset.

## PRIMARY GOAL
Your main goal is to answer user questions. You will do this by either reasoning about information provided in a `Context` block or by searching for new information using your tools.

## CONTEXTUAL AWARENESS & DATA FORMAT
**Crucially, some user queries will be prefixed with a `Context:` block containing a JSON object.** This JSON represents podcast segments the user has selected in the UI.

The JSON will look like this:
`{"selected_segments": [{"podcast_title": "...", "host": "...", "date": "...", "original_text": "...", ...}]}`

- **When you see a `Context:` block, you MUST parse the JSON and prioritize using the data within to answer the user's question.**
- The `original_text` field is the full transcript. The `host` field tells you the host of the podcast. The `date` field tells you when it was released.
- **Do NOT use a search tool if the answer can be found within the provided JSON context.** For example, if the user asks "who is the host?", you must answer using the `host` field from the JSON.

## TOOL USAGE
Only use your search tools if the user is asking for *new* information that is not available in the provided context.
- `SEARCH_EPISODE_SEGMENTS`: Finds specific podcast segments from the entire dataset.
- `SEARCH_TOPICS`: Finds broad topics from the entire dataset.

## GUIDELINES
- If a user asks a general question without context, use your tools.
- If a user asks a question with context, use the context.
- Keep responses concise and focused on podcast discovery.
- If a search tool returns no results, inform the user clearly.
"""

def safe_get_text(response):
    """Safely extracts text from a Gemini response to prevent crashes."""
    try:
        return response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError, ValueError):
        return None

def update_chat(prompt, conversation_history=[], selection_context=None):
    """
    Manages the conversation with the Gemini model, including multi-turn context,
    tool usage, and awareness of user selections.
    """
    full_user_prompt = prompt
    if selection_context:
        full_user_prompt = f"Context: {selection_context}\n\nUser Query: {prompt}"

    messages = conversation_history + [f"User: {full_user_prompt}"]

    structured_history = []
    for msg in messages:
        if msg.startswith("User:"):
            role = 'user'
            text = msg.replace("User: ", "", 1)
        elif msg.startswith("Discovery:"):
            role = 'model'
            text = msg.replace("Discovery: ", "", 1)
        else:
            continue
        structured_history.append(types.Content(role=role, parts=[types.Part(text=text)]))

    final_contents = [
        types.Content(role='user', parts=[types.Part(text=CHATBOT_INSTRUCTIONS)]),
        types.Content(role='model', parts=[types.Part(text="Understood. I will use JSON context when provided.")])
    ] + structured_history

    response = client.models.generate_content(
        model=model, contents=final_contents, config=config
    )

    function_result = None
    try:
        function_call = response.candidates[0].content.parts[0].function_call
    except (IndexError, AttributeError):
        function_call = None

    if function_call:
        result = None
        if function_call.name == "SEARCH_EPISODE_SEGMENTS":
            result = search_episode_segments(**function_call.args)
        elif function_call.name == "SEARCH_TOPICS":
            result = search_topics(**function_call.args)
        function_result = result

        second_call_contents = final_contents + [
            response.candidates[0].content,
            types.Content(parts=[types.Part(function_response=types.FunctionResponse(name=function_call.name, response=result))])
        ]
        final_response = client.models.generate_content(model=model, contents=second_call_contents)
        final_text = safe_get_text(final_response)
    else:
        final_text = safe_get_text(response)

    if not final_text or not final_text.strip():
        final_text = "I'm sorry, I could not generate a response. The topic might be restricted. Please try a different query."

    return final_text, function_result

@callback(
    [
        Output('main-plot', 'figure', allow_duplicate=True),
        Output('conversation-store', 'data'),
        Output('selected-points-store', 'data', allow_duplicate=True),
        Output('chat-history-output', 'children'),
        Output('focus-mode-store', 'data'),
        Output('close-focus-button', 'style'),
        Output('chat-input', 'value', allow_duplicate=True)
     ],
    Input('send-button', 'n_clicks'),
    [
        State('main-plot', 'figure'),
        State('chat-input', 'value'),
        State('conversation-store', 'data'),
        State('selected-points-store', 'data')
     ],
     prevent_initial_call=True
)
def handle_chat_callback(n_clicks, current_fig, value, conversation, selected_points):
    if not n_clicks or not value:
        return dash.no_update, conversation, dash.no_update, dash.no_update, dash.no_update, {'display': 'none'}, dash.no_update

    fig_update = dash.no_update
    focus_mode_update = dash.no_update
    close_focus_style = {'display': 'none'}
    selection_context_json = None

    if selected_points:
        context_objects = []
        relevant_columns = [
            'podcast_title', 'episode_title', 'date', 'topic', 'topic_name', 
            'parent_topic_name', 'original_text', 'summary', 'title'
        ]
        for point in selected_points[:5]:
            segment_data = {key: point.get(key) for key in relevant_columns if key in point}
            raw_podcast_title = point.get('podcast_title')
            segment_data['host'] = HOST_MAP.get(raw_podcast_title, "N/A")
            context_objects.append(segment_data)
        
        selection_context_json = json.dumps({"selected_segments": context_objects})

    response = "An error occurred. Please try again."
    function_result = None

    try:
        response, function_result = update_chat(value, conversation, selection_context=selection_context_json)
    except Exception as e:
        print(f"Error calling update_chat: {e}")
        response = f"I'm sorry, a critical error occurred: {e}"

    user_prompt_for_history = value
    if selection_context_json:
         user_prompt_for_history = f"[Query about {len(selected_points)} selected segment(s)]\n{value}"
    
    conversation.append(f"User: {user_prompt_for_history}")
    conversation.append(f"Discovery: {response}")

    chat_display = []
    for message in conversation:
        if message.startswith("User:"):
            className = "chat-user-message"
            content = message.replace("User: ", "", 1)
        else:
            className = "chat-bot-message"
            content = message.replace("Discovery: ", "", 1)
        chat_display.append(html.P(content, className=className, style={'whiteSpace': 'pre-wrap'}))
    chat_history_div = html.Div(chat_display, className="chat-history-display")

    selected_points_data = dash.no_update
    if function_result:
        new_selection = []
        if 'segments' in function_result:
            for segment in function_result['segments']:
                new_selection.append({
                    "customdata": segment["customdata"], "episode_title": segment["episode_title"],
                    "segment_id": segment["segment_id"], "podcast_title": segment["podcast_title"],
                    "date": segment["date"], "description": segment["original_text"],
                    "summary": segment["summary"]
                })
            if new_selection:
                segment_indices = [s['customdata'] for s in new_selection]
                fig_update = focus_on_segments(go.Figure(current_fig), segment_indices, len(segment_indices))
                focus_mode_update = True
                close_focus_style = {'display': 'inline-block'}
        elif 'topics' in function_result:
            topic_ids = [int(t['title'].split(' ')[1]) for t in function_result.get('topics', [])]
            if topic_ids:
                df = Dataset.get()
                segments_in_topics = df[df['topic'].isin(topic_ids)].head(10)
                for idx, row in segments_in_topics.iterrows():
                    new_selection.append({
                        "customdata": idx, "episode_title": row["episode_title"],
                        "segment_id": row["segment_id"], "podcast_title": row["podcast_title"],
                        "date": row["date"], "description": row["original_text"],
                        "summary": row.get("summary", "Summary not available.")
                    })
                fig_update = focus_on_topics(go.Figure(current_fig), topic_ids, len(topic_ids))
                focus_mode_update = True
                close_focus_style = {'display': 'inline-block'}

        if new_selection:
            selected_points_data = new_selection

    return fig_update, conversation, selected_points_data, chat_history_div, focus_mode_update, close_focus_style, ""