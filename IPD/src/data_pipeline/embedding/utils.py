import json
import os
from pathlib import Path
from openai import OpenAI
from bertopic.representation._base import BaseRepresentation
from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import pickle

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_podcast_data(base_dir):
    """
    Load all JSON files from podcast directories and combine them into a single list.
    Prints an example data row and returns the complete dataset.
    
    Args:
        base_dir (str): Base directory containing podcast folders
    
    Returns:
        list: Combined list of all podcast episode segments
    """
    combined_data = []
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Base directory '{base_dir}' does not exist!")
        return combined_data
    
    for podcast_folder in base_path.iterdir():
        if podcast_folder.is_dir():
            print(f"Processing podcast folder: {podcast_folder.name}")
            
            json_files = list(podcast_folder.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        combined_data.extend(data)
                    elif isinstance(data, dict):
                        combined_data.append(data)
                    
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON in {json_file}: {e}")
                except Exception as e:
                    print(f"  Error processing {json_file}: {e}")
    
    print(f"\nLoaded {len(combined_data)} total segments")
    
    if combined_data:
        print("\nExample data row:")
        print("-" * 50)
        example_row = combined_data[0]
        for key, value in example_row.items():
            if isinstance(value, str) and len(value) > 100:
                display_value = value[:100] + "..."
            else:
                display_value = value
            print(f"{key}: {display_value}")
        
        podcasts = set(item.get('podcast_title', 'Unknown') for item in combined_data)
        episodes = set(item.get('episode_title', 'Unknown') for item in combined_data)
        
        print(f"\nDataset Statistics:")
        print(f"  Total segments: {len(combined_data)}")
        print(f"  Unique podcasts: {len(podcasts)}")
        print(f"  Unique episodes: {len(episodes)}")
    
    return combined_data

def generate_topic_label(documents, keywords, system_prompt, prompt_template):
    formatted_prompt = prompt_template.replace('[DOCUMENTS]', '\n'.join([f"- {doc}" for doc in documents]))
    formatted_prompt = formatted_prompt.replace('[KEYWORDS]', ', '.join(keywords))
    
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ],
    )
    topic_label = response.choices[0].message.content.strip()
    return topic_label

class GPT3TopicLabeler(BaseRepresentation):
    def __init__(self, system_prompt, prompt_template):
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
    
    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """Extract topic representations using GPT-3"""
        topic_representations = {}
        
        for topic in set(topics):
            if topic == -1:
                continue
            topic_docs = [documents.iloc[i]['Document'] for i in range(len(documents)) if documents.iloc[i]['Topic'] == topic]
            
            try:
                if hasattr(topic_model, 'get_topic') and topic_model.get_topic(topic):
                    keywords = [word for word, _ in topic_model.get_topic(topic)[:10]]
                else:
                    if c_tf_idf is not None and topic < len(c_tf_idf.toarray()):
                        top_indices = c_tf_idf.toarray()[topic].argsort()[-10:][::-1]
                        keywords = [f"keyword_{i}" for i in top_indices]
                    else:
                        keywords = []
            except:
                keywords = []
            
            try:
                sample_docs = topic_docs[:5] if len(topic_docs) > 5 else topic_docs
                
                doc_text = '\n'.join([f"- {doc[:200]}..." if len(str(doc)) > 200 else f"- {doc}" for doc in sample_docs])
                keyword_text = ', '.join(keywords) if keywords else 'no specific keywords'
                
                formatted_prompt = self.prompt_template.replace('[DOCUMENTS]', doc_text)
                formatted_prompt = formatted_prompt.replace('[KEYWORDS]', keyword_text)
                
                response = client.chat.completions.create(
                    # model="gpt-3.5-turbo-0125",
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": formatted_prompt},
                    ],
                )
                topic_label = response.choices[0].message.content.strip()
                
                topic_representations[topic] = [(topic_label, 1.0)]
                
            except Exception as e:
                print(f"Error generating label for topic {topic}: {e}")
                if keywords:
                    topic_representations[topic] = [(f"Topic about {keywords[0]}", 1.0)]
                else:
                    topic_representations[topic] = [(f"Topic {topic}", 1.0)]
        
        return topic_representations

def generate_parent_topics_with_kmeans(topic_model, embeddings, topics, topic_labels_map, 
                                     n_parent_topics=None, min_parent_topics=3, 
                                     max_parent_topics=12, client=None):
    """
    Generate parent topics using k-means clustering on topic centroids.
    
    Args:
        topic_model: Fitted BERTopic model
        embeddings: Document embeddings used for BERTopic
        topics: Topic assignments for each document
        topic_labels_map: Map from topic ID to topic label
        n_parent_topics: Fixed number of parent topics (if None, will optimize)
        min_parent_topics: Minimum number of parent topics to consider
        max_parent_topics: Maximum number of parent topics to consider
        client: OpenAI client for generating parent topic names
    
    Returns:
        Dict[str, str]: Map from child topic label to parent topic label
    """
    valid_topic_ids = [tid for tid in topic_labels_map.keys() if tid != -1]
    
    if len(valid_topic_ids) < min_parent_topics:
        print(f"Only {len(valid_topic_ids)} topics found, skipping parent topic generation")
        return {}
    
    topic_centroids = {}
    for topic_id in valid_topic_ids:
        topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        if topic_doc_indices:
            topic_embeddings = embeddings[topic_doc_indices]
            topic_centroids[topic_id] = np.median(topic_embeddings, axis=0)
    
    if len(topic_centroids) < min_parent_topics:
        print(f"Only {len(topic_centroids)} valid topic centroids, skipping parent topic generation")
        return {}
    
    topic_ids_array = list(topic_centroids.keys())
    centroids_array = np.array([topic_centroids[tid] for tid in topic_ids_array])
    
    if n_parent_topics is None:
        n_parent_topics = find_optimal_parent_clusters(
            centroids_array, min_parent_topics, max_parent_topics
        )
        print(f"Optimal number of parent topics: {n_parent_topics}")
        
    kmeans = KMeans(n_clusters=n_parent_topics, random_state=42, n_init=10)
    parent_cluster_assignments = kmeans.fit_predict(centroids_array)
    
    parent_clusters = defaultdict(list)
    for i, topic_id in enumerate(topic_ids_array):
        parent_cluster_id = parent_cluster_assignments[i]
        parent_clusters[parent_cluster_id].append(topic_id)
    
    child_to_parent_map = {}
    
    for parent_cluster_id, child_topic_ids in parent_clusters.items():
        child_labels = [topic_labels_map[tid] for tid in child_topic_ids]
        
        try: 
            parent_name = generate_parent_name_with_gpt(child_labels, client)
        except Exception as e:
            print(f"Error generating parent name with GPT: {e}")
        
        for child_label in child_labels:
            child_to_parent_map[child_label] = parent_name
    
    return child_to_parent_map

def find_optimal_parent_clusters(centroids_array, min_k, max_k):
    """
    Find optimal number of clusters using silhouette score and elbow method.
    """
    if len(centroids_array) <= min_k:
        return min_k
    
    max_k = min(max_k, len(centroids_array) - 1)
    
    silhouette_scores = []
    inertias = []
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(centroids_array)
        
        sil_score = silhouette_score(centroids_array, cluster_labels)
        silhouette_scores.append(sil_score)
        inertias.append(kmeans.inertia_)
    
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    best_k_elbow = find_elbow_point(k_range, inertias)
    
    if abs(best_k_silhouette - best_k_elbow) <= 2:
        optimal_k = best_k_silhouette
    else:
        optimal_k = int(np.round((best_k_silhouette + best_k_elbow) / 2))
    
    return max(min_k, min(optimal_k, max_k))

def find_elbow_point(k_range, inertias):
    """
    Find elbow point using the largest decrease in inertia.
    """
    if len(inertias) < 2:
        return k_range[0]
    
    deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    
    elbow_idx = np.argmax(deltas)
    return k_range[elbow_idx]

def generate_parent_name_with_gpt(child_labels, client):
    """
    Generate a parent topic name using GPT based on child topic labels.
    """
    child_list = "\n- ".join(child_labels)
    
    system_prompt = """
You are an expert in topic modeling. Your task is to create a concise, descriptive parent topic name that encompasses a group of related sub-topics. The topic name should be:
1. 2-5 words short
2. Descriptive and specific
3. Capture the common theme across all sub-topics
4. Clear and concise
5. Use the same language as the sub-topics
The topic name should not:
1. include the podcast title

Your response should only include the parent topic name, nothing else.
"""
    
    user_prompt = f"""
Create a parent topic name for these related sub-topics:
- {child_list}

What is the overarching theme that connects these sub-topics?
"""
    
    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        
        parent_name = response.choices[0].message.content.strip()
        parent_name = parent_name.replace('"', '').replace("'", "")
        return parent_name
        
    except Exception as e:
        print(f"Error generating parent name with GPT: {e}")
