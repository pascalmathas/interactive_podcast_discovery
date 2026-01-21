import os
import sys
import json
import glob
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    TextGeneration,
)
from bertopic.vectorizers import ClassTfidfTransformer

from openai import OpenAI

from src.data_pipeline.embedding import *
from src.data_pipeline.embedding.utils import *

def run_bertopic_pipeline(data_path,
                            cache_path,
                            openai_api_key=None,
                            embedding_model_name='all-MiniLM-L6-v2',
                            force_retrain=False,
                            umap_params=None,
                            hdbscan_params=None,
                            bertopic_params=None,
                            seed=42):

    if openai_api_key:
        print("Confirmed OpenAI API key in run_bertopic_pipeline")
        client = OpenAI(api_key=openai_api_key)
    else:
        print("No OpenAI API key provided in run_bertopic_pipeline")
        client = None
    
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'n_components': 5,
            'min_dist': 0.0,
            'metric': 'cosine',
            'random_state': seed
        }
    
    if hdbscan_params is None:
        hdbscan_params = {
            'min_cluster_size': 5,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'prediction_data': True
        }
    
    if bertopic_params is None:
        bertopic_params = {
            'top_n_words': 5,
            'verbose': True
        }
    
    SYSTEM_PROMPT = """
    You are a helpful assistant specializing in artificial intelligence and machine learning research and applications. Your task is to create concise, descriptive topic labels extracted from podcast transcripts.
    """

    EXAMPLE_PROMPT = """
    I have a topic that contains the following documents:
    - Large language models like GPT and BERT have revolutionized natural language processing, enabling few-shot learning and emergent capabilities at scale.
    - Transformer architectures with attention mechanisms have proven highly effective across diverse domains, from computer vision to protein folding prediction.
    - Fine-tuning pre-trained models has become a standard approach, allowing practitioners to adapt powerful foundation models to specific downstream tasks.
    The topic is described by the following keywords: 'transformer, attention, language, model, GPT, BERT, fine-tuning, pre-training, few-shot, foundation models'.
    Based on the information about the topic above, please create a short topic label. Make sure to only return the topic label and nothing else.
    Large language models and transformer architectures
    """

    MAIN_PROMPT = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: '[KEYWORDS]'.
    Based on the information above, extract a short topic label. Make sure to only return the topic label and nothing else.
    """
    
    PROMPT = EXAMPLE_PROMPT + MAIN_PROMPT
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    embeddings_path = os.path.join(cache_path, "embeddings.npy")
    model_path = os.path.join(cache_path, "bertopic_model")
    topics_path = os.path.join(cache_path, "topics.npy")
    probs_path = os.path.join(cache_path, "probs.npy")
    labels_path = os.path.join(cache_path, "custom_topic_labels.pkl")
    reduced_embeddings_path = os.path.join(cache_path, "reduced_embeddings.npy")
    dataframe_path = os.path.join(cache_path, "podcast_segments_with_topics.pkl")
    parent_topics_path = os.path.join(cache_path, "parent_topics_map.pkl")
    
    print(f"Loading dataset from {data_path}...")
    dataset = load_podcast_data(data_path)
    transcripts = [doc['original_text'] for doc in dataset]
    print(f"Dataset loaded with {len(dataset)} rows")
    
    embedding_model = SentenceTransformer(embedding_model_name)
    
    if os.path.exists(embeddings_path) and not force_retrain:
        print("Loading existing embeddings from cache...")
        embeddings = np.load(embeddings_path)
    else:
        print("Generating embeddings... This might take a while.")
        embeddings = embedding_model.encode(transcripts, show_progress_bar=True, device="mps")
        print("Saving embeddings to cache...")
        np.save(embeddings_path, embeddings)
    
    if os.path.exists(model_path) and not force_retrain:
        print("Loading existing BERTopic model from cache...")
        topic_model = BERTopic.load(model_path)
        
        print("Loading topics and probabilities from cache...")
        topics = np.load(topics_path)
        probs = np.load(probs_path)
    else:
        print("No cached model found or force_retrain=True. Fitting a new BERTopic model...")
        
        umap_model = UMAP(**umap_params)
        hdbscan_model = HDBSCAN(**hdbscan_params)
        
        # DEV-NOTE: A custom CountVectorizer is introduced to improve keyword quality.
        # By removing common English stop words and filtering terms based on document
        # frequency (min_df, max_df), we prevent overly common or rare words from
        # polluting the topic representations, leading to cleaner keywords.
        # vectorizer_model = CountVectorizer(stop_words="english", min_df=10, max_df=0.85)

        # DEV-NOTE: The c-TF-IDF transformer is now customized with BM25 weighting.
        # BM25 is a ranking function that improves on standard TF-IDF by better
        # handling documents of varying lengths, which is suitable for our podcast segments.
        # This leads to more robust and representative keyword scores.
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
        
        keybert = KeyBERTInspired()
        mmr = MaximalMarginalRelevance(diversity=0.3)
        
        representation_model = {
            "KeyBERT": keybert,
            "MMR": mmr,
        }
        
        if client:
            try:
                gpt3 = GPT3TopicLabeler(SYSTEM_PROMPT, PROMPT)
                representation_model["GPT3"] = gpt3
            except ImportError:
                print("Warning: GPT3TopicLabeler not found. Proceeding without GPT-3 labeling.")
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            # DEV-NOTE: Pass the custom vectorizer and c-TF-IDF models to BERTopic.
            # vectorizer_model=vectorizer_model,
            # ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            **bertopic_params
        )
        
        topics, probs = topic_model.fit_transform(transcripts, embeddings)
        
        print("Saving fitted model and results to cache...")
        topic_model.save(model_path, serialization="pickle")
        np.save(topics_path, topics)
        np.save(probs_path, probs)

    if os.path.exists(reduced_embeddings_path) and not force_retrain:
        print("Loading existing reduced embeddings from cache...")
        reduced_embeddings = np.load(reduced_embeddings_path)
    else:
        print("Generating reduced embeddings for visualization...")
        vis_umap_params = umap_params.copy()
        vis_umap_params['n_components'] = 2
        reduced_embeddings = UMAP(**vis_umap_params).fit_transform(embeddings)
        print("Saving reduced embeddings to cache...")
        np.save(reduced_embeddings_path, reduced_embeddings)
    
    print("Generating topic labels...")
    topic_info = topic_model.get_topic_info()
    
    if "GPT3" in topic_model.get_topics(full=True):
        gpt3_representations = topic_model.get_topics(full=True)["GPT3"]
    else:
        print("Warning: 'GPT3' representation not found in the model. Using fallback for all topics.")
        gpt3_representations = {}
    
    topic_ids = list(topic_model.get_topics().keys())
    topic_labels = []
    
    for topic_id in topic_ids:
        if topic_id in gpt3_representations and gpt3_representations[topic_id]:
            gpt3_label = gpt3_representations[topic_id][0][0].split("\n")[0]
            topic_labels.append(gpt3_label)
        else:
            if topic_id == -1:
                topic_labels.append("Outliers")
            else:
                keybert_words = topic_model.get_topic(topic_id)
                if keybert_words:
                    fallback_label = f"Topic about {keybert_words[0][0]}"
                else:
                    fallback_label = f"Topic {topic_id}"
                topic_labels.append(fallback_label)
    
    topic_labels_map = dict(zip(topic_ids, topic_labels))
    
    topic_model.set_topic_labels(topic_labels)
    
    print(f"Saving custom topic labels to: {labels_path}")
    with open(labels_path, "wb") as f:
        pickle.dump(topic_labels_map, f)
        
    print("Creating dataframe...")
    df = pd.DataFrame(dataset)
    df['topic'] = topics
    df['topic_name'] = df['topic'].map(topic_labels_map)
    df['title'] = df['episode_title']
    df['description'] = df['original_text']
    df['text'] = df['original_text']
    
    print("Processing hierarchical topics with k-means clustering...")
    try:
        child_to_parent_map = None

        if os.path.exists(parent_topics_path) and not force_retrain:
            print(f"Loading cached parent topics from {parent_topics_path}...")
            with open(parent_topics_path, "rb") as f:
                child_to_parent_map = pickle.load(f)
        
        else:
            print("Generating new parent topics using k-means clustering...")
            
            child_to_parent_map = generate_parent_topics_with_kmeans(
                topic_model=topic_model,
                embeddings=embeddings,
                topics=topics,
                topic_labels_map=topic_labels_map,
                n_parent_topics=14,
                min_parent_topics=3,
                max_parent_topics=min(10, len(topic_labels_map) // 2),
                client=client
            )

            if child_to_parent_map:
                print(f"Generated {len(set(child_to_parent_map.values()))} parent topics")
                print(f"Saving parent topics to cache: {parent_topics_path}")
                with open(parent_topics_path, "wb") as f:
                    pickle.dump(child_to_parent_map, f)
            else:
                print("Warning: K-means parent topic generation returned no results.")

        if child_to_parent_map:
            df['parent_topic_name'] = df['topic_name'].map(child_to_parent_map)
            df['parent_topic_name'].fillna('General Topics', inplace=True)

            parent_names = sorted(df['parent_topic_name'].unique())
            parent_name_to_id = {name: i for i, name in enumerate(parent_names)}
            df['parent_topic_id'] = df['parent_topic_name'].map(parent_name_to_id)
            print("Successfully added parent topic information to dataframe.")
            print(f"Parent topics created: {list(set(child_to_parent_map.values()))}")
            
        else:
            raise ValueError("Could not load or generate parent topics.")

    except Exception as e:
        print(f"Could not process hierarchical topics: {e}. Assigning fallback values.")
        df['parent_topic_id'] = -1
        df['parent_topic_name'] = 'N/A'

    print("\nTopic distribution:")
    print(df['topic_name'].value_counts())

    print("\nParent Topic distribution:")
    print(df['parent_topic_name'].value_counts())
    
    print("\nPodcast distribution:")
    print(df['podcast_title'].value_counts())
    
    print(f"Saving dataframe to {dataframe_path}...")
    df.to_pickle(dataframe_path)
    
    print("\nPipeline complete!")
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns in dataframe: {df.columns.tolist()}")
    
    # ['episode_id', 'podcast_title', 'episode_title', 'date', 'segment_id', 
    # 'topic_description_llm', 'first_5_words', 'start_index', 'end_index', 
    # 'character_length', 'original_text', 'commercial_advertiser', 
    # 'profanity_types', 'last_5_5_words_topic', 'topic', 'topic_name', 
    # 'title', 'description', 'text', 'parent_topic_name']
    
    return topic_model, df, topic_labels_map
