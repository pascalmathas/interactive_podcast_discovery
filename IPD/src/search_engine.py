import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SearchEngine:
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, topic_model = None) -> None:
        self.df = df
        self.topic_model = topic_model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.document_embeddings = embeddings
        print("Search engine initialized with pre-computed embeddings.")
    
    def search(self, query: str, limit: int | None = None) -> pd.DataFrame:
        """
        Encode query and calculate cosine similarity with documents. Returns sorted df with relevance scores added.
        """
        if not query.strip():
            return pd.DataFrame()
        
        query_embedding = self.embedding_model.encode([query])

        scores = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        
        results = self.df.copy()
        results['relevance_score'] = scores
        results = results[scores > 0].sort_values('relevance_score', ascending=False)
        
        return results.head(limit) if limit else results
    
    def get_topic_relevance(self, query: str, min_score: float = 0.2) -> dict:
        """
        Returns results stats for topics with scores above min score
        """
        if not query.strip():
            return {}
        
        results = self.search(query)
        if results.empty:
            return {}
        
        results = results[results['relevance_score'] >= min_score]
        
        topic_scores = {}
        for topic in results['topic'].unique():
            topic_results = results[results['topic'] == topic]
            topic_scores[topic] = {
                'mean_score': topic_results['relevance_score'].mean(),
                'max_score': topic_results['relevance_score'].max(),
                'doc_count': len(topic_results)
            }
        
        return topic_scores
    
    def filter_topics_by_relevance(self, query: str, threshold: str | float = 'auto', metric: str = 'mean_score') -> list:
        """
        Returns list of topic IDs based on filtering criteria.
        """
        topic_scores = self.get_topic_relevance(query)
        if not topic_scores:
            return []
        
        scores = [info[metric] for info in topic_scores.values()]
        
        if not scores:  # Safety check for empty scores
            return []
        
        if threshold == 'auto':
            # TO DO: NOG TWEAKEN VOOR BESTE RESULTATEN
            # we kunnen ook andere opties gebruiken zoals Elbow Method of Top-K, of statistische methods etc.
            # Use median for more than 5 topics, or keep 90% of min for few topics
            threshold = np.percentile(scores, 50) if len(scores) > 5 else min(scores) * 0.9
        
        return [topic for topic, info in topic_scores.items() if info[metric] >= threshold]
