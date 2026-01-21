
import os
import pandas
import numpy
from bertopic import BERTopic
from src import config
from src.topic_labeler import GPT3TopicLabeler

from src.search_engine import SearchEngine


class Dataset:
    data: pandas.DataFrame | None = None
    embeddings = None
    embeddings_2d = None
    topic_model = None
    _search_engine = None

    @staticmethod
    def load():
        Dataset.data = pandas.read_pickle(config.PODCAST_SEGMENTS_PATH)
        Dataset.embeddings = numpy.load(config.EMBEDDINGS_PATH)
        Dataset.embeddings_2d = numpy.load(config.REDUCED_EMBEDDINGS_PATH)
        Dataset.topic_model = BERTopic.load(config.BERTOPIC_MODEL_PATH)

    @staticmethod
    def get() -> pandas.DataFrame | None:
        return Dataset.data
    
    @staticmethod
    def get_embeddings():
        return Dataset.embeddings

    @staticmethod
    def get_embeddings_2d():
        return Dataset.embeddings_2d

    @staticmethod
    def get_topic_model():
        return Dataset.topic_model

    @staticmethod
    def files_exist():
        return (os.path.isfile(config.PODCAST_SEGMENTS_PATH) and
                os.path.isfile(config.EMBEDDINGS_PATH) and
                os.path.isfile(config.REDUCED_EMBEDDINGS_PATH) and
                os.path.isfile(config.BERTOPIC_MODEL_PATH))

    @staticmethod
    def get_search_engine():
        if Dataset._search_engine is None:
            Dataset._search_engine = SearchEngine(
                Dataset.get(),
                Dataset.get_embeddings(),
                Dataset.get_topic_model()
            )
        return Dataset._search_engine

    @staticmethod
    def search(query):
        return Dataset.get_search_engine().search(query)

    @staticmethod
    def get_relevant_topics(query, threshold='auto'):
        return Dataset.get_search_engine().filter_topics_by_relevance(query, threshold)

    @staticmethod
    def search_topics_semantic(query, top_n=5):
        return Dataset.get_search_engine().search_topics_semantic(query, top_n)
