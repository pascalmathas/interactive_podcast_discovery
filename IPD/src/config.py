from pathlib import Path

# General settings
SCATTERPLOT_COLOR = 'rgba(31, 119, 180, 0.5)'
SCATTERPLOT_SELECTED_COLOR = 'red'
DEFAULT_PROJECTION = 'UMAP'
DATASET_SAMPLE_SIZE = 1000

# Feature toggles
HOVER_HIGHLIGHT_ENABLED = False  # Toggle for enabling/disabling parent topic hover highlighting

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
ENVIRONMENT = 'dev'

if ENVIRONMENT == 'dev':
    print('Importing DEVELOPMENT config...')
    # dev folders are one level deeper
    DATASET_DIR = ROOT_DIR.parent / 'dataset'
    
    print("Current working directory:", Path.cwd())
else:
    print('Importing PRODUCTION config...')
    DATASET_DIR = ROOT_DIR / 'dataset'

# Base directories
DATA_DIR = DATASET_DIR / 'data'
DATASET_TRANSCRIBED_DIR = DATASET_DIR / 'transcribed'
DATASET_SEGMENTED_DIR = DATASET_DIR / 'segmented'
DATASET_EMBEDDED_DIR = DATASET_DIR / 'embedded'

# Transcriptions
PODCAST_TANSCRIBED_PATHS = {
    "the_daily": DATASET_TRANSCRIBED_DIR / "parquet" / "The_Daily.parquet",
    "joe_rogan": DATASET_TRANSCRIBED_DIR / "parquet" / "joerogan.parquet",
    "lex_fridman": DATASET_TRANSCRIBED_DIR / "parquet" / "lex_fridman_podcast.parquet",
    "pom": DATASET_TRANSCRIBED_DIR / "parquet" / "POM - Een podcast over media.parquet",
    "mlst": DATASET_TRANSCRIBED_DIR
    / "parquet"
    / "Machine_Learning_Street_Talk_MLST_2025-06-20-2020-04-24.parquet",
}

# Segmentation
PODCAST_SEGMENTED_DIRS = {
    "the_daily": DATASET_SEGMENTED_DIR / "the_daily",
    "joe_rogan": DATASET_SEGMENTED_DIR / "joe_rogan",
    "lex_fridman": DATASET_SEGMENTED_DIR / "lex_fridman",
    "pom": DATASET_SEGMENTED_DIR / "pom",
    "mlst": DATASET_SEGMENTED_DIR / "mlst",
}

# BERTopic
BERTOPIC_DIR = DATASET_DIR / 'bertopic'
PODCAST_SEGMENTS_PATH = BERTOPIC_DIR / 'podcast_segments_with_topics.pkl'
EMBEDDINGS_PATH = BERTOPIC_DIR / 'embeddings.npy'
REDUCED_EMBEDDINGS_PATH = BERTOPIC_DIR / 'reduced_embeddings.npy'
BERTOPIC_MODEL_PATH = BERTOPIC_DIR / 'bertopic_model'
