import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data configurations
DATASET_NAME = "amazon_polarity"
DATASET_CONFIG = "amazon_polarity"
DATASET_SPLIT = "train"
MAX_PRODUCTS = 50000

# Directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
SEARCH_DIR = os.path.join(BASE_DIR, "search")
RECOMMENDATION_DIR = os.path.join(BASE_DIR, "recommendation")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")
API_DIR = os.path.join(BASE_DIR, "api")
UI_DIR = os.path.join(BASE_DIR, "ui")

# Model configurations
ENCODER_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search configurations
TOP_K_CANDIDATES = 50
BATCH_SIZE = 256
