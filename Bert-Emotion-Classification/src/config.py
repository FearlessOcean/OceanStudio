from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent

RAW_DATASETS_DIR=ROOT_DIR / "data" / "raw"
PROCESSED_DATASETS_DIR=ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
PRE_TRAINED_DIR = ROOT_DIR / "pretrained"


SEQ_LEN=128
EMBEDDING_DIM=128
HIDDEN_SIZE=256
BATCH_SIZE=64
LEARNING_RATE=1e-3
EPOCH=30