import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data'
DATA_PATH.mkdir(exist_ok=True) 

# Birth and chart
BIRTHS_DIR = DATA_PATH / 'births'
CHARTS_DIR = DATA_PATH / 'charts'

BATCH_FILENAME = "batch_0004.jsonl"
BIRTH_PATH = BIRTHS_DIR / BATCH_FILENAME
CHART_PATH = CHARTS_DIR / BATCH_FILENAME

# Interpretations base dir
INTERPRETATIONS_DIR = DATA_PATH / "interpretations"
INTERPRETATIONS_DIR.mkdir(exist_ok=True)


# SFT base dir
SFT_DIR = DATA_PATH / "sft_dataset"
SFT_DIR.mkdir(exist_ok=True)

# RAG base dir
RAG_DIR = DATA_PATH / "rag_corpus"
RAG_DIR.mkdir(exist_ok=True)

RAG_INDEX_PATH = RAG_DIR / "faiss_index.bin"
RAG_DOCS_PATH = RAG_DIR / "docs.jsonl"
