import os

class Config:
    DB_PATH = os.getenv("DB_PATH", "memories.db")
    
    # Increased threshold to filter out weak false positives
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
    
    DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.65"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.35"))
    CONTRADICTION_SIM_THRESHOLD = 0.70
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_EMBED_MODEL = "nomic-embed-text"
    OLLAMA_LLM_MODEL = "llama3.2:1b"