from sentence_transformers import SentenceTransformer
from src.app.v1.core.config import settings
from typing import List
import torch

class EmbeddingService:
    """
    Embedding service using sentence-transformers (FREE, LOCAL)
    Model: all-MiniLM-L6-v2 (384 dimensions)
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to load model only once"""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model (loads only once)"""
        if EmbeddingService._model is None:
            print(f"🔄 Loading embedding model: {settings.EMBEDDING_MODEL}")
            
            # Use CPU for compatibility (GPU if available)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            EmbeddingService._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device
            )
            
            print(f"✅ Embedding model loaded on {device}")
            print(f"📊 Dimension: {settings.EMBEDDING_DIMENSION}")
    
    @property
    def model(self):
        """Get the loaded model"""
        return EmbeddingService._model
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            # Encode text to embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embedding.tolist()
        
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Create embeddings for multiple texts (faster than one-by-one)
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        try:
            print(f"🧮 Creating embeddings for {len(texts)} texts...")
            
            # Encode all texts in batches
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True
            )
            
            print(f"✅ Created {len(embeddings)} embeddings")
            
            return embeddings.tolist()
        
        except Exception as e:
            print(f"❌ Error creating batch embeddings: {e}")
            raise
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for query
        (Same as create_embedding for this model)
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.create_embedding(query)