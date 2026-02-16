from pinecone import Pinecone, ServerlessSpec
from src.app.v1.core.config import settings
from typing import List, Dict, Optional
import time

class PineconeService:
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.namespace = settings.PINECONE_NAMESPACE  # Single namespace: "default"
        
        # Create index if doesn't exist
        self._initialize_index()
        
        self.index = self.pc.Index(self.index_name)
    
    def _initialize_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIMENSION,  # 4096 for llama-text-embed-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                print("Waiting for index to be ready...")
                time.sleep(1)
            
            print(f"✅ Index {self.index_name} created successfully")
        else:
            print(f"✅ Index {self.index_name} already exists")
    
    def upsert_chunks(self, user_id: str, chunks_data: List[Dict]):
        """
        Insert document chunks with embeddings
        All users in same namespace, isolated by user_id in metadata
        
        chunks_data format:
        [
            {
                "chunk_id": "...",
                "document_id": "...",
                "chunk_index": 0,
                "embedding": [...],
                "metadata": {...}
            }
        ]
        """
        vectors = []
        
        for chunk in chunks_data:
            # Create unique vector ID with user_id prefix
            vector_id = f"{user_id}_{chunk['document_id']}_chunk_{chunk['chunk_index']}"
            
            vectors.append({
                "id": vector_id,
                "values": chunk["embedding"],
                "metadata": {
                    **chunk["metadata"],
                    "user_id": user_id  # ⭐ CRITICAL: User isolation key
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace  # ⭐ Same namespace for all users
                )
                print(f"✅ Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
                raise
    
    def query_similar(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ):
        """
        Query similar chunks
        Filter by user_id to ensure data isolation
        """
        # Build filter - MUST filter by user_id
        filter_dict = {"user_id": {"$eq": user_id}}  # ⭐ User isolation
        
        # Optional: Filter by specific documents
        if document_ids and len(document_ids) > 0:
            filter_dict["document_id"] = {"$in": document_ids}
        
        try:
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,  # ⭐ Critical for multi-tenancy
                include_metadata=True
            )
            
            return results.matches
        
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            raise
    
    def delete_document(self, user_id: str, document_id: str):
        """Delete all chunks of a specific document"""
        try:
            self.index.delete(
                namespace=self.namespace,
                filter={
                    "user_id": {"$eq": user_id},
                    "document_id": {"$eq": document_id}
                }
            )
            print(f"✅ Deleted document {document_id} for user {user_id}")
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            raise
    
    def delete_user_data(self, user_id: str):
        """Delete all data for a user"""
        try:
            self.index.delete(
                namespace=self.namespace,
                filter={"user_id": {"$eq": user_id}}
            )
            print(f"✅ Deleted all data for user {user_id}")
        except Exception as e:
            print(f"❌ Error deleting user data: {e}")
            raise
    
    def get_index_stats(self) -> dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}