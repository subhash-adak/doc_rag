

from groq import Groq
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import time
import uuid
from datetime import datetime
from sentence_transformers import CrossEncoder

from src.app.v1.core.config import settings
from src.app.v1.services.embedding_service import EmbeddingService
from src.app.v1.services.pinecone_service import PineconeService
from src.app.v1.models.database import UserStats

class RAGService:
    
    _reranker = None
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        
        # Initialize Groq client with error handling
        try:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Groq client: {e}")
            print("   Make sure GROQ_API_KEY is set in .env")
            self.groq_client = None
        
        if RAGService._reranker is None:
            try:
                print("🔄 Loading reranker model...")
                RAGService._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Reranker model loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load reranker: {e}")
                RAGService._reranker = None
    
    async def query(
        self,
        user_id: str,
        query_text: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_reranking: bool = True,
        db: Session = None
    ) -> Dict:
        start_time = time.time()
        
        try:
            query_embedding = self.embedding_service.create_query_embedding(query_text)
            
            # ⭐ Get more results if reranking is enabled
            search_k = top_k * 4 if (use_reranking and RAGService._reranker) else top_k
            
            matches = self.pinecone_service.query_similar(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=min(search_k, 20),
                document_ids=document_ids
            )
            
            if not matches or len(matches) == 0:
                return {
                    "query": query_text,
                    "answer": "I couldn't find any relevant information in your documents to answer this question.",
                    "sources": [],
                    "confidence": 0.0,
                    "reranked": False,
                    "response_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # ⭐ FIXED: Rerank if we have 3+ results and reranking is enabled
            reranked = False
            if use_reranking and RAGService._reranker and len(matches) >= 3:
                initial_count = len(matches)
                matches = self._rerank_results(query_text, matches, top_k)
                reranked = True
                print(f"✅ Reranked {initial_count} results down to {len(matches)}")
            else:
                # Just take top_k without reranking
                matches = matches[:top_k]
                if use_reranking:
                    print(f"⚠️  Reranking skipped: only {len(matches)} results (need 3+)")
            
            context_parts = []
            sources = []
            
            for match in matches:
                chunk_text = self._get_full_chunk_text(match, db)
                context_parts.append(chunk_text)
                
                sources.append({
                    "document_id": match.metadata['document_id'],
                    "filename": match.metadata['filename'],
                    "chunk_index": match.metadata['chunk_index'],
                    "page_number": match.metadata.get('page_number'),
                    "score": float(match.score),
                    "preview": match.metadata['chunk_text'][:150] + "..."
                })
            
            context = "\n\n---\n\n".join(context_parts)
            answer = self._generate_answer(query_text, context)
            
            top_scores = [match.score for match in matches[:3]]
            confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update user stats only
            if db:
                try:
                    user_stats = db.query(UserStats).filter(UserStats.user_id == user_id).first()
                    if user_stats:
                        user_stats.total_queries += 1
                        user_stats.last_activity = datetime.utcnow()
                        db.commit()
                except Exception as e:
                    print(f"⚠️  Could not update user stats: {e}")
                    db.rollback()
            
            return {
                "query": query_text,
                "answer": answer,
                "sources": sources,
                "confidence": float(confidence),
                "reranked": reranked,
                "response_time_ms": response_time_ms
            }
        
        except Exception as e:
            print(f"❌ Error in RAG query: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "query": query_text,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reranked": False,
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _rerank_results(self, query: str, matches: List, top_k: int) -> List:
        """Rerank results using CrossEncoder"""
        try:
            # Prepare query-document pairs
            pairs = [[query, match.metadata['chunk_text']] for match in matches]
            
            # Get reranking scores
            rerank_scores = RAGService._reranker.predict(pairs)
            
            # Combine matches with scores
            scored_matches = list(zip(matches, rerank_scores))
            
            # Sort by rerank score (descending)
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            return [match for match, score in scored_matches[:top_k]]
            
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}, falling back to original order")
            return matches[:top_k]
    
    def _get_full_chunk_text(self, match, db: Session) -> str:
        """Get full chunk text from MySQL (Pinecone only has 200 char preview)"""
        if db:
            try:
                from src.app.v1.models.database import DocumentChunk
                chunk = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == match.metadata['document_id'],
                    DocumentChunk.chunk_index == match.metadata['chunk_index']
                ).first()
                if chunk:
                    return chunk.chunk_text
            except Exception as e:
                print(f"⚠️  Could not fetch full chunk text: {e}")
        
        # Fallback to preview
        return match.metadata['chunk_text']
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq LLM"""
        if not self.groq_client:
            return "Error: Groq API client not initialized. Please check your GROQ_API_KEY."
        
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context.

Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the documents to answer that question."
3. Be concise and accurate
4. If you reference specific information, mention which document or section it came from
5. Use a professional but friendly tone"""

        user_prompt = f"""Context from documents:

{context}

---

Question: {query}

Please provide a clear and concise answer based on the context above."""

        try:
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error generating answer with Groq: {e}")
            return f"Error generating answer: {str(e)}"