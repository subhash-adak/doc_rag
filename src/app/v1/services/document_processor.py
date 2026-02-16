from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime

from src.app.v1.core.config import settings
from src.app.v1.models.database import Document, DocumentChunk, UserStats, DocumentStatus
from src.app.v1.utils.text_extractor import TextExtractor
from src.app.v1.services.embedding_service import EmbeddingService
from src.app.v1.services.pinecone_service import PineconeService

class DocumentProcessor:
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
    
    async def process_document(
        self,
        document_id: str,
        file_path: str,
        file_type: str,
        user_id: str,
        db: Session
    ):
        """
        Background task to process uploaded document
        1. Extract text
        2. Chunk text
        3. Create embeddings
        4. Store in Pinecone
        5. Update MySQL
        """
        try:
            print(f"📄 Processing document: {document_id}")
            
            # 1. Extract text
            text, doc_metadata = TextExtractor.extract_text(file_path, file_type)
            
            if not text or len(text.strip()) < 10:
                raise Exception("No text content extracted from document")
            
            # 2. Chunk text
            chunks = self.text_splitter.split_text(text)
            print(f"✂️  Created {len(chunks)} chunks")
            
            # 3. Create embeddings (batch)
            chunk_texts = chunks
            embeddings = self.embedding_service.create_embeddings_batch(chunk_texts)
            print(f"🧮 Created {len(embeddings)} embeddings")
            
            # 4. Prepare data for Pinecone
            chunks_data = []
            document = db.query(Document).filter(Document.document_id == document_id).first()
            
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                
                # Estimate page number
                page_number = self._estimate_page(idx, file_type, doc_metadata)
                
                # ⭐ Build metadata without None values (Pinecone requirement)
                metadata = {
                    "document_id": document_id,
                    "filename": document.filename,
                    "file_type": file_type,
                    "chunk_index": idx,
                    "chunk_text": chunk_text[:200],  # First 200 chars
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Only add page_number if it exists
                if page_number is not None:
                    metadata["page_number"] = page_number
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": idx,
                    "embedding": embedding,
                    "metadata": metadata
                })
                
                # Save chunk to MySQL (can have None)
                db_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=idx,
                    chunk_text=chunk_text,
                    chunk_size=len(chunk_text),
                    page_number=page_number,
                    pinecone_id=f"{user_id}_{document_id}_chunk_{idx}"
                )
                db.add(db_chunk)
            
            # 5. Upsert to Pinecone
            self.pinecone_service.upsert_chunks(user_id, chunks_data)
            print(f"☁️  Uploaded to Pinecone")
            
            # 6. Update document status
            document.status = DocumentStatus.COMPLETED
            document.page_count = doc_metadata.get('page_count')
            document.word_count = len(text.split())
            document.chunk_count = len(chunks)
            
            # 7. Update user stats
            user_stats = db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if user_stats:
                user_stats.total_documents += 1
                user_stats.storage_used += document.file_size
                user_stats.last_activity = datetime.utcnow()
            
            db.commit()
            print(f"✅ Document {document_id} processed successfully")
            
        except Exception as e:
            print(f"❌ Error processing document {document_id}: {e}")
            
            # Update document status to failed
            document = db.query(Document).filter(Document.document_id == document_id).first()
            if document:
                document.status = DocumentStatus.FAILED
                document.error_message = str(e)[:500]  # Truncate long error messages
                db.commit()
            
            raise
    
    def _estimate_page(self, chunk_idx: int, file_type: str, metadata: dict) -> int:
        """
        Estimate page number for chunk
        Returns None if page number cannot be determined
        """
        if file_type == 'PDF' and metadata.get('page_count'):
            chunks_per_page = 2
            estimated_page = min(chunk_idx // chunks_per_page + 1, metadata['page_count'])
            return estimated_page
        
        return None  # Return None for non-PDF or if page_count unavailable