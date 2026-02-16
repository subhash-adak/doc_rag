from sqlalchemy import Column, String, Integer, BigInteger, Text, Enum, ForeignKey, JSON, Boolean
from sqlalchemy.sql import func, text
from sqlalchemy.dialects.mysql import DATETIME
from src.app.v1.database.connection import Base
import enum

class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class FileType(str, enum.Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    XLSX = "XLSX"
    XLS = "XLS"

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6)'))
    updated_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)'))

class Document(Base):
    __tablename__ = "documents"
    
    document_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_path = Column(Text, nullable=False)
    page_count = Column(Integer)
    word_count = Column(Integer)
    chunk_count = Column(Integer)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PROCESSING, index=True)
    error_message = Column(Text)
    created_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6)'), index=True)
    updated_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)'))

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    chunk_id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)
    page_number = Column(Integer)
    pinecone_id = Column(String(200), index=True)
    created_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6)'))

class UserStats(Base):
    __tablename__ = "user_stats"
    
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    total_documents = Column(Integer, default=0)
    total_queries = Column(Integer, default=0)
    storage_used = Column(BigInteger, default=0)
    last_activity = Column(DATETIME(fsp=6))

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500))
    message_count = Column(Integer, default=0)
    created_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6)'), index=True)
    updated_at = Column(DATETIME(fsp=6), server_default=text('CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)'), index=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    message_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON)
    reranked = Column(Boolean, default=False)
    response_time_ms = Column(Integer)
    created_at = Column(DATETIME(fsp=6), index=True)