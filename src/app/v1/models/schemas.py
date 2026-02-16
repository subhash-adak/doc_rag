from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class FileTypeEnum(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    XLSX = "XLSX"
    XLS = "XLS"

class DocumentStatusEnum(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: FileTypeEnum
    file_size: int
    page_count: Optional[int]
    word_count: Optional[int]
    chunk_count: Optional[int]
    status: DocumentStatusEnum
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    message: str

class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]

class UserStatsResponse(BaseModel):
    total_documents: int
    total_queries: int
    storage_used: int
    storage_used_mb: float
    last_activity: Optional[datetime]
    
    class Config:
        from_attributes = True

class ChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessageCreate(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = None
    use_reranking: Optional[bool] = Field(default=True)

class ChatMessageResponse(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    sources: Optional[List[dict]] = None
    reranked: bool = False
    response_time_ms: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None
    
    class Config:
        from_attributes = True

class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    total: int
    has_more: bool
    cursor: Optional[str] = None

class ChatMessagesResponse(BaseModel):
    messages: List[ChatMessageResponse]
    total: int
    has_more: bool
    cursor: Optional[str] = None

class TitleUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)

class TitleGenerateResponse(BaseModel):
    title: str