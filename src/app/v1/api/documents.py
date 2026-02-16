from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import uuid
import os
import shutil

from src.app.v1.core.config import settings
from src.app.v1.models.schemas import (
    DocumentUploadResponse, 
    DocumentResponse, 
    DocumentListResponse,
    UserStatsResponse
)
from src.app.v1.models.database import User, Document, DocumentChunk, UserStats, DocumentStatus, FileType
from src.app.v1.services.auth_service import get_current_user
from src.app.v1.services.document_processor import DocumentProcessor
from src.app.v1.services.pinecone_service import PineconeService
from src.app.v1.database.connection import get_db
from src.app.v1.database.connection import SessionLocal

router = APIRouter(prefix="/documents", tags=["Documents"])

async def process_document_background(document_id, file_path, file_type, user_id):
    db = SessionLocal()
    try:
        processor = DocumentProcessor()
        await processor.process_document(document_id, file_path, file_type, user_id, db)
    finally:
        db.close()

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a document for processing
    
    Supported formats: PDF, DOCX, TXT, XLSX, XLS
    Max file size: 10MB
    
    The document will be processed in the background:
    1. Text extraction
    2. Chunking
    3. Embedding generation
    4. Storage in Pinecone
    """
    
    # 1. Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS_LIST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS_LIST)}"
        )
    
    # 2. Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()  # Get position (size)
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    # 3. Create document ID and file path
    document_id = str(uuid.uuid4())
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    file_path = os.path.join(user_upload_dir, f"{document_id}.{file_ext}")
    
    # 4. Save file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    
    # 5. Create document entry in database
    file_type_enum = FileType[file_ext.upper()]
    
    new_document = Document(
        document_id=document_id,
        user_id=current_user.user_id,
        filename=file.filename,
        file_type=file_type_enum,
        file_size=file_size,
        file_path=file_path,
        status=DocumentStatus.PROCESSING
    )
    
    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    
    background_tasks.add_task(
        process_document_background,
        document_id, file_path, file_ext.upper(), current_user.user_id
    )
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        file_type=file_ext.upper(),
        file_size=file_size,
        status="processing",
        message="Document uploaded successfully. Processing in background."
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    status_filter: str = None,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of user's documents
    
    - **status_filter**: Optional filter by status (processing, completed, failed)
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return (max 100)
    """
    
    if limit > 100:
        limit = 100
    
    # Build query
    query = db.query(Document).filter(Document.user_id == current_user.user_id)
    
    # Apply status filter if provided
    if status_filter:
        try:
            status_enum = DocumentStatus[status_filter.upper()]
            query = query.filter(Document.status == status_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid options: processing, completed, failed"
            )
    
    # Get total count
    total = query.count()
    
    # Get documents with pagination
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return DocumentListResponse(
        total=total,
        documents=documents
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific document
    """
    document = db.query(Document).filter(
        Document.document_id == document_id,
        Document.user_id == current_user.user_id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document
    
    This will:
    1. Delete from Pinecone
    2. Delete from MySQL
    3. Delete file from disk
    """
    
    # 1. Find document
    document = db.query(Document).filter(
        Document.document_id == document_id,
        Document.user_id == current_user.user_id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # 2. Delete from Pinecone
        pinecone_service = PineconeService()
        pinecone_service.delete_document(current_user.user_id, document_id)
        
        # 3. Delete chunks from MySQL (cascade will handle this, but explicit is better)
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # 4. Update user stats
        user_stats = db.query(UserStats).filter(UserStats.user_id == current_user.user_id).first()
        if user_stats:
            user_stats.total_documents = max(0, user_stats.total_documents - 1)
            user_stats.storage_used = max(0, user_stats.storage_used - document.file_size)
        
        # 5. Delete document from MySQL
        db.delete(document)
        db.commit()
        
        # 6. Delete file from disk
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/stats/me", response_model=UserStatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's statistics
    """
    user_stats = db.query(UserStats).filter(UserStats.user_id == current_user.user_id).first()
    
    if not user_stats:
        # Create if doesn't exist
        user_stats = UserStats(user_id=current_user.user_id)
        db.add(user_stats)
        db.commit()
        db.refresh(user_stats)
    
    return UserStatsResponse(
        total_documents=user_stats.total_documents,
        total_queries=user_stats.total_queries,
        storage_used=user_stats.storage_used,
        storage_used_mb=round(user_stats.storage_used / (1024 * 1024), 2),
        last_activity=user_stats.last_activity
    )