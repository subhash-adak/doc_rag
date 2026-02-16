from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from typing import Optional

from src.app.v1.models.schemas import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatMessagesResponse,
    TitleUpdateRequest,
    TitleGenerateResponse
)
from src.app.v1.models.database import User
from src.app.v1.services.auth_service import get_current_user
from src.app.v1.services.chat_service import ChatService
from src.app.v1.database.connection import get_db
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/message", response_model=dict)
@limiter.limit("3/minute")
async def send_message(
    request: Request,
    message_data: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send message in chat session"""
    chat_service = ChatService()
    
    try:
        result = await chat_service.send_message(
            user_id=current_user.user_id,
            session_id=message_data.session_id,
            message=message_data.message,
            document_ids=message_data.document_ids,
            use_reranking=message_data.use_reranking,
            db=db
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    return {
        "session_id": result["session_id"],
        "user_message": ChatMessageResponse.from_orm(result["user_message"]),
        "assistant_message": ChatMessageResponse.from_orm(result["assistant_message"])
    }

@router.get("/sessions", response_model=ChatSessionListResponse)
async def get_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's chat sessions with cursor pagination"""
    chat_service = ChatService()
    
    result = chat_service.get_sessions(
        user_id=current_user.user_id,
        limit=limit,
        cursor=cursor,
        db=db
    )
    
    return ChatSessionListResponse(
        sessions=[ChatSessionResponse.from_orm(s) for s in result["sessions"]],
        total=result["total"],
        has_more=result["has_more"],
        cursor=result["cursor"]
    )

@router.get("/sessions/{session_id}/messages", response_model=ChatMessagesResponse)
async def get_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages in a session with cursor pagination"""
    chat_service = ChatService()
    
    result = chat_service.get_messages(
        user_id=current_user.user_id,
        session_id=session_id,
        limit=limit,
        cursor=cursor,
        db=db
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ChatMessagesResponse(
        messages=[ChatMessageResponse.from_orm(m) for m in result["messages"]],
        total=result["total"],
        has_more=result["has_more"],
        cursor=result["cursor"]
    )

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete chat session"""
    chat_service = ChatService()
    
    deleted = chat_service.delete_session(
        user_id=current_user.user_id,
        session_id=session_id,
        db=db
    )
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}

@router.put("/sessions/{session_id}/title")
async def update_title(
    session_id: str,
    request: TitleUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update session title"""
    chat_service = ChatService()
    
    updated = chat_service.update_title(
        user_id=current_user.user_id,
        session_id=session_id,
        title=request.title,
        db=db
    )
    
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Title updated successfully", "title": request.title}