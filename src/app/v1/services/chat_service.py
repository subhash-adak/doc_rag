from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import time
import re
from groq import Groq

from src.app.v1.core.config import settings
from src.app.v1.models.database import ChatSession, ChatMessage, User
from src.app.v1.services.rag_service import RAGService

class ChatService:
    
    def __init__(self):
        self.rag_service = RAGService()
        try:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
            print("✅ Groq client initialized")
        except Exception as e:
            print(f"❌ Groq client failed: {e}")
            self.groq_client = None
    
    def create_session(self, user_id: str, db: Session) -> ChatSession:
        """Create new chat session"""
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            title="New Chat"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        print(f"✅ Created session {session.session_id} with title: {session.title}")
        return session
    
    def get_or_create_session(self, user_id: str, session_id: Optional[str], db: Session) -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if session:
                print(f"✅ Found existing session: {session.session_id}")
            return session
        
        print("🔵 Creating new session...")
        return self.create_session(user_id, db)
    
    async def send_message(
        self,
        user_id: str,
        session_id: Optional[str],
        message: str,
        document_ids: Optional[List[str]],
        use_reranking: bool,
        db: Session
    ) -> Dict:
        """Send message and get response"""
        
        # Get or create session
        session = self.get_or_create_session(user_id, session_id, db)
        
        if session_id and not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Capture timestamp for user message
        user_message_time = datetime.utcnow()
        
        # Save user message with explicit timestamp
        user_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session.session_id,
            user_id=user_id,
            role="user",
            content=message,
            created_at=user_message_time
        )
        db.add(user_message)
        db.flush()
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)
        
        # Get RAG response
        rag_response = await self.rag_service.query(
            user_id=user_id,
            query_text=message,
            document_ids=document_ids,
            top_k=5,
            use_reranking=use_reranking,
            db=db
        )
        
        # Capture timestamp for assistant message
        assistant_message_time = datetime.utcnow()
        if assistant_message_time <= user_message_time:
            assistant_message_time = user_message_time + timedelta(microseconds=1)
        
        # Save assistant message with explicit timestamp
        assistant_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session.session_id,
            user_id=user_id,
            role="assistant",
            content=rag_response['answer'],
            sources=rag_response.get('sources'),
            reranked=rag_response.get('reranked', False),
            response_time_ms=rag_response.get('response_time_ms'),
            created_at=assistant_message_time
        )
        db.add(assistant_message)
        
        # Update session message_count
        session.message_count += 2
        
        # ⭐ Generate title if first message
        generated_title = None
        if session.message_count == 2:
            print(f"🔵 First message in session, generating title...")
            generated_title = self.generate_title(message)
            print(f"✅ Generated title: '{generated_title}'")
            session.title = generated_title
        
        # Commit all changes
        db.commit()
        db.refresh(user_message)
        db.refresh(assistant_message)
        db.refresh(session)
        
        print(f"✅ Session saved with title: '{session.title}'")
        
        # ⭐ RETURN TITLE IN RESPONSE
        return {
            "session_id": session.session_id,
            "session_title": session.title,  # ⭐ NEW: Return current title
            "title_updated": generated_title is not None,  # ⭐ NEW: Flag if title was just generated
            "user_message": user_message,
            "assistant_message": assistant_message
        }
    
    def get_sessions(
        self,
        user_id: str,
        limit: int,
        cursor: Optional[str],
        db: Session
    ) -> Dict:
        """Get user's chat sessions with cursor pagination"""
        
        query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
        
        if cursor:
            try:
                cursor_dt = datetime.fromisoformat(cursor)
                query = query.filter(ChatSession.updated_at < cursor_dt)
            except ValueError:
                pass
        
        sessions = query.order_by(ChatSession.updated_at.desc()).limit(limit + 1).all()
        
        has_more = len(sessions) > limit
        if has_more:
            sessions = sessions[:limit]
        
        for session in sessions:
            last_msg = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.session_id
            ).order_by(ChatMessage.created_at.desc()).first()
            session.last_message = last_msg.content[:100] if last_msg else None
        
        next_cursor = sessions[-1].updated_at.isoformat() if sessions and has_more else None
        
        return {
            "sessions": sessions,
            "total": db.query(ChatSession).filter(ChatSession.user_id == user_id).count(),
            "has_more": has_more,
            "cursor": next_cursor
        }
    
    def get_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int,
        cursor: Optional[str],
        db: Session
    ) -> Optional[Dict]:
        """Get session messages with cursor pagination"""
        
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            return None
        
        query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
        
        if cursor:
            try:
                cursor_dt = datetime.fromisoformat(cursor)
                query = query.filter(ChatMessage.created_at < cursor_dt)
            except ValueError:
                pass
        
        messages = query.order_by(ChatMessage.created_at.desc()).limit(limit + 1).all()
        
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        
        messages.reverse()
        
        next_cursor = messages[0].created_at.isoformat() if messages and has_more else None
        
        return {
            "messages": messages,
            "total": db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count(),
            "has_more": has_more,
            "cursor": next_cursor
        }
    
    def delete_session(self, user_id: str, session_id: str, db: Session) -> bool:
        """Delete chat session and all messages"""
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            return False
        
        db.delete(session)
        db.commit()
        return True
    
    def update_title(self, user_id: str, session_id: str, title: str, db: Session) -> bool:
        """Update session title"""
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            return False
        
        session.title = title
        db.commit()
        return True
    
    def generate_title(self, first_message: str) -> str:
        """
        Generate concise title from first message using LLM
        Falls back to smart extraction if LLM fails
        """
        print(f"🔵 Generating title from: '{first_message[:60]}...'")
        
        # Try LLM first
        if self.groq_client:
            try:
                print("🔵 Calling Groq API...")
                
                response = self.groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a title generator. Create short, clear titles (3-5 words max) that capture the main topic. Return ONLY the title, no quotes, no markdown, no preamble."
                        },
                        {
                            "role": "user",
                            "content": f"""Create a title for this question:
"{first_message[:200]}"

Examples:
"What is machine learning?" → Machine Learning Overview
"How do I fix my car?" → Car Repair Help
"Explain quantum physics" → Quantum Physics
"What's in my resume?" → Resume Review
"Hi / Hello / Good morning / How are you etc" → Greeting
Title:"""
                        }
                    ],
                    temperature=0.3,
                    max_tokens=15,
                    top_p=0.9
                )
                
                raw_title = response.choices[0].message.content.strip()
                print(f"🔵 Raw Groq response: '{raw_title}'")
                
                # ⭐ Clean the title
                cleaned_title = self._clean_title(raw_title)
                print(f"✅ Cleaned title: '{cleaned_title}'")
                
                # Validate
                if cleaned_title and len(cleaned_title) >= 3:
                    return cleaned_title[:100]
                else:
                    print("⚠️  Invalid title from Groq, using fallback")
                    
            except Exception as e:
                print(f"⚠️  Groq failed: {e}, using fallback")
        
        # Fallback: Smart extraction
        return self._extract_title_fallback(first_message)
    
    def _clean_title(self, title: str) -> str:
        """
        Clean LLM-generated title
        Removes markdown, quotes, common prefixes, etc.
        """
        if not title:
            return ""
        
        # Remove markdown formatting
        title = re.sub(r'\*\*', '', title)  # Remove **bold**
        title = re.sub(r'__', '', title)    # Remove __bold__
        title = re.sub(r'\*', '', title)    # Remove *italic*
        title = re.sub(r'_', '', title)     # Remove _italic_
        title = re.sub(r'#+\s*', '', title) # Remove ## headers
        title = re.sub(r'`', '', title)     # Remove `code`
        
        # Remove quotes
        title = title.strip('"\'""''')
        
        # Remove common prefixes
        prefixes_to_remove = [
            'title:', 'subject:', 'topic:', 'summary:',
            'Title:', 'Subject:', 'Topic:', 'Summary:',
            'TITLE:', 'SUBJECT:', 'TOPIC:', 'SUMMARY:'
        ]
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix):].strip()
        
        # Remove trailing punctuation except question marks
        title = re.sub(r'[.!,;:]+$', '', title)
        
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        
        return title.strip()
    
    def _extract_title_fallback(self, message: str) -> str:
        """
        Fallback method: Extract title from message
        Smart keyword extraction without common words
        """
        print("🔵 Using fallback title extraction")
        
        # Clean the message
        clean = message.strip()
        
        # If it's a question, use the question (without question words)
        if '?' in clean[:100]:
            question = clean.split('?')[0]
            # Remove question words
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'could', 'would', 'should', 'will']
            words = question.lower().split()
            filtered_words = [w for w in words if w not in question_words]
            
            if filtered_words:
                title = ' '.join(filtered_words[:6]).title()
                if len(title) >= 3:
                    return title[:80]
        
        # Otherwise, extract meaningful words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can',
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you', 'your'
        }
        
        # Extract words
        words = clean.lower().split()[:15]
        keywords = [w.strip('?.!,;:') for w in words if w.strip('?.!,;:') not in stop_words]
        
        if keywords:
            # Take first 5 keywords
            title = ' '.join(keywords[:5]).title()
            return title[:80]
        
        # Last resort: just truncate
        return clean[:50] + ('...' if len(clean) > 50 else '')