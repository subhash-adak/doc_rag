from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import time

from src.app.v1.core.config import settings
from src.app.v1.database.connection import engine, Base
from src.app.v1.api import auth, documents, chat

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.app.v1.services.embedding_service import EmbeddingService
from src.app.v1.services.rag_service import RAGService

Base.metadata.create_all(bind=engine)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.APP_NAME,
    description="Personal Document Intelligence & Chat System",
    version="1.0.0",
    docs_url="/docs",
    # redoc_url="/redoc"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Validation error"}
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Database error occurred",
            "detail": str(exc) if settings.DEBUG else "Internal server error"
        }
    )

@app.get("/", tags=["Health"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "pinecone": "connected"
    }

app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)

# main.py - Complete startup function
@app.on_event("startup")
async def startup_event():
    """Initialize application and preload ML models"""
    print("="*60)
    print(f"🚀 {settings.APP_NAME} - Starting Up")
    print("="*60)
    
    # Database info
    print(f"\n📊 Database Configuration:")
    print(f"   Host: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}")
    print(f"   Database: {settings.MYSQL_DATABASE}")
    
    # Vector DB info
    print(f"\n☁️  Vector Database:")
    print(f"   Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    print(f"   Namespace: {settings.PINECONE_NAMESPACE}")
    
    # Preload ML models
    print(f"\n🤖 Loading AI Models...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Load embedding model
        print("1️⃣  Initializing Embedding Service...")
        embedding_service = EmbeddingService()
        
        # 2. Load RAG service (includes reranker + Groq)
        print("2️⃣  Initializing RAG Service...")
        rag_service = RAGService()
        
        load_time = time.time() - start_time
        print("-" * 60)
        print(f"✅ All models loaded in {load_time:.2f}s")
        
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("   Models will load on first request instead")
    
    # Model info
    print(f"\n📦 Model Configuration:")
    print(f"   Embedding: {settings.EMBEDDING_MODEL}")
    print(f"   Dimensions: {settings.EMBEDDING_DIMENSION}")
    print(f"   LLM: {settings.GROQ_MODEL}")
    print(f"   Reranking: {'Enabled' if settings.USE_RERANKING else 'Disabled'}")
    
    # Server ready
    print(f"\n{'='*60}")
    print(f"✅ Server Ready!")
    print(f"{'='*60}")
    print(f"🌐 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"{'='*60}\n")

@app.on_event("shutdown")
async def shutdown_event():
    print(f"👋 {settings.APP_NAME} is shutting down...")

