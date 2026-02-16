from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str 
    DEBUG: bool 
    API_V1_PREFIX: str 
    
    # MySQL
    MYSQL_HOST: str
    MYSQL_PORT: int 
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DATABASE: str
    
    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    # JWT
    SECRET_KEY: str
    ALGORITHM: str 
    ACCESS_TOKEN_EXPIRE_MINUTES: int 
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str 
    PINECONE_INDEX_NAME: str 
    PINECONE_NAMESPACE: str 
    
    # Embedding 
    EMBEDDING_MODEL: str 
    EMBEDDING_DIMENSION: int
    
    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str 
    
    # File Upload
    UPLOAD_DIR: str 
    MAX_FILE_SIZE_MB: int 
    ALLOWED_EXTENSIONS: str 
    
    @property
    def MAX_FILE_SIZE(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def ALLOWED_EXTENSIONS_LIST(self) -> List[str]:
        return self.ALLOWED_EXTENSIONS.split(',')
    
    # Chunking
    CHUNK_SIZE: int 
    CHUNK_OVERLAP: int 

    # Reranking
    USE_RERANKING: bool 
    RERANKING_MODEL: str 
    RERANKING_TOP_K_MULTIPLIER: int 
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Create upload directory if not exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)