from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import uuid

from src.app.v1.core.config import settings
from src.app.v1.models.database import User, UserStats
from src.app.v1.models.schemas import UserCreate, UserLogin, Token, TokenData
from src.app.v1.database.connection import get_db

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token
security = HTTPBearer()

class AuthService:
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    async def register_user(user_data: UserCreate, db: Session) -> Token:
        # Check if user exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        user_id = str(uuid.uuid4())
        hashed_password = AuthService.get_password_hash(user_data.password)
        
        new_user = User(
            user_id=user_id,
            email=user_data.email,
            password_hash=hashed_password,
            full_name=user_data.full_name
        )
        
        db.add(new_user)
        
        # Create user stats
        user_stats = UserStats(user_id=user_id)
        db.add(user_stats)
        
        db.commit()
        db.refresh(new_user)
        
        # Create token
        access_token = AuthService.create_access_token(data={"sub": user_id})
        
        return Token(
            access_token=access_token,
            user_id=user_id,
            email=new_user.email
        )
    
    @staticmethod
    async def login_user(credentials: UserLogin, db: Session) -> Token:
        # Find user
        user = db.query(User).filter(User.email == credentials.email).first()
        if not user or not AuthService.verify_password(credentials.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create token
        access_token = AuthService.create_access_token(data={"sub": user.user_id})
        
        return Token(
            access_token=access_token,
            user_id=user.user_id,
            email=user.email
        )

# Dependency to get current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.user_id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    
    return user