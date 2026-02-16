from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.app.v1.models.schemas import UserCreate, UserLogin, Token, UserResponse
from src.app.v1.services.auth_service import AuthService, get_current_user
from src.app.v1.database.connection import get_db
from src.app.v1.models.database import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    
    - **email**: Valid email address
    - **password**: Minimum 6 characters
    - **full_name**: Optional full name
    """
    return await AuthService.register_user(user_data, db)


@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login with email and password
    
    Returns JWT access token
    """
    return await AuthService.login_user(credentials, db)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    
    Requires: Bearer token in Authorization header
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user
    
    Note: Since we're using stateless JWT, the client should delete the token
    """
    return {
        "message": "Successfully logged out. Please delete your token.",
        "user_id": current_user.user_id
    }