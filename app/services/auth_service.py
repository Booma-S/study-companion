from sqlalchemy.orm import Session
from app.exceptions.auth import EmailAlreadyExistsError
from app.schemas.user import UserCreate
from app.schemas.auth import LoginRequest
from app.services.user_service import (create_user, get_user_by_email,)
from app.core.hashing import verify_password
from app.core.security import create_access_token
from app.services.user_service import get_user_by_email

# Register a new user.
def register_user(db: Session, user_data: UserCreate):
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise EmailAlreadyExistsError("Email is already registered.")
    return create_user(db, user_data)

# Authenticate a user and return a JWT access token.
def login_user(db, login_data):
    user = get_user_by_email(db, login_data.email)
    if not user:
        raise ValueError("Invalid email or password.")

    if not verify_password(login_data.password, user.hashed_password):
        raise ValueError("Invalid email or password.")
    
    access_token = create_access_token(user.email)
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }