from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.exceptions.auth import EmailAlreadyExistsError

from app.db.database import get_db
from app.schemas.user import UserCreate, UserResponse
from app.services.auth_service import register_user

from app.schemas.auth import LoginRequest
from app.schemas.token import Token
from app.services.auth_service import login_user
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Creates a new user account after validating the input and hashing the password.",
    responses={
        201: {"description": "User registered successfully"},
        409: {"description": "Email already registered"},
    },
)
def register(user_data: UserCreate, db: Session = Depends(get_db),):
    try:
        return register_user(db, user_data)
    except EmailAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    
@router.post(
    "/login",
    response_model=Token,
    summary="Login user",
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    try:
        login_data = LoginRequest(
            email=form_data.username,
            password=form_data.password,
        )

        return login_user(db, login_data)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )