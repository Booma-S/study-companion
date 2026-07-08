from sqlalchemy.orm import Session
from app.exceptions.auth import EmailAlreadyExistsError
from app.schemas.user import UserCreate
from app.schemas.auth import LoginRequest
from app.services.user_service import (
    create_user,
    get_user_by_email,
)


def register_user(db: Session, user_data: UserCreate):
    """
    Register a new user.
    """

    existing_user = get_user_by_email(
        db,
        user_data.email
    )

    if existing_user:
        raise EmailAlreadyExistsError("Email is already registered.")

    return create_user(
        db,
        user_data
    )