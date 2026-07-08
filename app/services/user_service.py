from sqlalchemy.orm import Session
from app.models.user import User
from app.core.hashing import hash_password
from app.schemas.user import UserCreate


def get_user_by_email(db: Session, email: str) -> User | None:
    """
    Retrieve a user by email address.
    """
    return (
        db.query(User)
        .filter(User.email == email)
        .first()
    )


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """
    Retrieve a user by ID.
    """
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

def create_user(db: Session, user_data: UserCreate) -> User:
    """
    Create a new user.
    """

    user = User(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user