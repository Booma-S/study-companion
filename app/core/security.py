from datetime import UTC, datetime, timedelta

from jose import jwt

from app.core.config import settings


def create_access_token(subject: str) -> str:
    """
    Create a JWT access token.
    """

    expire = datetime.now(UTC) + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    payload = {
        "sub": subject,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )

from jose import JWTError


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.
    """

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )

        return payload

    except JWTError:
        return {}