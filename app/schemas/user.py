from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        examples=["Booma"]
    )

    email: EmailStr = Field(
        ...,
        examples=["booma@example.com"]
    )


class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        examples=["StrongPassword123"]
    )


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)