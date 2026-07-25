from fastapi import FastAPI
from app.core.config import settings
from app.core.security import create_access_token
from app.api.router import api_router

app = FastAPI()
app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Study Companion API"}


@app.get("/token-test")
def token_test():
    token = create_access_token("booma@example.com")

    return {
        "token": token
    }
