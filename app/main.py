from fastapi import FastAPI

from app.core.config import settings

app = FastAPI()


@app.get("/")
def root():
    return {
        "algorithm": settings.algorithm,
        "expiry": settings.access_token_expire_minutes,
    }