from fastapi import FastAPI
from app.core.config import settings
from app.api.router import api_router

app = FastAPI()
app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Study Companion API"}

