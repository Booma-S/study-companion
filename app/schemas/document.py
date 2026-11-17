from datetime import datetime

from pydantic import BaseModel, ConfigDict
from typing import List

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    uploaded_at: datetime
    extracted_text: str | None = None

    model_config = ConfigDict(from_attributes=True)

DocumentListResponse = List[DocumentResponse]

class DocumentTextResponse(BaseModel):
    document_id: int
    filename: str
    text: str