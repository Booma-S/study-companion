from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListItemResponse(DocumentResponse):
    pass


DocumentListResponse = List[DocumentListItemResponse]


class DocumentTextResponse(BaseModel):
    document_id: int
    filename: str
    text: str
