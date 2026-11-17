from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentTextResponse,
)
from app.services.document_service import (
    create_document,get_document_by_id,get_user_documents,delete_document,)

from app.services.text_extraction_service import extract_text
from app.services.text_cleaning_service import clean_extracted_text

router = APIRouter()

UPLOAD_DIR = Path("uploads/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
}


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user),):
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type.",
        )

    unique_filename = f"{uuid.uuid4()}{extension}"

    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_text(
            str(file_path),
            extension.lstrip("."),
        )
        extracted_text = clean_extracted_text(extracted_text)
    except FileNotFoundError:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded file could not be found.",
        )

    except ValueError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    document = create_document(
        db=db,
        filename=file.filename,
        file_type=extension.lstrip("."),
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        owner_id=current_user.id,
        extracted_text=extracted_text,
    )
    return document

@router.get("",response_model=DocumentListResponse,summary="Get all user documents",)
def list_documents(
    db: Session = Depends(get_db),current_user: User = Depends(get_current_user),):
    return get_user_documents(db, current_user.id,)

@router.get("/{document_id}/download",summary="Download a document",)
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(
        db,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this document.",
        )

    file_path = Path(document.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server.",
        )

    return FileResponse(
        path=file_path,
        filename=document.filename,
        media_type="application/octet-stream",
    )

@router.get(
    "/{document_id}/text",
    response_model=DocumentTextResponse,
    summary="Extract text from a document",)
def extract_document_text(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(
        db,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this document.",
        )

    # If we already extracted and stored the text at upload time,
    # reuse it instead of re-running (slow) OCR on every request.
    if document.extracted_text is not None:
        return DocumentTextResponse(
            document_id=document.id,
            filename=document.filename,
            text=document.extracted_text,
        )

    try:
        text = extract_text(
            document.file_path,
            document.file_type,
        )
        text = clean_extracted_text(text)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return DocumentTextResponse(
        document_id=document.id,
        filename=document.filename,
        text=text,
    )

@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a document",)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    document = get_document_by_id(
        db,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this document.",
        )

    return document

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
def remove_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(
        db,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this document.",
        )

    file_path = Path(document.file_path)

    if file_path.is_file():
        file_path.unlink()

    delete_document(db, document)