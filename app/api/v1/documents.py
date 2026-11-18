import logging
import uuid
import zipfile
from pathlib import Path

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

from app.core.config import settings
from app.core.security import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentTextResponse,
)
from app.services.document_service import (
    create_document,
    delete_document,
    get_document_by_id,
    get_user_documents,
    update_document_text,
)
from app.services.text_cleaning_service import clean_extracted_text
from app.services.text_extraction_service import extract_text

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(getattr(settings, "upload_dir", "uploads/documents"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_UNCOMPRESSED_SIZE = 250 * 1024 * 1024  # 250 MB


def _validate_file_integrity(file_path: Path, extension: str) -> None:
    """
    Validate that the file begins with expected signatures and that
    Office Open XML formats (DOCX/PPTX) are genuine, valid archives.
    """
    if extension == ".pdf":
        try:
            with file_path.open("rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    raise ValueError("File is not a valid PDF document.")
        except Exception as exc:
            raise ValueError(f"Invalid PDF file: {exc}") from exc

    elif extension in {".docx", ".pptx"}:
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                # Zip-bomb guards
                infolist = zf.infolist()
                if len(infolist) > MAX_ZIP_ENTRIES:
                    raise ValueError("Archive contains too many entries.")
                total_size = sum(info.file_size for info in infolist)
                if total_size > MAX_ZIP_UNCOMPRESSED_SIZE:
                    raise ValueError("Archive uncompressed size is suspiciously large.")

                namelist = set(zf.namelist())
                if "[Content_Types].xml" not in namelist:
                    raise ValueError("Archive is missing required [Content_Types].xml.")

                if extension == ".docx" and "word/document.xml" not in namelist:
                    raise ValueError("Archive is missing required word/document.xml.")

                if extension == ".pptx" and "ppt/presentation.xml" not in namelist:
                    raise ValueError("Archive is missing required ppt/presentation.xml.")

        except zipfile.BadZipFile as exc:
            raise ValueError("File is not a valid ZIP-based Office Open XML archive.") from exc
        except Exception as exc:
            raise ValueError(f"Invalid Office document: {exc}") from exc


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Only PDF, DOCX, and PPTX are allowed.",
        )

    unique_filename = f"{uuid.uuid4()}{extension}"
    file_path = UPLOAD_DIR / unique_filename

    # 1. Bounded chunked streaming upload (rejects early if file exceeds MAX_FILE_SIZE)
    file_size = 0
    chunk_size = 1024 * 1024  # 1 MB
    try:
        with file_path.open("wb") as buffer:
            while chunk := file.file.read(chunk_size):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        file_path.unlink(missing_ok=True)
        logger.error("Failed to write uploaded file to disk: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        )

    if file_size == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 2. Comprehensive archive & format integrity validation
    try:
        _validate_file_integrity(file_path, extension)
    except ValueError as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # 3. Extract text and insert into database with failure cleanup
    try:
        extracted_text = extract_text(
            str(file_path),
            extension.lstrip("."),
        )
        extracted_text = clean_extracted_text(extracted_text)

        document = create_document(
            db=db,
            filename=file.filename or "unknown",
            file_type=extension.lstrip("."),
            file_path=str(file_path),
            file_size=file_size,
            owner_id=current_user.id,
            extracted_text=extracted_text,
        )
        return document

    except FileNotFoundError:
        file_path.unlink(missing_ok=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded file could not be found.",
        )
    except ValueError as exc:
        file_path.unlink(missing_ok=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        db.rollback()
        logger.error("Document processing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the document.",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="Get all user documents",
)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_user_documents(db, current_user.id)


@router.get(
    "/{document_id}/download",
    summary="Download a document",
)
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(db, document_id)

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
    summary="Extract text from a document",
)
def extract_document_text(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(db, document_id)

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

    # If already extracted and stored at upload time, reuse it
    if document.extracted_text is not None:
        return DocumentTextResponse(
            document_id=document.id,
            filename=document.filename,
            text=document.extracted_text,
        )

    # Fallback extraction: extract and cache in DB
    try:
        text = extract_text(
            document.file_path,
            document.file_type,
        )
        text = clean_extracted_text(text)
        update_document_text(db=db, document=document, extracted_text=text)

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
    except Exception as exc:
        db.rollback()
        logger.error("Fallback extraction failed for document %d: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract document text.",
        )

    return DocumentTextResponse(
        document_id=document.id,
        filename=document.filename,
        text=text,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document",
)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_document_by_id(db, document_id)

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
    document = get_document_by_id(db, document_id)

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

    # Delete from database first, then cleanup file
    file_path = Path(document.file_path)
    delete_document(db, document)

    if file_path.is_file():
        try:
            file_path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete physical file %s: %s", file_path, exc)
