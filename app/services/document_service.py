from sqlalchemy.orm import Session

from app.models.document import Document


def create_document(
    db: Session,
    filename: str,
    file_type: str,
    file_path: str,
    file_size: int,
    owner_id: int,
    extracted_text: str | None = None,
) -> Document:
    document = Document(
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        file_size=file_size,
        owner_id=owner_id,
        extracted_text=extracted_text,
    )
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    except Exception:
        db.rollback()
        raise


def get_user_documents(db: Session, owner_id: int) -> list[Document]:
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )


def get_document_by_id(db: Session, document_id: int) -> Document | None:
    return (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )


def update_document_text(db: Session, document: Document, extracted_text: str) -> Document:
    try:
        document.extracted_text = extracted_text
        db.commit()
        db.refresh(document)
        return document
    except Exception:
        db.rollback()
        raise


def delete_document(db: Session, document: Document) -> None:
    try:
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()
        raise
