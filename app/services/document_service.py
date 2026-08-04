from sqlalchemy.orm import Session

from app.models.document import Document


def create_document(
    db: Session,
    filename: str,
    file_type: str,
    file_path: str,
    file_size: int,
    owner_id: int,
) -> Document:

    document = Document(
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        file_size=file_size,
        owner_id=owner_id,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document

def get_user_documents(db: Session, owner_id: int,):
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )