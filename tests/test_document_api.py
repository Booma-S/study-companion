import io
import zipfile
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.core.security import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.models.document import Document
from app.schemas.document import DocumentListItemResponse
from app.services.document_service import update_document_text

client = TestClient(app)


def test_token_test_removed():
    """Verify that the insecure /token-test endpoint has been removed."""
    response = client.get("/token-test")
    assert response.status_code == 404


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Study Companion API"}


def test_document_list_item_schema_excludes_extracted_text():
    """Verify DocumentListItemResponse only contains metadata, not extracted_text."""
    fields = DocumentListItemResponse.model_fields.keys()
    assert "extracted_text" not in fields
    assert "id" in fields
    assert "filename" in fields
    assert "file_size" in fields


def test_upload_rejects_oversized_file():
    """Verify that files exceeding 50 MB are rejected with HTTP 413."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    # Generate a dummy stream that exceeds 50 MB (e.g. 51 MB)
    large_size = 51 * 1024 * 1024
    large_stream = io.BytesIO(b"0" * large_size)

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("large_file.pdf", large_stream, "application/pdf")},
        )
        assert response.status_code == 413
        assert "exceeds maximum allowed size" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_upload_rejects_fake_docx_archive():
    """Verify that a plain zip file missing word/document.xml is rejected with HTTP 400."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    # Create a zip file that does not contain word/document.xml
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("some_other_file.txt", "hello")
    zip_buffer.seek(0)

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("fake_doc.docx", zip_buffer, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert response.status_code == 400
        assert "missing required" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_upload_rejects_fake_pptx_archive():
    """Verify that a plain zip file missing ppt/presentation.xml is rejected with HTTP 400."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("some_other_file.txt", "hello")
    zip_buffer.seek(0)

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("fake_pres.pptx", zip_buffer, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
        )
        assert response.status_code == 400
        assert "missing required" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_update_document_text_persists():
    """Verify that update_document_text assigns extracted text and commits to db."""
    mock_db = MagicMock()
    mock_doc = Document(id=1, filename="test.pdf", extracted_text=None)

    updated = update_document_text(mock_db, mock_doc, "Extracted test content")

    assert updated.extracted_text == "Extracted test content"
    assert mock_db.commit.called
    assert mock_db.refresh.called


def test_upload_rejects_empty_file():
    """Verify that an empty (0 byte) file is rejected with HTTP 400."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_upload_rejects_invalid_pdf_signature():
    """Verify that a non-PDF file with .pdf extension fails magic signature validation with HTTP 400."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("fake.pdf", io.BytesIO(b"This is not a real PDF file content"), "application/pdf")},
        )
        assert response.status_code == 400
        assert "valid pdf" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_upload_rejects_zip_bomb_entry_count():
    """Verify that an archive with > 10,000 entries is rejected with HTTP 400."""
    mock_user = User(id=1, email="test@example.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")
        for i in range(10001):
            zf.writestr(f"file_{i}.txt", "x")
    zip_buffer.seek(0)

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("bomb.docx", zip_buffer, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert response.status_code == 400
        assert "too many entries" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_cross_user_access_forbidden():
    """Verify that a user accessing another user's document gets HTTP 403."""
    mock_user = User(id=1, email="user1@example.com")
    mock_db = MagicMock()
    mock_doc = Document(id=42, filename="secret.pdf", owner_id=2, file_path="uploads/secret.pdf", extracted_text="secret")

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("app.api.v1.documents.get_document_by_id", return_value=mock_doc):
        try:
            resp = client.get("/api/v1/documents/42")
            assert resp.status_code == 403
            assert "permission" in resp.json()["detail"].lower()

            resp = client.get("/api/v1/documents/42/download")
            assert resp.status_code == 403

            resp = client.get("/api/v1/documents/42/text")
            assert resp.status_code == 403

            resp = client.delete("/api/v1/documents/42")
            assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()


def test_fallback_extraction_caches_text():
    """Verify that fallback extraction updates the DB and caches text for subsequent calls."""
    mock_user = User(id=1, email="user1@example.com")
    mock_db = MagicMock()
    mock_doc = Document(
        id=10,
        filename="test.pdf",
        owner_id=1,
        file_path="uploads/test.pdf",
        file_type="pdf",
        extracted_text=None,
    )

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("app.api.v1.documents.get_document_by_id", return_value=mock_doc), \
         patch("app.api.v1.documents.extract_text", return_value="Newly extracted text") as mock_extract, \
         patch("app.api.v1.documents.update_document_text") as mock_update:

        try:
            # First call: extracted_text is None, so extract_text and update_document_text are called
            resp = client.get("/api/v1/documents/10/text")
            assert resp.status_code == 200
            assert resp.json()["text"] == "Newly extracted text"
            assert mock_extract.called
            assert mock_update.called

            # Simulate database cache update
            mock_doc.extracted_text = "Newly extracted text"
            mock_extract.reset_mock()

            # Second call: extracted_text is cached, extract_text should NOT be called
            resp = client.get("/api/v1/documents/10/text")
            assert resp.status_code == 200
            assert resp.json()["text"] == "Newly extracted text"
            assert not mock_extract.called
        finally:
            app.dependency_overrides.clear()

