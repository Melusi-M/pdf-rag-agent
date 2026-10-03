from pathlib import Path

import pytest

from pdf_rag_agent.config import AppConfig
from pdf_rag_agent.uploads import (
    sanitize_pdf_filename,
    store_uploaded_pdf,
    validate_pdf_content,
)


def test_sanitize_pdf_filename_removes_path_components() -> None:
    result = sanitize_pdf_filename(
        "../../private/lease.pdf"
    )

    assert result == "lease.pdf"


def test_sanitize_pdf_filename_handles_windows_path() -> None:
    result = sanitize_pdf_filename(
        r"C:\Users\Someone\contract.pdf"
    )

    assert result == "contract.pdf"


def test_validate_pdf_content_rejects_non_pdf() -> None:
    with pytest.raises(
        ValueError,
        match="does not appear to be a PDF",
    ):
        validate_pdf_content(b"This is not a PDF")


def test_store_uploaded_pdf_is_idempotent(
    tmp_path: Path,
) -> None:
    config = AppConfig.from_project_root(tmp_path)
    content = b"%PDF-1.4\nexample"

    first_result = store_uploaded_pdf(
        original_filename="lease.pdf",
        content=content,
        config=config,
    )

    second_result = store_uploaded_pdf(
        original_filename="lease.pdf",
        content=content,
        config=config,
    )

    assert first_result.already_exists is False
    assert second_result.already_exists is True
    assert first_result.document_id == second_result.document_id
    assert first_result.stored_path == second_result.stored_path
    assert first_result.stored_path.read_bytes() == content