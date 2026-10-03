import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pdf_rag_agent.config import AppConfig


MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    original_name: str
    stored_path: Path
    already_exists: bool


def sanitize_pdf_filename(filename: str) -> str:
    normalized_name = filename.replace("\\", "/")
    basename = Path(normalized_name).name.strip()

    if not basename:
        raise ValueError("Uploaded filename cannot be empty")

    path = Path(basename)

    if path.suffix.lower() != ".pdf":
        raise ValueError("Uploaded file must have a .pdf extension")

    sanitized_stem = re.sub(
        r"[^A-Za-z0-9._ -]+",
        "_",
        path.stem,
    ).strip(" ._-")

    if not sanitized_stem:
        sanitized_stem = "document"

    return f"{sanitized_stem[:100]}.pdf"


def validate_pdf_content(content: bytes) -> None:
    if not content:
        raise ValueError("Uploaded PDF cannot be empty")

    if len(content) > MAX_PDF_SIZE_BYTES:
        raise ValueError(
            "Uploaded PDF exceeds the 20 MB size limit"
        )

    if b"%PDF-" not in content[:1024]:
        raise ValueError(
            "Uploaded content does not appear to be a PDF"
        )


def store_uploaded_pdf(
    original_filename: str,
    content: bytes,
    config: AppConfig,
) -> StoredDocument:
    safe_filename = sanitize_pdf_filename(
        original_filename
    )
    validate_pdf_content(content)
    config.create_directories()

    document_id = hashlib.sha256(content).hexdigest()
    stored_filename = (
        f"{document_id[:16]}-{safe_filename}"
    )
    destination = (
        config.data_directory / stored_filename
    )

    already_exists = destination.exists()

    if not already_exists:
        temporary_path = destination.with_suffix(
            ".pdf.tmp"
        )
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)

    return StoredDocument(
        document_id=document_id,
        original_name=safe_filename,
        stored_path=destination,
        already_exists=already_exists,
    )