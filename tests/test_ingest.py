from pathlib import Path

import pytest
from langchain_core.documents import Document

from pdf_rag_agent import ingest
from pdf_rag_agent.config import AppConfig


def test_load_and_split_pdf_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(
        FileNotFoundError,
        match="PDF not found",
    ):
        ingest.load_and_split_pdf(missing_pdf)


def test_load_and_split_pdf_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="PDF path is not a file",
    ):
        ingest.load_and_split_pdf(tmp_path)


def test_load_and_split_pdf_rejects_non_pdf_file(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text(
        "Not a PDF",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="File is not a PDF",
    ):
        ingest.load_and_split_pdf(text_file)


def test_ingest_pdf_uses_supplied_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig.from_project_root(tmp_path)

    chunks = [
        Document(
            page_content="Example lease content",
            metadata={
                "document_name": "Original Lease.pdf",
                "document_id": "document-hash",
                "chunk_id": 0,
            },
        )
    ]

    class FakeVectorStore:
        def __init__(self) -> None:
            self.documents = []
            self.ids = []

        def add_documents(
            self,
            documents,
            ids,
        ) -> None:
            self.documents = documents
            self.ids = ids

    fake_vector_store = FakeVectorStore()
    received_configs: list[AppConfig] = []

    monkeypatch.setattr(
        ingest,
        "load_and_split_pdf",
        lambda *_args, **_kwargs: chunks,
    )

    def fake_get_vector_store(
        received_config: AppConfig,
    ) -> FakeVectorStore:
        received_configs.append(received_config)
        return fake_vector_store

    monkeypatch.setattr(
        ingest,
        "get_vector_store",
        fake_get_vector_store,
    )

    result = ingest.ingest_pdf(
        tmp_path / "lease.pdf",
        config=config,
        document_name="Original Lease.pdf",
        document_id="document-hash",
    )

    assert result == 1
    assert received_configs == [config]
    assert fake_vector_store.documents == chunks
    assert fake_vector_store.ids == ["document-hash-0"]

def test_load_pdf_pages_extracts_text_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "example.pdf"

    class FakePage:
        def __init__(
            self,
            text: str | None,
        ) -> None:
            self.text = text
            self.received_options = {}

        def extract_text(
            self,
            **kwargs,
        ) -> str | None:
            self.received_options = kwargs
            return self.text

    fake_pages = [
        FakePage("First page content"),
        FakePage("   "),
        FakePage(None),
    ]

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = fake_pages

    monkeypatch.setattr(
        ingest,
        "PdfReader",
        FakeReader,
    )

    documents = ingest.load_pdf_pages(pdf_path)

    assert len(documents) == 1
    assert documents[0].page_content == (
        "First page content"
    )
    assert documents[0].metadata == {
        "source": str(pdf_path),
        "document_name": "example.pdf",
        "page": 0,
    }

    assert fake_pages[0].received_options == {
        "extraction_mode": "layout",
        "layout_mode_space_vertically": False,
    }