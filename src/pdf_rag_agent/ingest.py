from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_rag_agent.config import (
    AppConfig,
    get_default_config,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=8)
def _get_cached_vector_store(
    chroma_directory: str,
    collection_name: str,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=chroma_directory,
    )


def get_vector_store(
    config: AppConfig | None = None,
) -> Chroma:
    selected_config = config or get_default_config()
    selected_config.create_directories()

    return _get_cached_vector_store(
        chroma_directory=str(
            selected_config.chroma_directory
        ),
        collection_name=selected_config.collection_name,
    )


def load_pdf_pages(
    path: Path,
    document_name: str | None = None,
    document_id: str | None = None,
) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []
    displayed_name = document_name or path.name

    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text(
           extraction_mode="layout",
         layout_mode_space_vertically=False,
) or ""

        if not page_text.strip():
            continue

        metadata = {
            "source": str(path),
            "document_name": displayed_name,
            "page": page_number,
        }

        if document_id is not None:
            metadata["document_id"] = document_id

        documents.append(
            Document(
                page_content=page_text,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            f"PDF contains no extractable text: {path}"
        )

    return documents

def load_and_split_pdf(
    pdf_path: str | Path,
    document_name: str | None = None,
    document_id: str | None = None,
) -> list[Document]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"PDF path is not a file: {pdf_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"File is not a PDF: {pdf_path}"
        )

    pages = load_pdf_pages(path,document_name=document_name,
document_id=document_id,)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks

def ingest_pdf(
    pdf_path: str | Path,
    config: AppConfig | None = None,
    document_name: str | None = None,
    document_id: str | None = None,
) -> int:
    chunks = load_and_split_pdf(
        pdf_path,
        document_name=document_name,
        document_id=document_id,
    )
    vector_store = get_vector_store(config)

    ids = [
        (
            f"{document_id or chunk.metadata['document_name']}-"
            f"{chunk.metadata['chunk_id']}"
        )
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)

def count_indexed_chunks(
    config: AppConfig | None = None,
) -> int:
    vector_store = get_vector_store(config)
    result = vector_store.get(include=[])

    return len(result.get("ids", []))


def is_document_indexed(
    document_id: str,
    config: AppConfig | None = None,
) -> bool:
    normalized_id = document_id.strip()

    if not normalized_id:
        raise ValueError("Document ID cannot be empty")

    vector_store = get_vector_store(config)
    result = vector_store.get(
        where={"document_id": normalized_id},
        include=[],
    )

    return bool(result.get("ids"))