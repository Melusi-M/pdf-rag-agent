from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATABASE_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_and_split_pdf(pdf_path: str):
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    # Add useful metadata that can later appear in citations.
    for page in pages:
        page.metadata["document_name"] = path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def ingest_pdf(pdf_path: str) -> int:
    chunks = load_and_split_pdf(pdf_path)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=DATABASE_PATH,
    )

    ids = [
        f"{chunk.metadata['document_name']}-{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)


if __name__ == "__main__":
    number_of_chunks = ingest_pdf("./data/Lease_Agreement_signed.pdf")
    print(f"Stored {number_of_chunks} chunks.")