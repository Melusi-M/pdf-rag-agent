from functools import lru_cache

from langchain.tools import tool
from langchain_chroma import Chroma

from ingest import (
    COLLECTION_NAME,
    DATABASE_PATH,
    get_embedding_model,
)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=DATABASE_PATH,
    )


@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded PDF documents for information relevant to the query.

    Use this tool when the user asks about facts, policies, figures,
    requirements, dates, clauses or other information contained in the PDFs.
    """

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query=query,
        k=5,
    )

    if not documents:
        return "No relevant document content was found."

    results: list[str] = []

    for index, document in enumerate(documents, start=1):
        document_name = document.metadata.get(
            "document_name",
            "Unknown document",
        )

        # PyPDFLoader normally stores pages using zero-based indexing.
        stored_page = document.metadata.get("page")
        displayed_page = (
            stored_page + 1
            if isinstance(stored_page, int)
            else "Unknown"
        )

        results.append(
            f"""
SOURCE {index}
Document: {document_name}
Page: {displayed_page}
Content:
{document.page_content}
""".strip()
        )

    return "\n\n".join(results)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression.

    Use this for calculations based on figures found in documents.
    Only arithmetic characters are accepted.
    """

    allowed_characters = set("0123456789+-*/().% ")

    if not expression or not set(expression).issubset(allowed_characters):
        return "Invalid arithmetic expression."

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {},
        )
        return str(result)
    except Exception as error:
        return f"Calculation failed: {error}"