from langchain_chroma import Chroma

from ingest import (
    COLLECTION_NAME,
    DATABASE_PATH,
    get_embedding_model,
)


def search(query: str, number_of_results: int = 4):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=DATABASE_PATH,
    )

    return vector_store.similarity_search_with_score(
        query=query,
        k=number_of_results,
    )


if __name__ == "__main__":
    question = input("Ask a question: ")

    results = search(question)

    for document, score in results:
        page = document.metadata.get("page", "Unknown")
        document_name = document.metadata.get("document_name", "Unknown")

        print("\n" + "=" * 70)
        print(f"Document: {document_name}")
        print(f"Page: {page}")
        print(f"Distance score: {score}")
        print(document.page_content[:700])