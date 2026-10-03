from pdf_rag_agent.ingest import get_vector_store


def search(
    query: str,
    number_of_results: int = 4,
):
    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query=query,
        k=number_of_results,
    )


if __name__ == "__main__":
    question = input("Ask a question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty")

    results = search(question)

    for document, score in results:
        page = document.metadata.get("page")
        displayed_page = (
            page + 1
            if isinstance(page, int)
            else "Unknown"
        )

        document_name = document.metadata.get(
            "document_name",
            "Unknown",
        )

        print("\n" + "=" * 70)
        print(f"Document: {document_name}")
        print(f"Page: {displayed_page}")
        print(f"Distance score: {score:.4f}")
        print(document.page_content[:700])