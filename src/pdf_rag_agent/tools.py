from langchain.tools import tool

from pdf_rag_agent.arithmetic import evaluate_arithmetic
from pdf_rag_agent.ingest import get_vector_store


def create_search_documents_tool(
    document_id: str | None = None,
    maximum_distance: float = 1.20,
):
    if maximum_distance <= 0:
        raise ValueError(
            "Maximum distance must be greater than zero"
        )

    normalized_document_id: str | None = None

    if document_id is not None:
        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise ValueError(
                "Document ID cannot be empty"
            )

    @tool("search_documents")
    def search_documents(query: str) -> str:
        """Search indexed PDFs for sufficiently relevant information."""

        normalized_query = query.strip()

        if not normalized_query:
            return "Search query cannot be empty."

        vector_store = get_vector_store()

        search_arguments: dict[str, object] = {
            "query": normalized_query,
            "k": 5,
        }

        if normalized_document_id is not None:
            search_arguments["filter"] = {
                "document_id": normalized_document_id
            }

        scored_documents = (
            vector_store.similarity_search_with_score(
                **search_arguments
            )
        )

        relevant_documents = [
            (document, distance)
            for document, distance in scored_documents
            if distance <= maximum_distance
        ]

        if not relevant_documents:
            return (
                "No sufficiently relevant document content "
                "was found for this question."
            )

        results: list[str] = []

        for index, (document, distance) in enumerate(
            relevant_documents,
            start=1,
        ):
            document_name = document.metadata.get(
                "document_name",
                "Unknown document",
            )

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
Distance: {distance:.2f}
Content:
{document.page_content}
""".strip()
            )

        return "\n\n".join(results)

    return search_documents


search_documents = create_search_documents_tool()


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe basic arithmetic expression."""

    try:
        return str(
            evaluate_arithmetic(expression)
        )
    except (ValueError, ZeroDivisionError) as error:
        return f"Calculation failed: {error}"