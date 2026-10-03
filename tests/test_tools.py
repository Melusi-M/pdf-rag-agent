import pytest
from langchain_core.documents import Document

from pdf_rag_agent import tools


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_arguments = {}

    def similarity_search_with_score(
        self,
        **kwargs,
    ):
        self.received_arguments = kwargs

        return [
            (
                Document(
                    page_content=(
                        "Relevant document content"
                    ),
                    metadata={
                        "document_name": "example.pdf",
                        "page": 2,
                        "document_id": "document-123",
                    },
                ),
                0.75,
            )
        ]


def test_search_tool_filters_by_document_id(
    monkeypatch,
) -> None:
    fake_vector_store = FakeVectorStore()

    monkeypatch.setattr(
        tools,
        "get_vector_store",
        lambda: fake_vector_store,
    )

    search_tool = (
        tools.create_search_documents_tool(
            document_id="document-123",
        )
    )

    result = search_tool.invoke(
        {"query": "What is the termination date?"}
    )

    assert fake_vector_store.received_arguments == {
        "query": "What is the termination date?",
        "k": 5,
        "filter": {
            "document_id": "document-123",
        },
    }

    assert "example.pdf" in result
    assert "Page: 3" in result
    assert "Distance: 0.75" in result
    assert "Relevant document content" in result


def test_search_tool_can_search_all_documents(
    monkeypatch,
) -> None:
    fake_vector_store = FakeVectorStore()

    monkeypatch.setattr(
        tools,
        "get_vector_store",
        lambda: fake_vector_store,
    )

    search_tool = (
        tools.create_search_documents_tool()
    )

    search_tool.invoke(
        {"query": "Find relevant information"}
    )

    assert fake_vector_store.received_arguments == {
        "query": "Find relevant information",
        "k": 5,
    }


def test_search_tool_rejects_distant_results(
    monkeypatch,
) -> None:
    class DistantVectorStore:
        def similarity_search_with_score(
            self,
            **kwargs,
        ):
            return [
                (
                    Document(
                        page_content="Unrelated content",
                        metadata={
                            "document_name": "example.pdf",
                            "page": 0,
                        },
                    ),
                    1.50,
                )
            ]

    monkeypatch.setattr(
        tools,
        "get_vector_store",
        lambda: DistantVectorStore(),
    )

    search_tool = (
        tools.create_search_documents_tool(
            maximum_distance=1.20,
        )
    )

    result = search_tool.invoke(
        {
            "query": (
                "A completely unrelated question"
            )
        }
    )

    assert result == (
        "No sufficiently relevant document content "
        "was found for this question."
    )


def test_search_tool_rejects_invalid_distance() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        tools.create_search_documents_tool(
            maximum_distance=0,
        )