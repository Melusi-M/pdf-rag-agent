import streamlit as st

from pdf_rag_agent.agent import ask_agent
from pdf_rag_agent.config import get_default_config
from pdf_rag_agent.ingest import (
    count_indexed_chunks,
    ingest_pdf,
    is_document_indexed,
)
from pdf_rag_agent.uploads import store_uploaded_pdf


st.set_page_config(
    page_title="PDF RAG Agent",
    page_icon="📄",
)

config = get_default_config()

st.title("PDF RAG Agent")
st.caption(
    "Ask questions about your uploaded PDF documents."
)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if "last_processed" in st.session_state:
    st.success(
        st.session_state.pop("last_processed")
    )

try:
    indexed_count = count_indexed_chunks(config)
except Exception as error:
    indexed_count = 0
    st.sidebar.error(
        f"Could not inspect the document index: {error}"
    )

if indexed_count:
    st.sidebar.success(
        f"{indexed_count} chunks indexed and searchable."
    )
else:
    st.sidebar.warning(
        "No documents indexed yet. "
        "Upload a PDF and click Process PDF."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    st.write(f"Selected: `{uploaded_file.name}`")

    if st.button(
        "Process PDF",
        type="primary",
    ):
        try:
            with st.spinner("Processing document..."):
                stored_document = store_uploaded_pdf(
                    original_filename=uploaded_file.name,
                    content=uploaded_file.getvalue(),
                    config=config,
                )

                if is_document_indexed(
                    stored_document.document_id,
                    config=config,
                ):
                    message = (
                        "This PDF has already been indexed."
                    )
                else:
                    chunk_count = ingest_pdf(
                        stored_document.stored_path,
                        config=config,
                        document_name=(
                            stored_document.original_name
                        ),
                        document_id=(
                            stored_document.document_id
                        ),
                    )

                    message = (
                        "PDF processed successfully: "
                        f"{chunk_count} chunks."
                    )

                previous_document_id = (
                    st.session_state.get(
                        "active_document_id"
                    )
                )

                if (
                    previous_document_id is not None
                    and previous_document_id
                    != stored_document.document_id
                ):
                    st.session_state[
                        "chat_messages"
                    ] = []

                st.session_state[
                    "active_document_id"
                ] = stored_document.document_id

                st.session_state[
                    "active_document_name"
                ] = stored_document.original_name

            st.session_state["last_processed"] = message
            st.rerun()

        except Exception as error:
            st.error(f"Processing failed: {error}")

active_document_id = st.session_state.get(
    "active_document_id"
)
active_document_name = st.session_state.get(
    "active_document_name"
)

search_all_documents = st.sidebar.checkbox(
    "Search all indexed documents",
    value=False,
)

if search_all_documents:
    st.sidebar.info(
        "Questions will search the complete document index."
    )
elif active_document_name:
    st.sidebar.info(
        f"Active document: {active_document_name}"
    )
else:
    st.sidebar.warning(
        "Process a PDF before asking document questions."
    )

if st.sidebar.button("Clear chat"):
    st.session_state["chat_messages"] = []
    st.rerun()

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input(
    "Ask something about your documents",
    disabled=(
        not search_all_documents
        and active_document_id is None
    ),
)

if question:
    existing_history = list(
        st.session_state["chat_messages"]
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            search_document_id = (
                None
                if search_all_documents
                else active_document_id
            )

            with st.spinner("Searching documents..."):
                answer = ask_agent(
                    question,
                    document_id=search_document_id,
                    history=existing_history,
                )

            st.write(answer)

            st.session_state[
                "chat_messages"
            ].extend(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            )

        except Exception as error:
            st.error(f"Agent failed: {error}")