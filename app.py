import tempfile
from pathlib import Path

import streamlit as st

from agent import ask_agent
from ingest import ingest_pdf
from tools import get_vector_store


st.set_page_config(
    page_title="PDF RAG Agent",
    page_icon="📄",
)

st.title("PDF RAG Agent")
st.caption("Ask questions about your uploaded PDF documents.")

indexed_count = get_vector_store()._collection.count()

if "last_processed" in st.session_state:
    st.success(st.session_state.pop("last_processed"))

if indexed_count:
    st.sidebar.success(f"{indexed_count} chunks indexed and searchable.")
else:
    st.sidebar.warning(
        "No documents indexed yet. Upload a PDF and click Process PDF."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
    ) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        temporary_path = temporary_file.name

    # Preserve the original filename for better source labels.
    destination = Path("./data") / uploaded_file.name
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(Path(temporary_path).read_bytes())

    if st.button("Process PDF"):
        try:
            with st.spinner("Processing document..."):
                chunk_count = ingest_pdf(str(destination))

            st.session_state["last_processed"] = (
                f"PDF processed successfully: {chunk_count} chunks."
            )
            st.rerun()
        except Exception as error:
            st.error(f"Processing failed: {error}")

question = st.chat_input("Ask something about your documents")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                answer = ask_agent(question)

            st.write(answer)
        except Exception as error:
            st.error(f"Agent failed: {error}")