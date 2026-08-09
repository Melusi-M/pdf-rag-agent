# PDF RAG Agent

An agentic RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and ask questions about them in natural language. Built as part of my ongoing exploration of AI agent development — moving beyond simple prompt-in/answer-out pipelines toward agents that reason, use tools, and decide for themselves how to find an answer.

## What makes it agentic?

Instead of a fixed retrieve-then-answer pipeline, the LLM is given **tools** and decides how to use them:

- **`search_documents`** — semantic search over the ingested PDF chunks. The agent will reformulate its query and search multiple times if the first results are inconclusive (e.g. a lease that says "terminates on the date in item 1.24" sends the agent hunting for the schedule that defines item 1.24).
- **`calculator`** — safe arithmetic for calculations based on figures found in documents.

The agent is instructed to always ground its answers in retrieved evidence, cite document names and page numbers, and admit when the documents genuinely don't contain the answer.

## Architecture

```
┌────────────┐    ┌─────────────────────┐    ┌──────────────┐
│ Streamlit  │───▶│ Ingestion            │───▶│ ChromaDB     │
│ (app.py)   │    │ PyPDF → split →      │    │ (persistent  │
│            │    │ embed (MiniLM-L6-v2) │    │ vector store)│
└─────┬──────┘    └─────────────────────┘    └──────▲───────┘
      │                                             │
      │           ┌─────────────────────┐    ┌──────┴───────┐
      └──────────▶│ LangChain agent      │───▶│ Tools:       │
                  │ (Claude)             │    │ search_docs, │
                  │ reason → act → loop  │    │ calculator   │
                  └─────────────────────┘    └──────────────┘
```

- **Ingestion** (`ingest.py`) — loads PDFs with PyPDF, splits into overlapping chunks with `RecursiveCharacterTextSplitter` (1500 chars, 250 overlap), embeds with `all-MiniLM-L6-v2`, and stores in a persistent ChromaDB collection. Deterministic chunk IDs make re-ingestion idempotent.
- **Tools** (`tools.py`) — the document search and calculator tools exposed to the agent. Embedding model and vector store are cached so they load once per process, not once per query.
- **Agent** (`agent.py`) — a LangChain agent powered by Claude, with a system prompt tuned to search before answering and to cite sources.
- **UI** (`app.py`) — Streamlit interface with upload, ingestion status, and chat. A sidebar indicator shows how many chunks are indexed so you always know whether the store is ready.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

## Run

```bash
streamlit run app.py
```

1. Upload a PDF and click **Process PDF**.
2. Ask questions in the chat — e.g. *"When does the lease expire?"* or *"What is the monthly rent, and what does it total over the initial period?"*

You can also test retrieval directly from the command line:

```bash
python test_search.py
```

## Lessons learned along the way

- **Prompt the agent to search first.** An agent with a search tool will still sometimes ask the user clarifying questions instead of using the tool. Making the system prompt explicit — *search first, retry with varied queries, only then say you don't know* — fixed this.
- **Cache your embedding model.** Recreating the embedding model per tool call meant reloading model weights on every search. Caching it cut multi-search questions from painfully slow to instant.
- **Chunk overlap matters for legal documents.** Cross-references ("the date set out in item 1.24") span sections, so generous overlap plus an agent willing to re-search beats trying to get the perfect chunk size.

## Roadmap

- [ ] Multi-document management (list, delete, re-index)
- [ ] Conversation memory across chat turns
- [ ] Streaming responses in the UI
- [ ] Evaluation harness for retrieval quality
