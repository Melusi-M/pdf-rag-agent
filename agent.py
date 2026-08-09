import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from tools import calculator, search_documents


load_dotenv()


@lru_cache(maxsize=1)
def build_agent():
    model_name = os.getenv("CLAUDE_MODEL")

    if not model_name:
        raise RuntimeError(
            "CLAUDE_MODEL is missing from the .env file."
        )

    model = ChatAnthropic(
        model=model_name,
        temperature=0,
        max_tokens=2000,
    )

    system_prompt = """
You are a careful PDF research assistant.

Rules:
1. You MUST call search_documents before answering any question that could
   depend on the uploaded PDFs. Never ask the user which document they mean,
   whether a document is uploaded, or for more details before searching --
   search first, with your best-guess query, and retry with different
   phrasings if the first search is not conclusive.
2. Base document answers only on retrieved evidence.
3. Cite the document name and page number for factual claims.
4. Do not invent information that is absent from the retrieved text.
5. Only say the documents lack the information after several searches with
   varied queries have returned nothing relevant.
6. Use the calculator tool for arithmetic.
7. Keep answers clear and structured.
"""

    return create_agent(
        model=model,
        tools=[
            search_documents,
            calculator,
        ],
        system_prompt=system_prompt,
    )


def ask_agent(question: str) -> str:
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    final_message = result["messages"][-1]
    content = final_message.content

    # Anthropic responses may arrive as a list of content blocks.
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return content


if __name__ == "__main__":
    user_question = input("Question: ")
    print(ask_agent(user_question))