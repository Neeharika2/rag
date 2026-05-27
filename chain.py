import json
import uuid
from typing import List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import DEFAULT_ALPHA, DEFAULT_TOP_K, GEMINI_MODEL
from retrieval.retriever import build_retriever

_chat_histories: dict[str, InMemoryChatMessageHistory] = {}

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about government budgets, "
    "policy documents, and public finance. Use the following retrieved context to "
    "answer the question. If the context does not contain enough information, say so. "
    "Always cite which source document your answer comes from when possible.\n\n"
    "Context:\n{context}"
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about government budgets, "
    "policy documents, and public finance. Use the following retrieved context to "
    "answer the question. If the context does not contain enough information, say so. "
    "Always cite which source document your answer comes from when possible. "
    "Maintain a conversational tone and refer to previous messages when relevant.\n\n"
    "Context:\n{context}"
)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryChatMessageHistory()
    return _chat_histories[session_id]


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def docs_to_sources(docs: List[Document]) -> List[dict]:
    return [
        {
            "id": doc.metadata.get("chunk_index", ""),
            "source": doc.metadata.get("source", "unknown"),
            "score": doc.metadata.get("score", 0),
        }
        for doc in docs
    ]


def build_qa_chain(top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA):
    from retrieval.retriever import _get_llm

    llm = _get_llm()
    retriever = build_retriever(top_k=top_k, alpha=alpha)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        RunnableParallel({
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_chat_chain(top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA):
    from retrieval.retriever import _get_llm

    llm = _get_llm()
    retriever = build_retriever(top_k=top_k, alpha=alpha)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnableParallel({
            "context": lambda x: (
                format_docs(retriever.invoke(x["question"]))
                if isinstance(x["question"], str)
                else format_docs(retriever.invoke(x["question"]))
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history


def query_rag(question: str, top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA) -> dict:
    retriever = build_retriever(top_k=top_k, alpha=alpha)
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "No relevant documents found.", "sources": []}

    sources = docs_to_sources(docs)
    chain = build_qa_chain(top_k=top_k, alpha=alpha)
    answer = chain.invoke({"question": question})

    return {"answer": answer, "sources": sources}


def query_chat(question: str, session_id: str, top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA) -> dict:
    if not session_id:
        session_id = str(uuid.uuid4())

    retriever = build_retriever(top_k=top_k, alpha=alpha)
    docs = retriever.invoke(question)
    sources = docs_to_sources(docs)

    chain = build_chat_chain(top_k=top_k, alpha=alpha)
    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    return {"answer": answer, "sources": sources, "session_id": session_id}


def stream_rag(question: str, top_k: int = DEFAULT_TOP_K, alpha: float = DEFAULT_ALPHA):
    retriever = build_retriever(top_k=top_k, alpha=alpha)
    docs = retriever.invoke(question)

    if not docs:
        yield json.dumps({"type": "sources", "data": []}) + "\n"
        yield json.dumps({"type": "token", "text": "No relevant documents found."}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    sources = docs_to_sources(docs)
    yield json.dumps({"type": "sources", "data": sources}) + "\n"

    context = format_docs(docs)
    chain = build_qa_chain(top_k=top_k, alpha=alpha)

    for chunk in chain.stream({"question": question}):
        yield json.dumps({"type": "token", "text": chunk}) + "\n"

    yield json.dumps({"type": "done"}) + "\n"