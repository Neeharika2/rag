from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from injestion.chunk import chunk_file
from injestion.embedding import embed_chunks
from injestion.parser import parse_pdf


class IngestionState(TypedDict):
    pdf_path: str
    parsed_file: str
    chunks_file: str
    embedded_count: int


def parse_node(state: IngestionState) -> IngestionState:
    parsed_file = parse_pdf(state["pdf_path"])
    return {**state, "parsed_file": parsed_file}


def chunk_node(state: IngestionState) -> IngestionState:
    chunks_file = chunk_file(state["parsed_file"])
    return {**state, "chunks_file": chunks_file}


def embed_node(state: IngestionState) -> IngestionState:
    embedded_count = embed_chunks(state["chunks_file"])
    return {**state, "embedded_count": embedded_count}


def build_ingestion_graph():
    graph = StateGraph(IngestionState)

    graph.add_node("parse", parse_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", END)

    return graph.compile()


def run_ingestion_graph(pdf_path: str) -> IngestionState:
    graph = build_ingestion_graph()
    initial_state: IngestionState = {
        "pdf_path": pdf_path,
        "parsed_file": "",
        "chunks_file": "",
        "embedded_count": 0,
    }
    return graph.invoke(initial_state)
