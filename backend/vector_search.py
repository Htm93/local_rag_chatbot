"""
Vector store layer.

- Seeds the Chroma collection from data.csv on first run (same behaviour as
  the original script).
- Exposes `retriever` for the LangGraph agent.
- Exposes `ingest_file()` so the FastAPI backend can add newly uploaded
  documents (PDF / TXT / CSV / DOCX) to the same collection at runtime.

Everything here is 100% local: HuggingFace sentence-transformers for
embeddings, Chroma for storage. No network calls, no cloud APIs.
"""

import os
import uuid
from typing import List

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_LOCATION = "chroma_db"
COLLECTION_NAME = "data"
SEED_CSV = "data.csv"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def _seed_documents_from_csv() -> List[Document]:
    """Recreates the original CSV -> Document behaviour, if data.csv exists."""
    if not os.path.exists(SEED_CSV):
        return []

    df = pd.read_csv(SEED_CSV)
    documents = []
    for i, row in df.iterrows():
        documents.append(
            Document(
                page_content=f'{row.get("Title", "")} {row.get("data", "")}',
                metadata={"date": str(row.get("Date", "")), "source": "data.csv"},
                id=str(i),
            )
        )
    return documents


_needs_seed = not os.path.exists(DB_LOCATION)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

if _needs_seed:
    seed_docs = _seed_documents_from_csv()
    if seed_docs:
        vector_store.add_documents(documents=seed_docs, ids=[d.id for d in seed_docs])

# k: number of chunks pulled per query
retriever = vector_store.as_retriever(search_kwargs={"k": 8})


def ingest_file(filepath: str) -> int:
    """
    Load a file from disk, split it into chunks, embed each chunk, and add
    it to the running Chroma collection. Returns the number of chunks added.

    Supported extensions: .pdf, .txt, .csv, .docx
    """
    ext = os.path.splitext(filepath)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = loader_cls(filepath)
    raw_docs = loader.load()
    chunks = splitter.split_documents(raw_docs)

    if not chunks:
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(filepath)

    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)