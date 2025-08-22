import os, shutil


from pathlib import Path

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
#from langchain_chroma import Chroma

from langchain_community.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma


from .settings import settings

def _embeddings():
    try:
        return OllamaEmbeddings(model=settings.embed_model)
    except Exception:
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def _vs():
    # Always use a stable collection name
    return Chroma(
        collection_name="docs",
        persist_directory=settings.vector_db_path,
        embedding_function=_embeddings(),
    )

def ingest():
    docs = load_docs(settings.docs_path)
    if not docs:
        return {"ok": False, "message": f"No docs found in {settings.docs_path}. Add PDFs/TXT/MD first."}

    chunks = chunk_docs(docs)

    # 🔧 reset the vector store cleanly (avoid stale handles)
    if os.path.exists(settings.vector_db_path):
        shutil.rmtree(settings.vector_db_path, ignore_errors=True)

    vs = _vs()  # creates collection if missing
    vs.add_documents(chunks)
    vs.persist()

    return {"ok": True, "message": f"Ingested {len(chunks)} chunks from {len(docs)} docs."}

def load_docs(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif path.suffix.lower() in [".txt", ".md"]:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs

def chunk_docs(docs: List):
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    return splitter.split_documents(docs)

# def ingest() -> Dict[str, Any]:
#     docs = load_docs(settings.docs_path)
#     if not docs:
#         return {"ok": False, "message": "No docs found"}
#     chunks = chunk_docs(docs)
#     vs = _vs()
#     vs.delete_collection()
#     vs.add_documents(chunks)
#     vs.persist()
#     return {"ok": True, "message": f"Ingested {len(chunks)} chunks from {len(docs)} docs."}

def retriever():
    return _vs().as_retriever(search_kwargs={"k": settings.top_k})

def _format_docs(docs: List) -> str:
    return "\n\n".join([f"[{i+1}] {d.metadata.get('source')} \n{d.page_content}" for i, d in enumerate(docs)])

def rag_answer(question: str) -> Dict[str, Any]:
    r = retriever()
    template = ChatPromptTemplate.from_messages([
        ("system", "Answer using context. Cite sources as [1], [2]..."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    llm = ChatOllama(model=settings.chat_model, temperature=0.2)
    chain = ({"question": RunnablePassthrough(), "context": r | _format_docs} 
             | template | llm | StrOutputParser())
    answer = chain.invoke(question)
    docs = r.get_relevant_documents(question)
    sources = [{"tag": f"[{i+1}]", "source": d.metadata.get("source")} for i,d in enumerate(docs)]
    return {"answer": answer, "sources": sources}
