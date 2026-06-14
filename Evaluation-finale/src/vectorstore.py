from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_DIR,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore() -> Chroma:
    """Charge les PDF de data/pdfs/, les decoupe en chunks et construit
    le vectorstore Chroma persiste dans data/chroma_db/."""
    documents = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    if not documents:
        raise FileNotFoundError(
            f"Aucun PDF trouve dans {PDF_DIR}. Lancez d'abord download_pdfs.py."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def get_vectorstore() -> Chroma:
    """Charge le vectorstore Chroma deja persiste (construit par ingest.py)."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            f"Vectorstore introuvable dans {CHROMA_DIR}. Lancez d'abord : uv run python ingest.py"
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
