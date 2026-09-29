import os
from pathlib import Path

# Prioritize local cached Hugging Face models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
from docx import Document
from pptx import Presentation


# Default knowledge base folder
DEFAULT_DOCS_DIR = Path("files")


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return the SentenceTransformer embedding model.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """
    Split text into overlapping chunks based on words.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def extract_text(path: Path) -> str:
    """
    Extract text from TXT, PDF, DOCX and PPTX files.
    """

    extension = path.suffix.lower()

    # -------------------------
    # TXT
    # -------------------------
    if extension == ".txt":
        return path.read_text(encoding="utf-8")

    # -------------------------
    # PDF
    # -------------------------
    elif extension == ".pdf":
        text_parts = []

        reader = PdfReader(str(path))

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        return "\n".join(text_parts)

    # -------------------------
    # DOCX
    # -------------------------
    elif extension == ".docx":
        document = Document(str(path))

        paragraphs = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        return "\n".join(paragraphs)

    # -------------------------
    # PPTX
    # -------------------------
    elif extension == ".pptx":
        presentation = Presentation(str(path))

        slides_text = []

        for slide in presentation.slides:

            slide_parts = []

            for shape in slide.shapes:

                if hasattr(shape, "text"):
                    text = shape.text.strip()

                    if text:
                        slide_parts.append(text)

            if slide_parts:
                slides_text.append("\n".join(slide_parts))

        return "\n".join(slides_text)

    # -------------------------
    # Unsupported
    # -------------------------
    else:
        return ""


def load_documents(docs_dir: Path = DEFAULT_DOCS_DIR) -> tuple:
    """
    Dynamically scan the knowledge base folder and load:

    .txt
    .pdf
    .docx
    .pptx

    Documents are converted to text and then chunked.
    """

    docs_dir = Path(docs_dir)

    if not docs_dir.exists():
        return [], {
            "doc_count": 0,
            "chunk_count": 0,
            "files": []
        }

    chunks = []
    files_processed = []

    supported_extensions = {
        ".txt",
        ".pdf",
        ".docx",
        ".pptx"
    }

    for path in sorted(docs_dir.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_extensions:
            continue

        try:
            text = extract_text(path)

        except Exception:
            continue

        trimmed = text.strip()

        if not trimmed:
            continue

        files_processed.append(path.name)

        text_chunks = chunk_text(trimmed)

        for i, chunk in enumerate(text_chunks):

            chunks.append({
                "source": path.name,
                "chunk_id": i + 1,
                "text": chunk
            })

    stats = {
        "doc_count": len(files_processed),
        "chunk_count": len(chunks),
        "files": files_processed
    }

    return chunks, stats


def build_vector_index(
    chunks: list,
    model: SentenceTransformer
) -> tuple:
    """
    Generate normalized embeddings and store them in FAISS.
    """

    if not chunks:
        return None, 0

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    return index, dim


def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list,
    model: SentenceTransformer,
    top_k: int = 5
) -> list:
    """
    Perform vector similarity search.
    """

    if not query or index is None or not chunks:
        return []

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, ids = index.search(
        query_vector,
        k
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        item = chunks[idx]

        results.append({
            "score": round(float(score), 4),
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"]
        })

    return results