"""
ingest.py - AeroVantage X9 Drone Manual Ingestion Pipeline.

Extracts text from aerovantage_x9_drone_manual_50p.pdf, performs text cleaning,
detects boilerplate template sections, chunks text with recursive character splitting,
generates dense vector embeddings using local sentence-transformers (all-MiniLM-L6-v2),
and persists the vector store in ChromaDB (./chroma_db).
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pypdf
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "drone_manual")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PDF_PATHS = [
    os.getenv("PDF_PATH", ""),
    "./data/aerovantage_x9_drone_manual_50p.pdf",
    "./aerovantage_x9_drone_manual_50p.pdf"
]

_EMBEDDING_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(f"Loading local embedding model '{EMBEDDING_MODEL_NAME}'...")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def resolve_pdf_path() -> Path:
    for candidate in DEFAULT_PDF_PATHS:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("Could not find 'aerovantage_x9_drone_manual_50p.pdf' in ./data or project root.")


def compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_text(raw_text: str) -> str:
    """Normalize quotes, dashes, and whitespace while preserving numbers, units, and codes."""
    text = raw_text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove running footer lines
    text = re.sub(r"Page\s+\d+\s+of\s+\d+\s*\|\s*Confidential[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Document\s+Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # Strip excess horizontal whitespace on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_chapter_title(text: str) -> str:
    match = re.search(r"(CHAPTER\s+\d+\s*\|\s*MODULE\s+[\d\.]+:?[^\n=]+)", text)
    if match:
        return match.group(1).strip()
    return "Official Technical Manual"


def is_template_boilerplate(text: str) -> bool:
    """
    Detect repetitive template paragraphs present in the 50-page corpus.
    Pages containing substantive technical sections (SECTION x.y) or concrete specifications
    are non-boilerplate.
    """
    has_section = bool(re.search(r"SECTION\s+\d+\.\d+:", text))
    if has_section:
        return False

    has_bp1 = "establishes the baseline operational procedures" in text
    has_bp2 = "Standard Procedural Guidelines" in text
    has_bp3 = "Archival & Verification Notes" in text
    return (has_bp1 or has_bp2 or has_bp3)


def recursive_character_split(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:
    """
    Splits text recursively using delimiters: double newlines, single newlines, sentences, spaces.
    Ensures chunks conform to chunk_size with chunk_overlap.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(txt: str, seps: List[str]) -> List[str]:
        if len(txt) <= chunk_size:
            return [txt] if txt.strip() else []
        sep = seps[0] if seps else ""
        if not sep:
            # Hard character cut if no separator matches
            step = max(1, chunk_size - chunk_overlap)
            return [txt[i:i + chunk_size] for i in range(0, len(txt), step)]

        parts = txt.split(sep)
        chunks = []
        cur = ""
        for p in parts:
            candidate = cur + (sep if cur else "") + p
            if len(candidate) <= chunk_size:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur)
                    tail = cur[-chunk_overlap:] if len(cur) > chunk_overlap else cur
                    cur = tail + sep + p if len(tail) + len(sep) + len(p) <= chunk_size else p
                else:
                    sub = _split(p, seps[1:])
                    chunks.extend(sub)
        if cur and (not chunks or cur != chunks[-1]):
            chunks.append(cur)
        return chunks

    return _split(text, separators)


def build_chunks_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = pypdf.PdfReader(str(pdf_path))
    chunks: List[Dict[str, Any]] = []

    logger.info(f"Processing {len(reader.pages)} pages from {pdf_path.name}...")

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-based page index
        raw_text = page.extract_text() or ""
        cleaned = normalize_text(raw_text)
        chapter_title = extract_chapter_title(cleaned)
        boilerplate = is_template_boilerplate(cleaned)

        if boilerplate:
            # Strip repetitive boilerplate paragraphs, retaining heading and module reference
            concise_text = f"{chapter_title}\nModule Page {page_num} Standard Operational Procedures."
            chunks.append({
                "chunk_id": f"p{page_num}_c0",
                "page": page_num,
                "chapter_title": chapter_title,
                "is_boilerplate": True,
                "text": concise_text,
                "source_file": pdf_path.name
            })
        else:
            # Preserve full substantive technical content
            page_chunks = recursive_character_split(cleaned, chunk_size=800, chunk_overlap=150)
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "chunk_id": f"p{page_num}_c{chunk_idx}",
                    "page": page_num,
                    "chapter_title": chapter_title,
                    "is_boilerplate": False,
                    "text": chunk_text.strip(),
                    "source_file": pdf_path.name
                })

    return chunks


def ingest_pdf(
    pdf_path: Path = None,
    force: bool = False
) -> Tuple[int, bool]:
    """
    Ingests the AeroVantage manual into ChromaDB.
    Returns: (total_chunks_count, was_reembedded_bool)
    """
    if pdf_path is None:
        pdf_path = resolve_pdf_path()
    else:
        pdf_path = Path(pdf_path).resolve()

    pdf_hash = compute_file_hash(pdf_path)
    chroma_dir = Path(CHROMA_DB_DIR).resolve()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    meta_path = chroma_dir / "ingest_metadata.json"
    chunks_cache_path = chroma_dir / "chunks_metadata.json"

    # Check persistence: do we already have this exact collection persisted?
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_collections = [c.name for c in client.list_collections()]

    if not force and COLLECTION_NAME in existing_collections and meta_path.is_file() and chunks_cache_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            if saved_meta.get("pdf_hash") == pdf_hash:
                col = client.get_collection(COLLECTION_NAME)
                count = col.count()
                if count > 0:
                    logger.info(
                        f"ChromaDB collection '{COLLECTION_NAME}' already populated with {count} chunks "
                        f"(PDF hash {pdf_hash[:8]} matches). Skipping re-embedding."
                    )
                    return count, False
        except Exception as e:
            logger.warning(f"Failed verifying existing ChromaDB persistence: {e}. Re-ingesting.")

    logger.info("Starting ingestion and embedding pipeline...")
    chunks = build_chunks_from_pdf(pdf_path)
    total_chunks = len(chunks)
    logger.info(f"Generated {total_chunks} chunks across {pdf_path.name}.")

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    logger.info(f"Computing dense vector embeddings for {total_chunks} chunks with '{EMBEDDING_MODEL_NAME}'...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True).tolist()

    # Reset collection in ChromaDB
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "chapter_title": c["chapter_title"],
            "is_boilerplate": c["is_boilerplate"],
            "source_file": c["source_file"]
        }
        for c in chunks
    ]

    # Ingest in batches
    batch_size = 50
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )

    # Save metadata files for quick reload & persistence verification
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "pdf_hash": pdf_hash,
            "pdf_path": str(pdf_path),
            "num_chunks": total_chunks,
            "embedding_model": EMBEDDING_MODEL_NAME
        }, f, indent=2)

    with open(chunks_cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(f"Successfully ingested and persisted {total_chunks} chunks into ChromaDB at '{chroma_dir}'.")
    return total_chunks, True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest AeroVantage X9 Drone Manual into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-embedding even if cached")
    args = parser.parse_args()
    count, reembedded = ingest_pdf(force=args.force)
    print(f"Ingestion complete: {count} chunks (Re-embedded: {reembedded})")
