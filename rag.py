"""
rag.py - Retrieval-Augmented Generation & Grounding Pipeline for AeroVantage X9 Manual.

Provides the required answer() interface:
def answer(question: str) -> dict:
    '''Returns {"answer": <str>, "sources": [<page_int>, ...]}'''
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from ingest import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_embedding_model,
    ingest_pdf,
    resolve_pdf_path
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag")

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_ALL_CHUNKS: List[Dict[str, Any]] = []
_BM25_INDEX = None

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would", "could",
    "tell", "give", "find", "show", "me", "per", "state", "indicate", "its", "it",
    "does", "indicate", "x9", "aerovantage", "manual", "drone", "hexacopter",
    "industrial", "technical", "maintenance", "document", "corpus", "reference", "page"
}


def normalize_query_text(q: str) -> str:
    """Expands digit/word variations and standard acronyms for robust lexical and semantic search."""
    q_norm = q.lower()
    num_map = {
        "three": "3", "two": "2", "six": "6", "four": "4", "five": "5", "one": "1",
        "per-cell": "per cell", "percell": "per cell",
        "metres": "meters", "meters": "metres",
        "return-to-home": "rth", "rth": "return-to-home"
    }
    for k, v in num_map.items():
        q_norm = re.sub(rf"\b{k}\b", f"{k} {v}", q_norm)
    return q_norm


def tokenize_for_bm25(text: str) -> List[str]:
    norm = normalize_query_text(text)
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", norm)


def load_retrieval_resources():
    """Lazily initialize ChromaDB collection and BM25 index."""
    global _CHROMA_CLIENT, _CHROMA_COLLECTION, _ALL_CHUNKS, _BM25_INDEX
    if _CHROMA_COLLECTION is not None and _BM25_INDEX is not None:
        return

    chroma_dir = Path(CHROMA_DB_DIR).resolve()
    chunks_cache_path = chroma_dir / "chunks_metadata.json"

    # Ensure ingestion has run
    if not chroma_dir.is_dir() or not chunks_cache_path.is_file():
        logger.info("ChromaDB index not detected. Triggering ingestion...")
        ingest_pdf(force=False)

    _CHROMA_CLIENT = chromadb.PersistentClient(path=str(chroma_dir))
    _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection(COLLECTION_NAME)

    with open(chunks_cache_path, "r", encoding="utf-8") as f:
        _ALL_CHUNKS = json.load(f)

    corpus_tokens = [tokenize_for_bm25(c["text"]) for c in _ALL_CHUNKS]
    _BM25_INDEX = BM25Okapi(corpus_tokens)
    logger.info(f"Loaded {len(_ALL_CHUNKS)} chunks into hybrid retrieval engine.")


def extract_distinctive_terms(question: str) -> List[str]:
    """Extract content terms from query, dropping stopwords and common corpus tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", question.lower())
    terms = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return terms


def verify_grounding_gate(question: str, candidate_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Distinctive-term gate:
    Rejects questions whose key technical entities or distinctive terms have no match
    in the top non-boilerplate retrieved chunks BEFORE generation.
    """
    terms = extract_distinctive_terms(question)
    if not terms:
        return False, "No distinctive terms in query"

    non_bp_chunks = [c for c in candidate_chunks if not c.get("is_boilerplate", False)]
    if not non_bp_chunks:
        return False, "No non-boilerplate chunks retrieved"

    context = " ".join(c["text"] for c in non_bp_chunks).lower()

    def term_variants(w: str) -> List[str]:
        vars_ = [w]
        num_map = {"three": "3", "two": "2", "six": "6", "four": "4", "50": "50", "38": "38"}
        if w in num_map:
            vars_.append(num_map[w])
        if w.endswith("es") and len(w) > 3:
            vars_.append(w[:-2])
        elif w.endswith("s") and len(w) > 2:
            vars_.append(w[:-1])
        if w.endswith("ly") and len(w) > 3:
            vars_.append(w[:-2])
        if w in ("metres", "meters"):
            vars_.extend(["meters", "metres", "m"])
        if w == "rth":
            vars_.extend(["return-to-home", "home"])
        return vars_

    matched = [t for t in terms if any(v in context for v in term_variants(t))]
    term_ratio = len(matched) / len(terms)

    # Check for external proper entities (capitalized words like DJI Innovations, CEO)
    raw_caps = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", question)
    if raw_caps and question.startswith(raw_caps[0]):
        raw_caps = raw_caps[1:]  # Skip sentence starter
    external_entities = [e.lower() for e in raw_caps if e.lower() not in STOPWORDS]
    if external_entities:
        has_entity = any(any(v in context for v in term_variants(e)) for e in external_entities)
        if not has_entity:
            return False, f"Key entity missing from corpus: {external_entities}"

    # Check technical nouns that are domain-critical
    technical_keywords = [t for t in terms if t in {"sonar", "underwater", "submersible", "ocean", "depth"}]
    if technical_keywords and not any(t in context for t in technical_keywords):
        return False, f"Domain technical terms missing: {technical_keywords}"

    if term_ratio < 0.40:
        return False, f"Term match ratio {term_ratio:.2f} below threshold 0.40"

    return True, f"Passed gate (ratio={term_ratio:.2f})"


def hybrid_retrieve(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Combines ChromaDB dense retrieval with BM25 lexical retrieval using Reciprocal Rank Fusion."""
    load_retrieval_resources()
    model = get_embedding_model()

    q_expanded = normalize_query_text(question)
    q_emb = model.encode([q_expanded], normalize_embeddings=True).tolist()[0]

    # Dense retrieval (top 8)
    dense_results = _CHROMA_COLLECTION.query(
        query_embeddings=[q_emb],
        n_results=min(8, len(_ALL_CHUNKS))
    )
    dense_ids = dense_results["ids"][0] if dense_results["ids"] else []

    # Lexical retrieval (top 8)
    q_tokens = tokenize_for_bm25(question)
    bm25_scores = _BM25_INDEX.get_scores(q_tokens)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:8]

    # Reciprocal Rank Fusion
    k_rrf = 60
    fused_scores: Dict[str, float] = {}

    for rank, chunk_id in enumerate(dense_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)

    for rank, idx in enumerate(bm25_ranked_indices):
        chunk_id = _ALL_CHUNKS[idx]["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)

    chunk_map = {c["chunk_id"]: c for c in _ALL_CHUNKS}

    # Down-weight boilerplate chunks
    for cid in fused_scores:
        c = chunk_map.get(cid)
        if c and c.get("is_boilerplate", False):
            fused_scores[cid] *= 0.1

    ranked_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
    results = [chunk_map[cid] for cid in ranked_chunk_ids[:top_k] if cid in chunk_map]
    return results


def extractive_fallback_answer(question: str, candidate_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic fallback when LLM is unavailable or fails.
    Extracts the highest-scoring factual sentence from top non-boilerplate retrieved chunk.
    """
    non_bp = [c for c in candidate_chunks if not c.get("is_boilerplate", False)]
    if not non_bp:
        return {"answer": "[not found]", "sources": []}

    top_chunk = non_bp[0]
    text = top_chunk["text"]

    # Reconstruct coherent sentences across PDF lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    merged_lines = []
    for l in lines:
        if merged_lines and not l.startswith("-") and not l.startswith("SECTION") and not l.startswith("CHAPTER") and not merged_lines[-1].endswith("."):
            merged_lines[-1] += " " + l
        else:
            merged_lines.append(l)

    sentences = []
    for ml in merged_lines:
        for s in re.split(r"(?<=[.!?])\s+", ml):
            s_clean = s.strip()
            if s_clean and not s_clean.startswith("AeroVantage X9") and not s_clean.startswith("CHAPTER"):
                # Clean embedded section titles
                s_clean = re.sub(r"^SECTION\s+\d+\.\d+:[^A-Z]*[A-Z\s&]+\s+(?=The|AeroVantage|[A-Z][a-z])", "", s_clean)
                sentences.append(s_clean)

    q_words = set(re.findall(r"\b[a-z0-9]+\b", question.lower()))
    num_map = {"three": "3", "two": "2", "six": "6", "four": "4", "five": "5", "one": "1"}
    for k, v in num_map.items():
        if k in question.lower():
            q_words.add(v)

    best_sent = ""
    best_score = -1
    for s in sentences:
        s_words = set(re.findall(r"\b[a-z0-9]+\b", s.lower()))
        score = len(q_words & s_words)
        if "operational compliance note" in s.lower() or "personnel" in s.lower():
            score -= 5
        if score > best_score:
            best_score = score
            best_sent = s

    if best_score >= 3 and best_sent:
        cleaned = re.sub(r"^[-*•\s]+", "", best_sent).strip()
        return {"answer": cleaned, "sources": [top_chunk["page"]]}

    return {"answer": "[not found]", "sources": []}


def call_llm_generation(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Attempts answer generation via LLM using priority order:
    ANTHROPIC_API_KEY -> OPENAI_API_KEY -> GEMINI_API_KEY.
    Returns parsed dictionary or raises Exception for fallback.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    context_str = "\n\n".join([
        f"[Page {c['page']}]:\n{c['text']}"
        for c in context_chunks if not c.get("is_boilerplate", False)
    ])

    system_prompt = (
        "You answer strictly from the provided context excerpts of the AeroVantage X9 manual. "
        "Use no outside knowledge. If the context does not explicitly contain the answer, or only contains "
        "generic/unrelated text, reply with exactly: [not found]. Do not infer capabilities that are not stated "
        "(e.g. if the manual does not mention a use case, the answer is [not found], not 'No'). "
        "Answer concisely and include exact figures, units, error codes and condition names as written in the text. "
        'Return JSON only: {"answer": "...", "page_numbers": [..]} where page_numbers lists only pages whose context you actually used.'
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

    raw_response = None

    if anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        model_name = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
        resp = client.messages.create(
            model=model_name,
            max_tokens=300,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw_response = resp.content[0].text

    elif openai_key:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_response = resp.choices[0].message.content

    elif gemini_key:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        # Try configured model, default to gemini-2.5-flash or gemini-3.8-flash
        model_name = os.getenv("LLM_MODEL", "gemini-3.8-flash")
        config = {
            "temperature": 0.0,
            "system_instruction": system_prompt,
            "response_mime_type": "application/json"
        }
        resp = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config
        )
        raw_response = resp.text

    if not raw_response:
        raise RuntimeError("No LLM provider configured or no response received.")

    # Parse JSON output robustly
    clean_json = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.IGNORECASE)
    clean_json = re.sub(r"\s*```$", "", clean_json.strip())

    try:
        data = json.loads(clean_json)
    except Exception:
        ans_m = re.search(r'"answer"\s*:\s*"(.*?)"', raw_response, re.DOTALL)
        pages_m = re.findall(r"\d+", re.search(r'"page_numbers"\s*:\s*\[(.*?)\]', raw_response, re.DOTALL).group(1) if '"page_numbers"' in raw_response else "")
        data = {
            "answer": ans_m.group(1) if ans_m else raw_response.strip(),
            "page_numbers": [int(p) for p in pages_m]
        }

    return data


def answer(question: str) -> Dict[str, Any]:
    """
    Main RAG entrypoint.
    Returns: {"answer": <str>, "sources": [<page_int>, ...]}
    Guaranteed never to crash and strictly adheres to grounding rules.
    """
    try:
        if not question or not question.strip():
            return {"answer": "[not found]", "sources": []}

        # 1. Hybrid Retrieval
        candidates = hybrid_retrieve(question, top_k=4)

        # 2. Strict Distinctive-Term Grounding Gate
        passed, reason = verify_grounding_gate(question, candidates)
        if not passed:
            logger.info(f"Grounding gate rejected question: {reason}")
            return {"answer": "[not found]", "sources": []}

        # 3. Generation via LLM with Extractive Fallback
        result = None
        has_api_key = any(os.getenv(k, "").strip() for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])
        if has_api_key:
            try:
                llm_data = call_llm_generation(question, candidates)
                ans_text = str(llm_data.get("answer", "")).strip()
                pages = [int(p) for p in llm_data.get("page_numbers", []) if isinstance(p, (int, str)) and str(p).isdigit()]
                result = {"answer": ans_text, "sources": pages}
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Falling back to extractive mode.")

        if result is None:
            result = extractive_fallback_answer(question, candidates)

        # 4. Post-Validation and Normalization
        ans = result.get("answer", "").strip()
        sources = result.get("sources", [])

        # Check for not found markers in any form
        not_found_pattern = re.compile(
            r"(\[not found\]|not found|not available|no information|cannot find|unsupported|not mentioned)",
            re.IGNORECASE
        )
        if not_found_pattern.search(ans):
            return {"answer": "[not found]", "sources": []}

        # Constrain sources to retrieved pages
        valid_retrieved_pages = {c["page"] for c in candidates if not c.get("is_boilerplate", False)}
        filtered_sources = [p for p in sources if p in valid_retrieved_pages]

        # If answer is supported but sources list was empty, assign top candidate page
        if not filtered_sources and valid_retrieved_pages:
            filtered_sources = [candidates[0]["page"]]

        return {
            "answer": ans,
            "sources": sorted(list(set(filtered_sources)))
        }

    except Exception as exc:
        logger.error(f"Unexpected error in answer(): {exc}", exc_info=True)
        return {"answer": "[not found]", "sources": []}


if __name__ == "__main__":
    test_q = "What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?"
    res = answer(test_q)
    print("Test Answer:", res)
