"""
RAG Uploader – unified entry point for both text and image-based PDF catalogues.

Detects whether the uploaded PDF is text-based or image-based and routes to
the correct ingestion pipeline:

  • Image-based (scanned) PDF  → catalogue_ingestor.py
       Uses GPT-4o Vision page-by-page, uploads page PNGs to Azure Blob Storage,
       stores rich product chunks with blob_url metadata in ChromaDB "starlight_vision".

  • Text-based PDF             → legacy text extraction path
       Uses PyMuPDF text + GPT-4o for extraction, stores in ChromaDB "starlight_vision"
       (same collection, so all catalogues are queried together).

Usage (Streamlit):
    from rag_uploader import process_pdf_to_chroma
    ok, msg = process_pdf_to_chroma(uploaded_file, progress_callback)
"""
import os
import io
import json
import tempfile
import logging

import fitz  # PyMuPDF
import chromadb

from azure_client import azure_manager

log = logging.getLogger("rag_uploader")

COLLECTION_NAME = "starlight_vision"   # shared with catalogue_ingestor
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def _is_image_only(pdf_path: str, sample_pages: int = 5) -> bool:
    """
    Return True if the PDF contains no extractable text (i.e. is scanned).
    Samples up to `sample_pages` pages.
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    text_found = 0
    for i in range(pages_to_check):
        text = doc[i].get_text("text").strip()
        if len(text) > 40:
            text_found += 1
    doc.close()
    return text_found == 0


# ---------------------------------------------------------------------------
# Text-based fallback pipeline (for PDFs that do contain text)
# ---------------------------------------------------------------------------

_TEXT_SYSTEM = """\
You are an expert extraction engine for LED lighting product catalogues.
Extract EVERY distinct product, series, or accessory from the supplied catalogue text.

For each item output a rich, self-contained description covering:
1. Product Name / Series
2. Application (residential, commercial, hospitality, retail, outdoor, etc.)
3. Key Features
4. All Technical Specifications: wattage (W), voltage (V), CCT (K), CRI, IP rating,
   beam angle (°), dimensions (mm), lumen output (lm), LED type, driver, mounting
5. Available variants (sizes, finishes, CCT options)

Output ONLY a valid JSON array of plain-text strings.
Each string = one product, fully self-contained. No markdown fences.
"""

_TEXT_USER_TMPL = "Extract all products from this catalogue section:\n\n{text}"

MAX_BATCH_CHARS = 4000


def _batch_pages(pages: list[str], max_chars: int = MAX_BATCH_CHARS) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for page in pages:
        if current and current_len + len(page) > max_chars:
            batches.append("\n\n".join(current))
            current, current_len = [page], len(page)
        else:
            current.append(page)
            current_len += len(page)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _extract_text_chunks(batch: str) -> list[str]:
    raw = azure_manager.chat_completion(
        [
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user", "content": _TEXT_USER_TMPL.format(text=batch)},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        chunks = json.loads(cleaned)
        if isinstance(chunks, list):
            return [str(c).strip() for c in chunks if str(c).strip()]
    except json.JSONDecodeError:
        return [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 80]
    return []


def _ingest_text_pdf(
    pdf_path: str,
    source_name: str,
    progress_callback=None,
) -> tuple[bool, str]:
    """Fallback: text-extraction pipeline for readable PDFs."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) > 40:
            pages.append(f"[Page {i}]\n{text}")
    doc.close()

    if not pages:
        return False, "No readable text found in this PDF."

    batches = _batch_pages(pages)
    all_chunks: list[str] = []

    for i, batch in enumerate(batches):
        if progress_callback:
            pct = 0.20 + 0.50 * (i / max(len(batches), 1))
            progress_callback(pct, f"Extracting products from batch {i+1}/{len(batches)}…")
        all_chunks.extend(_extract_text_chunks(batch))

    # Deduplicate
    seen: set[str] = set()
    unique = [c for c in all_chunks if not (c in seen or seen.add(c))]  # type: ignore

    if not unique:
        return False, "No product information could be extracted."

    if progress_callback:
        progress_callback(0.72, f"Embedding {len(unique)} chunks…")

    vectors = azure_manager.embed_documents(unique)
    ids = [f"{source_name}::text::chunk::{i}" for i in range(len(unique))]
    metas = [
        {
            "source":         source_name,
            "catalogue_name": source_name,
            "catalogue_type": "text_pdf",
            "page_number":    0,
            "blob_url":       "",
            "product_name":   "",
            "category":       "",
            "specs_preview":  "",
            "char_count":     len(c),
        }
        for i, c in enumerate(unique)
    ]

    if progress_callback:
        progress_callback(0.90, "Writing to vector database…")

    chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    # Replace stale entries
    try:
        ex = collection.get(where={"source": source_name})
        if ex.get("ids"):
            collection.delete(ids=ex["ids"])
    except Exception:
        pass

    collection.add(ids=ids, embeddings=vectors, documents=unique, metadatas=metas)

    if progress_callback:
        progress_callback(1.0, f"Done! {len(unique)} chunks indexed.")

    return True, f"Success: {len(unique)} product chunks from '{source_name}' added."


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def process_pdf_to_chroma(
    pdf_file,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Ingest a catalogue PDF into the shared ChromaDB 'starlight_vision' collection.

    Automatically detects whether the PDF is image-only (uses GPT-4o Vision per
    page) or text-based (uses text extraction + GPT-4o), then ingests accordingly.

    Args:
        pdf_file:          File-like object with a ``.name`` attribute
                           (Streamlit UploadedFile or open() handle).
        progress_callback: Optional ``callable(fraction: float, message: str)``.

    Returns:
        ``(True, success_message)`` or ``(False, error_message)``.
    """
    temp_path = ""
    source_name: str = getattr(pdf_file, "name", "unknown_catalogue.pdf")

    try:
        if progress_callback:
            progress_callback(0.03, "Saving catalogue file…")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            temp_path = tmp.name

        # ── Detect PDF type ─────────────────────────────────────────────
        if progress_callback:
            progress_callback(0.06, "Detecting catalogue format (text vs image)…")

        is_image = _is_image_only(temp_path)
        log.info("'%s' is_image_only=%s", source_name, is_image)

        if is_image:
            # ── Vision pipeline ─────────────────────────────────────────
            if progress_callback:
                progress_callback(0.08, "Image-based PDF detected → GPT-4o Vision pipeline…")

            from catalogue_ingestor import ingest_catalogue

            # Wrap progress: ingestor uses 0–1 scale, we offset slightly
            def _wrapped_cb(pct, msg):
                if progress_callback:
                    # Map ingestor 0–1 into our 0.08–1.0 range
                    progress_callback(0.08 + 0.92 * pct, msg)

            # Write the temp file path and pass it to the ingestor
            # (ingestor expects a file path, not a file object)
            import shutil
            named_temp = temp_path + "_" + source_name
            shutil.copy(temp_path, named_temp)

            summary = ingest_catalogue(
                pdf_path=named_temp,
                progress_callback=_wrapped_cb,
                force_reingest=True,
            )

            os.remove(named_temp)

            if summary["total_chunks"] == 0:
                return False, (
                    "No products were found in this catalogue. "
                    "Pages may be purely decorative or the quality may be too low."
                )

            return True, (
                f"Success: {summary['total_chunks']} product chunks indexed from "
                f"{summary['pages_processed']} pages of '{summary['catalogue_name']}'."
            )

        else:
            # ── Text pipeline ────────────────────────────────────────────
            if progress_callback:
                progress_callback(0.08, "Text-based PDF detected → extraction pipeline…")

            return _ingest_text_pdf(temp_path, source_name, progress_callback)

    except Exception as exc:
        log.error("Ingestion failed for '%s': %s", source_name, exc, exc_info=True)
        return False, f"Error processing catalogue: {exc}"

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
