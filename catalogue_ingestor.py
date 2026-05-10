"""
Catalogue Ingestor – Vision-based RAG pipeline for image-only PDF catalogues.

These Starlight catalogues are 100% scanned/rasterised — zero extractable text.
Every page is a single high-resolution raster image embedded in the PDF.

Pipeline (per page):
  1. Render the PDF page to a PNG pixmap with PyMuPDF
  2. Resize to two resolutions:
       • API image  (~900px wide)  – sent to GPT-4o Vision for extraction
       • Blob image (~1200px wide) – uploaded to Azure Blob Storage for emails
  3. Upload the blob image → get a permanent HTTP URL (blob_url)
  4. Send the API image to GPT-4o Vision with a structured extraction prompt
  5. Parse the returned JSON array of product objects
  6. For each product, build:
       • A rich embedding document  (text string that goes into ChromaDB)
       • A metadata dict           (blob_url, page_number, product_name, …)
  7. Embed documents with Azure text-embedding-3-large
  8. Upsert into ChromaDB collection "starlight_vision"

The collection is keyed on {source}::page::{n}::product::{i} so re-ingesting
the same PDF replaces stale entries cleanly.

Usage:
    from catalogue_ingestor import ingest_catalogue
    results = ingest_catalogue(
        pdf_path="Starlight_Linear_Catalogue.PDF",
        progress_callback=lambda pct, msg: print(f"{pct:.0%} {msg}")
    )
"""
import io
import os
import re
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import fitz          # PyMuPDF
from PIL import Image

import chromadb

from azure_client import azure_manager
from azure_blob import blob_manager

log = logging.getLogger("catalogue_ingestor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "starlight_vision"
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Image widths (pixels) — balance quality vs. payload size
API_IMAGE_MAX_W = 900    # sent to GPT-4o Vision
BLOB_IMAGE_MAX_W = 1200  # stored in Azure Blob for emails

# Rendering DPI multiplier for the low-res linear catalogue (612×792 pt pages)
LOW_RES_ZOOM = 2.5  # 72 DPI × 2.5 = 180 DPI effective render

# ---------------------------------------------------------------------------
# Vision extraction prompt
# ---------------------------------------------------------------------------
_VISION_SYSTEM = """\
You are a precision product-data extractor for Starlight LED lighting catalogues.

Analyse this catalogue page image and extract EVERY distinct product, model, \
or accessory shown.

For each product return a JSON object with these keys:
  "product_name"   – exact name / model number as printed on the page
  "category"       – one of: linear | downlight | spotlight | panel | strip |
                     surface_mount | pendant | outdoor | track | furniture |
                     kitchen | accessory | other
  "description"    – 1–2 sentence description of the product and its application
  "features"       – array of strings (bullet-point highlights)
  "specs"          – object with any of these fields found on the page:
                       wattage, voltage, cct, cri, ip_rating, beam_angle,
                       dimensions, lumen, led_type, driver, mounting,
                       material, finish_options, colour_temp_range
  "variants"       – array of available sizes / finishes / wattage options
  "page_context"   – one sentence describing the overall theme of this page

CRITICAL RULES:
- Only extract values that are VISIBLY PRINTED on this page.
- Never invent, extrapolate, or assume any specification.
- If a spec field is absent from the page, omit that key entirely.
- For the "specs" object use the exact units shown (e.g. "12W", "220-240V AC").

Return a JSON array (even for a single product). \
If this is a cover, contents, or purely decorative page with no products, return [].
Output ONLY the JSON array — no markdown fences, no preamble.
"""

_VISION_USER_TMPL = "Extract all products from page {page_label} of the Starlight catalogue."


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _render_page(page: fitz.Page) -> bytes:
    """
    Render a PDF page to PNG bytes.

    Low-resolution pages (letter / 612×792 pts) are upscaled for better
    vision model accuracy.  High-resolution A4 scans are rendered 1:1.
    """
    zoom = LOW_RES_ZOOM if page.rect.width <= 650 else 1.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _resize_png(image_bytes: bytes, max_width: int) -> bytes:
    """Resize a PNG so its width ≤ max_width, preserving aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.width <= max_width:
        return image_bytes
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Catalogue slug helper
# ---------------------------------------------------------------------------

def _catalogue_slug(filename: str) -> str:
    """
    Convert a raw filename into a clean, lowercase URL-safe slug.

    'STARLIGHT KITCHEN & FURNITURE 2024-2025 (3).pdf'
    → 'starlight_kitchen_furniture_2024_2025'
    """
    stem = Path(filename).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug[:60]  # cap length for safe blob names


def _catalogue_display_name(filename: str) -> str:
    """Human-readable name for use in metadata and email templates."""
    stem = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", stem).strip()
    # Title-case, drop trailing numbers / brackets
    name = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
    return name.title()


# ---------------------------------------------------------------------------
# GPT-4o Vision extraction
# ---------------------------------------------------------------------------

def _extract_products_from_page(
    api_image_bytes: bytes,
    page_label: str,
) -> list[dict]:
    """
    Call GPT-4o Vision and parse the returned JSON array of product objects.
    Returns an empty list if the page is decorative / no products found.
    """
    raw = azure_manager.vision_completion(
        image_bytes=api_image_bytes,
        text_prompt=_VISION_USER_TMPL.format(page_label=page_label),
        system_prompt=_VISION_SYSTEM,
        temperature=0.0,
        max_tokens=3000,
        json_mode=False,   # we request an array, not an object
    )

    if not raw:
        return []

    cleaned = raw.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        products = json.loads(cleaned)
        if isinstance(products, list):
            return products
        if isinstance(products, dict):
            # model occasionally returns {"products": [...]}
            for key in ("products", "items", "data"):
                if key in products and isinstance(products[key], list):
                    return products[key]
    except json.JSONDecodeError:
        log.warning("JSON parse failed for %s — skipping page.", page_label)

    return []


# ---------------------------------------------------------------------------
# Chunk & metadata builder
# ---------------------------------------------------------------------------

def _build_chunk_and_metadata(
    product: dict,
    catalogue_name: str,
    catalogue_slug: str,
    catalogue_type: str,
    page_number: int,
    blob_url: str,
    source_filename: str,
) -> tuple[str, dict]:
    """
    Convert a raw product dict into:
      • document string   – rich text ready for embedding
      • metadata dict     – stored alongside in ChromaDB
    """
    name = product.get("product_name", "Unknown Product")
    category = product.get("category", "other")
    description = product.get("description", "")
    features = product.get("features", [])
    specs = product.get("specs", {})
    variants = product.get("variants", [])

    # ── Build embedding document ──────────────────────────────────────────
    lines = [
        f"Product: {name}",
        f"Catalogue: {catalogue_name}  |  Page: {page_number}",
        f"Category: {category}",
    ]
    if description:
        lines.append(f"Description: {description}")
    if features:
        lines.append("Features: " + "; ".join(str(f) for f in features))
    if specs:
        spec_parts = []
        for k, v in specs.items():
            if v:
                spec_parts.append(f"{k.replace('_', ' ').title()}: {v}")
        if spec_parts:
            lines.append("Specifications: " + " | ".join(spec_parts))
    if variants:
        lines.append("Variants: " + ", ".join(str(v) for v in variants))

    document = "\n".join(lines)

    # ── Build specs preview (for email template) ─────────────────────────
    spec_preview_parts = []
    for key in ("wattage", "cct", "cri", "ip_rating", "beam_angle"):
        val = specs.get(key)
        if val:
            spec_preview_parts.append(str(val))
    specs_preview = " · ".join(spec_preview_parts[:4])

    # ── Metadata (ChromaDB values must be str / int / float / bool) ───────
    metadata: dict = {
        "source":           source_filename,
        "catalogue_name":   catalogue_name,
        "catalogue_slug":   catalogue_slug,
        "catalogue_type":   catalogue_type,
        "page_number":      page_number,
        "blob_url":         blob_url,
        "product_name":     name,
        "category":         category,
        "specs_preview":    specs_preview,
        "char_count":       len(document),
    }

    return document, metadata


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def ingest_catalogue(
    pdf_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    force_reingest: bool = False,
) -> dict:
    """
    Full ingestion pipeline for a single image-based PDF catalogue.

    Args:
        pdf_path:          Absolute or relative path to the PDF file.
        progress_callback: Optional ``callable(fraction, message)`` for UI.
        force_reingest:    If True, delete existing entries for this source
                           before ingesting (useful for re-uploading a revised PDF).

    Returns:
        A summary dict:
        {
          "pages_processed": int,
          "pages_skipped":   int,   # decorative / no products
          "total_chunks":    int,
          "blob_urls":       {page_number: blob_url, ...},
          "catalogue_name":  str,
          "catalogue_type":  str,
        }
    """
    pdf_path = str(pdf_path)
    filename = os.path.basename(pdf_path)
    slug = _catalogue_slug(filename)
    display_name = _catalogue_display_name(filename)

    # Determine catalogue type from filename
    lower = filename.lower()
    if any(k in lower for k in ("kitchen", "furniture")):
        catalogue_type = "kitchen_furniture"
    else:
        catalogue_type = "linear_led"

    log.info("Ingesting '%s' → slug=%s  type=%s", filename, slug, catalogue_type)

    if progress_callback:
        progress_callback(0.02, f"Opening '{display_name}'…")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # ── Prepare ChromaDB collection ─────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # If collection exists with wrong embedding dim, recreate it gracefully
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        log.warning("Collection issue (%s). Recreating…", exc)
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # Remove existing entries for this source if requested
    if force_reingest:
        try:
            existing = collection.get(where={"source": filename})
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
                log.info("Deleted %d stale entries for %s", len(existing["ids"]), filename)
        except Exception:
            pass

    # ── Per-page loop ────────────────────────────────────────────────────
    all_docs:  list[str]  = []
    all_metas: list[dict] = []
    all_ids:   list[str]  = []
    blob_url_map: dict[int, str] = {}

    pages_skipped = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_label = f"{page_num}/{total_pages}"

        if progress_callback:
            pct = 0.05 + 0.85 * (page_idx / total_pages)
            progress_callback(
                pct,
                f"Page {page_label} — rendering & extracting…",
            )

        page = doc[page_idx]

        # 1. Render page
        native_png = _render_page(page)

        # 2. Resize for blob (display quality)
        blob_png = _resize_png(native_png, BLOB_IMAGE_MAX_W)

        # 3. Upload blob → get URL
        blob_url = blob_manager.upload_page_image(blob_png, slug, page_num)
        blob_url_map[page_num] = blob_url

        # 4. Resize for API (smaller payload)
        api_png = _resize_png(native_png, API_IMAGE_MAX_W)

        # 5. Extract products with GPT-4o Vision
        products = _extract_products_from_page(api_png, page_label)

        if not products:
            log.debug("Page %s: no products extracted (decorative/cover).", page_label)
            pages_skipped += 1
            continue

        log.info("Page %s: %d product(s) extracted.", page_label, len(products))

        # 6. Build chunks + metadata
        for prod_idx, product in enumerate(products):
            document, metadata = _build_chunk_and_metadata(
                product=product,
                catalogue_name=display_name,
                catalogue_slug=slug,
                catalogue_type=catalogue_type,
                page_number=page_num,
                blob_url=blob_url,
                source_filename=filename,
            )
            chunk_id = f"{filename}::page::{page_num:03d}::product::{prod_idx:02d}"
            all_docs.append(document)
            all_metas.append(metadata)
            all_ids.append(chunk_id)

    doc.close()

    if not all_docs:
        if progress_callback:
            progress_callback(1.0, "No products found. Check that the PDF contains product pages.")
        return {
            "pages_processed": 0,
            "pages_skipped": pages_skipped,
            "total_chunks": 0,
            "blob_urls": blob_url_map,
            "catalogue_name": display_name,
            "catalogue_type": catalogue_type,
        }

    # 7. Embed all documents
    if progress_callback:
        progress_callback(0.91, f"Embedding {len(all_docs)} product chunks…")

    vectors = azure_manager.embed_documents(all_docs)

    # 8. Remove existing entries for this source (upsert = delete + add)
    try:
        existing = collection.get(where={"source": filename})
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # 9. Insert into ChromaDB
    if progress_callback:
        progress_callback(0.97, "Writing to vector database…")

    collection.add(
        ids=all_ids,
        embeddings=vectors,
        documents=all_docs,
        metadatas=all_metas,
    )

    pages_processed = total_pages - pages_skipped
    if progress_callback:
        progress_callback(
            1.0,
            f"Done! {len(all_docs)} products indexed from {pages_processed} pages.",
        )

    log.info(
        "Ingestion complete: %s | pages=%d | chunks=%d",
        filename, pages_processed, len(all_docs),
    )

    return {
        "pages_processed": pages_processed,
        "pages_skipped":   pages_skipped,
        "total_chunks":    len(all_docs),
        "blob_urls":       blob_url_map,
        "catalogue_name":  display_name,
        "catalogue_type":  catalogue_type,
    }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    paths = sys.argv[1:] or [
        "STARLIGHT KITCHEN & FURNITURE 2024-2025 (3).pdf",
        "Starlight_Linear_Catalogue.PDF",
    ]

    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        def _cb(pct, msg):
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"\r[{bar}] {pct:4.0%}  {msg:<60}", end="", flush=True)

        print(f"\nIngesting: {path}")
        summary = ingest_catalogue(path, progress_callback=_cb)
        print(f"\n\nResult: {summary}")
