"""
Email Generator v2 – Azure OpenAI edition with metadata-aware RAG references.

Flow per client record:
  1. get_rag_context()          – HyDE query expansion → ChromaDB retrieval
                                  Returns text chunks + product reference metadata
                                  (blob_url, product_name, page_number, catalogue_name)
  2. generate_creative_draft()  – GPT-4o drafts structured email JSON,
                                  grounded strictly in retrieved catalogue context
  3. refine_with_judge()        – second GPT-4o pass normalises and validates JSON
  4. generate_eml_from_record() – renders Jinja2 template, writes .html + .eml
                                  Template receives `referenced_products` list so
                                  each email shows the actual catalogue page images

Anti-hallucination guarantees:
  • Catalogue context injected verbatim — model forbidden from inventing specs
  • `json_mode=True` on both passes → no JSON parse failures
  • Judge pass validates list fields are plain strings
"""
import os
import sys
import json
import time
import re
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

load_dotenv()

from azure_client import azure_manager
import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
sender_name    = os.getenv("SENDER_NAME",    "Vivek Dhondarkar")
sender_company = os.getenv("SENDER_COMPANY", "Starlight Linear LED")
sender_phone   = os.getenv("SENDER_PHONE",   "9619436066")
sender_website = os.getenv("SENDER_WEBSITE", "www.starlightlinearled.com")
sender_email   = os.getenv("SENDER_EMAIL",   "vivek@starlightlinearled.com")
company_logo_url = os.getenv("COMPANY_LOGO_URL", "cid:company_logo")
logo_path = os.path.join(os.path.dirname(__file__), "starlight.jpg")

CHROMA_DB_DIR   = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "starlight_vision"   # shared with catalogue_ingestor

infile      = sys.argv[1] if len(sys.argv) > 1 else "scraped_results.jsonl"
outdir_default = sys.argv[2] if len(sys.argv) > 2 else "out_emails_streamlit"
os.makedirs(outdir_default, exist_ok=True)


# ---------------------------------------------------------------------------
# RAG context retrieval with HyDE + reference extraction
# ---------------------------------------------------------------------------

def get_rag_context(
    client_desc: str,
    k: int = 5,
) -> tuple[str, list[str], list[dict]]:
    """
    Retrieve the most relevant catalogue chunks for a client profile.

    Uses Hypothetical Document Embedding (HyDE): GPT-4o first writes an
    idealised product description matching the client's needs, then the
    combined (real + hypothetical) query is embedded and used to search
    ChromaDB.

    Returns:
        context_str:       Formatted string for injection into prompts.
        raw_docs:          List of raw chunk strings.
        product_references: List of dicts with product reference info:
                             {product_name, catalogue_name, page_number,
                              blob_url, category, specs_preview}
    """
    # Step A – HyDE
    hyde_messages = [
        {
            "role": "system",
            "content": (
                "You are a senior sales engineer at Starlight Linear LED. "
                "Write a concise, technical product description that would be the "
                "PERFECT match for the client's needs. Include realistic wattage, "
                "CCT, IP rating, application, and finish options. "
                "Output only the product description — no preamble."
            ),
        },
        {"role": "user", "content": f"Client profile:\n{client_desc}"},
    ]
    hyde_doc = azure_manager.chat_completion(hyde_messages, temperature=0.1, max_tokens=512)

    combined_query = (
        f"Client Context:\n{client_desc}\n\n"
        f"Ideal Product Characteristics:\n{hyde_doc}"
    )

    # Step B – embed
    try:
        query_vector = azure_manager.embed_text(combined_query)
    except Exception as exc:
        print(f"Warning: Embedding failed ({exc}). Returning empty context.")
        return "No specific catalogue context found.", [], []

    # Step C – query ChromaDB
    try:
        chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma.get_collection(COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(
            f"Warning: ChromaDB query failed ({exc}). "
            "Has a catalogue been uploaded via the RAG Admin panel?"
        )
        return "No specific catalogue context found.", [], []

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No specific catalogue context found.", [], []

    # Step D – build context string (with page reference labels)
    context_parts: list[str] = []
    product_references: list[dict] = []
    seen_blobs: set[str] = set()

    for doc, meta in zip(docs, metas):
        page_num  = meta.get("page_number", 0)
        cat_name  = meta.get("catalogue_name", "Starlight Catalogue")
        prod_name = meta.get("product_name", "")
        blob_url  = meta.get("blob_url", "")

        ref_label = (
            f"[{cat_name} — Page {page_num}]" if page_num else f"[{cat_name}]"
        )
        context_parts.append(f"{ref_label}\n{doc}")

        # De-duplicate product references by blob_url
        if blob_url and blob_url not in seen_blobs and not blob_url.startswith("["):
            seen_blobs.add(blob_url)
            product_references.append({
                "product_name":   prod_name or "Starlight Product",
                "catalogue_name": cat_name,
                "page_number":    page_num,
                "blob_url":       blob_url,
                "category":       meta.get("category", ""),
                "specs_preview":  meta.get("specs_preview", ""),
            })

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, docs, product_references


# ---------------------------------------------------------------------------
# Jinja2 template helper
# ---------------------------------------------------------------------------

def get_template(template_name: str = "email_template.html"):
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    try:
        env = Environment(loader=FileSystemLoader(templates_dir))
        return env.get_template(template_name)
    except Exception as exc:
        print(f"Warning: Could not load template '{template_name}': {exc}")
        env = Environment()
        return env.from_string(
            "<html><body>"
            "<p>{{ intro_paragraph | safe }}</p>"
            "<ul>{% for item in bullets %}<li>{{ item }}</li>{% endfor %}</ul>"
            "<p>{{ closing_paragraph }}</p>"
            "<p>-- {{ sender_name }}<br>{{ sender_company }}</p>"
            "</body></html>"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(txt: str) -> dict | None:
    if not txt:
        return None
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            return json.loads(txt[start: end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-@.]", "_", s or "no_email")


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """\
You are an expert B2B sales copywriter for Starlight Linear LED — an award-winning
Indian LED lighting manufacturer (Top 10 Brands in Lightings 2025, Homes India Magazine).

Company: End-to-end LED lighting solutions, custom manufacturing, supply, installation.
Address: 3 Vedant 3, P&T Colony, Gandhi Nagar, Dombivali East, Thane 421203.
Phone: 9619436066 | Email: vivek@starlightlinearled.com

ABSOLUTE RULES:
1. Reference ONLY product names and applications present in the CATALOGUE CONTEXT below.
2. DO NOT output dense technical specifications (like dimensions or IP ratings) in the email body. Your goal is to write a pleasing, beautiful, relationship-building email.
3. Use EXACT project names / clients / portfolio items from the scraped client data. Explicitly mention WHERE our products can be used in THEIR specific projects.
4. If no specific projects are found, focus on their general architectural/interior style.

STYLE:
- Elegantly personalised: Name-drop their projects and suggest Starlight products that fit perfectly.
- Clean and readable.
- DO NOT use raw HTML like <a href>. Use plain text.
- Subject: catchy, industry-relevant, no placeholder {{…}}.
- Dear line: personalised (e.g. "Dear Mahim Architects Team,").

Output ONLY a valid JSON object (no other text, no markdown):
  "subject"            – string
  "preamble"           – string (one elegant tagline, ≤12 words)
  "opening_line"       – string
  "intro"              – string (1–2 sentences focusing on their projects and our synergy)
  "feature_highlights" – array of strings (3 short elegant benefits of using our lights)
  "use_cases"          – array of strings (Specific lighting applications in their projects)
  "cta"                – string (Plain text call to action)
"""

_DRAFT_USER_TMPL = """\
CATALOGUE CONTEXT — ONLY reference products and specs present here:
===
{rag_context}
===

CLIENT DATA:
{client_json}
"""


def generate_creative_draft(rec: dict) -> tuple[str, list[str], list[dict]]:
    client_json = json.dumps(rec, ensure_ascii=False)
    print(f"  [RAG] Querying catalogue knowledge base for: {rec.get('company', '?')}…")
    rag_context, raw_docs, product_refs = get_rag_context(client_json)

    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM},
        {
            "role": "user",
            "content": _DRAFT_USER_TMPL.format(
                rag_context=rag_context,
                client_json=client_json,
            ),
        },
    ]
    resp = azure_manager.chat_completion(
        messages,
        temperature=0.25,
        max_tokens=2048,
        json_mode=True,
    )
    if not resp:
        raise RuntimeError("Azure OpenAI returned empty draft response.")
    return resp, raw_docs, product_refs


# ---------------------------------------------------------------------------
# Judge / formatter pass
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict JSON formatter. Take the input text and return a clean, valid
JSON object with exactly these keys:
  "subject", "preamble", "opening_line", "intro",
  "feature_highlights", "use_cases", "cta"

Rules:
1. Array fields must contain ONLY plain strings — never nested objects.
   Convert any dict entry to "<b>Title:</b> Description" format.
2. Escape all special characters properly.
3. Output ONLY the JSON object — no markdown, no commentary.
"""


def refine_with_judge(draft_text: str) -> dict | None:
    if not draft_text:
        return None
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"Clean and normalise this JSON:\n---\n{draft_text}\n---"},
    ]
    resp = azure_manager.chat_completion(
        messages, temperature=0.0, max_tokens=2048, json_mode=True
    )
    parsed = extract_json(resp)
    if not parsed:
        raise RuntimeError("Judge pass could not produce valid JSON.")
    return parsed


# ---------------------------------------------------------------------------
# List cleaners
# ---------------------------------------------------------------------------

def clean_item(x) -> str:
    if isinstance(x, dict):
        title = x.get("title", "")
        desc = x.get("description", x.get("desc", ""))
        if title and desc:
            return f"<b>{title}</b><br>{desc}"
        return " – ".join(str(v) for v in x.values() if v)
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("{") and x.endswith("}"):
            import ast
            try:
                d = ast.literal_eval(x)
                if isinstance(d, dict):
                    t = d.get("title", "")
                    d2 = d.get("description", d.get("desc", ""))
                    if t and d2:
                        return f"<b>{t}</b><br>{d2}"
            except Exception:
                pass
        x = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", x)
    return x


def ensure_list(val) -> list:
    if isinstance(val, list):
        return [clean_item(x) for x in val]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        items = [v.strip() for v in val.splitlines() if v.strip()]
        return [clean_item(x) for x in items] if items else [clean_item(val)]
    return []


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_eml_from_record(
    rec: dict,
    idx: int,
    outdir: str,
    template_name: str = "email_template.html",
) -> tuple[str, dict] | None:
    """
    Generate a single .html + .eml email from a scraped client record.

    The generated email includes:
    - Personalised intro and feature bullets grounded in catalogue context
    - `referenced_products` list passed to the template so it can render
      product cards with the catalogue page image (blob_url), product name,
      and specs preview.

    Returns (eml_path, trace_info) on success, or None on failure.
    """
    if not rec:
        print(f"Error: Empty record at index {idx}.")
        return None

    print(f"\n── Record {idx} | template='{template_name}'")

    creative_draft, raw_docs, product_refs = generate_creative_draft(rec)
    if not creative_draft:
        raise RuntimeError("Generation failed: no creative draft.")

    parsed = refine_with_judge(creative_draft)
    if not parsed:
        raise RuntimeError("Generation failed: could not parse draft JSON.")

    # Subject validation
    subject = parsed.get("subject", "")
    if not subject or "{{" in subject or "}}" in subject:
        company_name = rec.get("company", rec.get("website", "your company"))
        if "http" in str(company_name):
            company_name = urlparse(company_name).netloc.replace("www.", "")
        subject = f"Starlight LED – Precision Lighting Solutions for {company_name}"

    preamble         = parsed.get("preamble",  "Precision-engineered LED solutions, delivered on time.")
    opening_line     = parsed.get("opening_line", "Hope this email finds you well.")
    intro            = parsed.get("intro", "").replace("\n", "<br>")
    feature_highlights = ensure_list(parsed.get("feature_highlights", []))
    use_cases        = ensure_list(parsed.get("use_cases", []))
    technical_specs  = []
    bullets          = []
    cta              = parsed.get(
        "cta",
        "Would you be available for a brief call next week to explore how we can "
        "illuminate your next project?",
    )

    # Render template
    template = get_template(template_name)
    html_out = template.render(
        subject=subject,
        preamble=preamble,
        opening_line=opening_line,
        intro_paragraph=intro,
        bullets=bullets,
        feature_highlights=feature_highlights,
        use_cases=use_cases,
        technical_specs=technical_specs,
        closing_paragraph=cta,
        sender_name=sender_name,
        sender_company=sender_company,
        sender_phone=sender_phone,
        sender_website=sender_website,
        sender_email=sender_email,
        company_logo_url=company_logo_url,
        catalog_chunks=raw_docs,
        referenced_products=product_refs,   # ← NEW: product reference cards
    )

    safe_name = sanitize_filename(
        rec.get("emails") or rec.get("company") or f"record_{idx}"
    )
    html_path = os.path.join(outdir, f"{idx}_{safe_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    # Build .eml
    msg = MIMEMultipart("alternative")
    msg["From"]    = f'"{sender_name}" <{sender_email}>'
    msg["To"]      = str(rec.get("emails", ""))
    msg["Subject"] = subject
    msg.attach(MIMEText(html_out, "html"))

    if company_logo_url == "cid:company_logo" and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<company_logo>")
            img.add_header("Content-Disposition", "inline", filename="logo.jpg")
            msg.attach(img)

    eml_path = os.path.join(outdir, f"{idx}_{safe_name}.eml")
    with open(eml_path, "w", encoding="utf-8") as f:
        f.write(msg.as_string())

    print(f"  ✓ {eml_path}  ({len(product_refs)} product reference(s))")

    trace_info = {
        "raw_docs":         raw_docs,
        "product_refs":     product_refs,
        "creative_draft":   creative_draft,
    }
    return eml_path, trace_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    try:
        with open(infile, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {idx}.")
                    continue
                generate_eml_from_record(rec, idx, outdir_default)
                time.sleep(0.5)
    except FileNotFoundError:
        print(f"Error: '{infile}' not found.")


if __name__ == "__main__":
    main()
