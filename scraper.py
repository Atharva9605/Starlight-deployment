

import os
import re
import time
import json
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ---------------- Config ----------------
load_dotenv()
from azure_client import azure_manager

INPUT_EXCEL = ""      # Excel with column 'website'
MANUAL_WEBSITES = ["http://starlightlinearled.com/"]
OUTPUT_JSONL = "scraped_results.jsonl"
RAW_DIR = "raw_data"

TEST_LIMIT = None
RATE_LIMIT_SECONDS = 1.5
LLM_RATE_LIMIT_SECONDS = 1.0

os.makedirs(RAW_DIR, exist_ok=True)

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scraper")

# ---- HTTP session with retries ----
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})
retries = Retry(total=2, backoff_factor=1, status_forcelist=[429,500,502,503,504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ---- Azure OpenAI system prompt for website analysis ----
_SCRAPE_SYSTEM = """\
You are given text extracted from a company or organization's website.
Analyze carefully and extract the following, focusing on marketing and personalization for email outreach:

1. 3-5 sentence summary of what the company does (professional and clear).
2. Products/Services they offer (comma-separated).
3. Target audience (who they sell to).
4. Unique selling points (USPs).
5. Key phrases / achievements / positioning statements.
6. Tone & Style (how the company presents itself).

Return ONLY a valid JSON object with keys: summary, products_services, target_audience, usp, key_phrases, tone_style.
No extra explanation, no markdown fences.
"""


# ---------------- Helpers ----------------
def normalize_url(url: str) -> str:
    """Ensure the URL has a valid scheme."""
    url = url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def safe_get(url, timeout=12):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"Request failed for {url} → {e}")
        return ""

def clean_text_from_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)

def fetch_page_text(url):
    if not url:
        return ""
    html = safe_get(url)
    if not html:
        return ""
    return clean_text_from_html(html)

def save_raw_text(website_url, page_type, text):
    if not text:
        return ""
    domain = urlparse(website_url).netloc.replace(":", "_")
    folder = os.path.join(RAW_DIR, domain)
    os.makedirs(folder, exist_ok=True)
    filename = f"{page_type}.txt"
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def find_special_pages(base_url):
    about_keywords = r"(about|who|story|company|profile|vision|mission|team|our-story|what-we-do)"
    contact_keywords = r"(contact|reach|touch|connect|support|help|enquiry|inquiry|get-in-touch|reach-us|visit-us)"
    about, contact = None, None
    html = safe_get(base_url)
    if not html:
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().lower()
        full = urljoin(base_url, href)
        if not about and re.search(about_keywords, href):
            about = full
        if not contact and re.search(contact_keywords, href):
            contact = full
        if about and contact:
            break
    return about, contact

def extract_json_from_llm_output(raw_str):
    if not raw_str:
        return None
    m = re.search(r"\{.*\}", raw_str, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        s = m.group()
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*\]", "]", s)
        try:
            return json.loads(s)
        except Exception:
            return None

def invoke_chain_get_text(chunk):
    """Call Azure OpenAI to analyze the scraped website text."""
    messages = [
        {"role": "system", "content": _SCRAPE_SYSTEM},
        {"role": "user", "content": f"Website text:\n\n{chunk}"},
    ]
    return azure_manager.chat_completion(
        messages, temperature=0.0, max_tokens=2048, json_mode=True
    )

def analyze_text_with_llm(full_text):
    if not full_text.strip():
        return {k: "" for k in ["summary","products_services","target_audience","usp","key_phrases","tone_style"]}
    truncated = full_text[:15000]
    raw_out = invoke_chain_get_text(truncated)
    parsed = extract_json_from_llm_output(raw_out) or {"summary": raw_out}
    for k in ["summary","products_services","target_audience","usp","key_phrases","tone_style"]:
        val = parsed.get(k, "")
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val)
        elif not isinstance(val, str):
            val = str(val)
        parsed[k] = val.strip()
    return parsed

# ---------------- Core scraping ----------------
def scrape_and_process(website):
    website = normalize_url(website)  # ensure valid URL
    log.info(f"Processing: {website}")
    home_html = safe_get(website)
    if not home_html:
        log.info(f"  Unreachable: {website}")
        return None
    home_text = clean_text_from_html(home_html)

    about_url, contact_url = find_special_pages(website)
    about_text = fetch_page_text(about_url) if about_url else ""
    contact_text = fetch_page_text(contact_url) if contact_url else ""

    save_raw_text(website, "home", home_text)
    save_raw_text(website, "about", about_text)
    save_raw_text(website, "contact", contact_text)

    combined = " ".join([home_text, about_text, contact_text]).strip()
    if len(combined) > 30000:
        combined = combined[:30000]

    structured = analyze_text_with_llm(combined)

    emails = ", ".join(sorted(set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", combined))))
    phones = ", ".join(sorted(set(re.findall(r"\+?\d[\d\-\s]{7,}\d", combined))))

    return {
        "website": website,
        "about_url": about_url or "",
        "contact_url": contact_url or "",
        **structured,
        "emails": emails,
        "phones": phones
    }

def process_websites(input_excel=INPUT_EXCEL, manual_websites=MANUAL_WEBSITES, out_jsonl=OUTPUT_JSONL, test_limit=TEST_LIMIT):
    sites = []
    if input_excel and os.path.exists(input_excel):
        try:
            df = pd.read_excel(input_excel)
            sites.extend([s for s in df['website'].dropna().astype(str).tolist()])
        except Exception as e:
            log.warning(f"Could not read Excel '{input_excel}': {e}")
    if manual_websites:
        sites.extend(manual_websites)
    sites = list(dict.fromkeys([normalize_url(s) for s in sites if s and s.strip()]))

    processed_sites = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_sites.add(obj.get("website", ""))
                except:
                    pass

    sites = [s for s in sites if s not in processed_sites]
    if test_limit:
        sites = sites[:test_limit]

    total = len(sites)
    log.info(f"Starting: {total} new sites (skipped {len(processed_sites)})")

    reachable, unreachable = 0, 0
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx, site in enumerate(sites, start=1):
            log.info(f"[{idx}/{total}] {site}")
            row = scrape_and_process(site)
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                reachable += 1
            else:
                unreachable += 1
            time.sleep(RATE_LIMIT_SECONDS)

    log.info(f"Summary: reachable={reachable}, unreachable={unreachable}, total={total}")

if __name__ == "__main__":
    process_websites()
