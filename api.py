import os
import json
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import chromadb

# Import project modules
from scraper import scrape_and_process
from generator_v2 import generate_eml_from_record
from send_eml_gsuite import send_email_gsuite
from rag_uploader import process_pdf_to_chroma

app = FastAPI(title="Starlight AI-CRM Mailer API")

# Allow CORS for the React frontend (Lovable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Vercel/Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "starlight_vision"

def get_kb_status_local():
    try:
        chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            results = collection.get(limit=min(count, 500), include=["metadatas"])
            catalogues = set()
            for m in (results.get("metadatas") or []):
                name = m.get("catalogue_name", "")
                if name:
                    catalogues.add(name)
            return count, list(catalogues)
        return count, []
    except Exception:
        return 0, []

@app.get("/api/kb-status")
async def get_kb():
    chunks, catalogues = get_kb_status_local()
    return {"chunks": chunks, "catalogues": catalogues}

@app.post("/api/upload-catalogues")
async def upload_catalogues(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            class MockFile:
                name = file.filename
                def read(self):
                    with open(tmp_path, "rb") as f:
                        return f.read()
            
            mock_f = MockFile()
            # process_pdf_to_chroma runs synchronously, might block event loop slightly, but acceptable for this usecase.
            success, msg = process_pdf_to_chroma(mock_f, progress_callback=lambda f, m: None)
            results.append({"filename": file.filename, "success": success, "message": msg})
        finally:
            os.remove(tmp_path)
    return {"results": results}

@app.post("/api/upload-leads")
async def upload_leads(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        df = pd.read_excel(tmp_path)
        if "website" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Excel file must have a 'website' column."})
        
        df['status'] = "⏳ Pending"
        df['notes'] = ""
        records = df.fillna("").to_dict(orient="records")
        return {"leads": records}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        os.remove(tmp_path)


class ProcessRequest(BaseModel):
    leads: List[dict]
    sender_email: str
    recipient_override: Optional[str] = ""
    template: str = "email_template.html"
    delay: int = 3

@app.post("/api/process-stream")
async def process_leads(req: ProcessRequest):
    """
    Initiates processing of leads.
    Returns a Server-Sent Events (SSE) stream.
    """
    async def event_stream():
        outdir = "out_emails_api"
        os.makedirs(outdir, exist_ok=True)
        
        for idx, row in enumerate(req.leads):
            website = row.get('website', '')
            if not website:
                continue
                
            yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': '⚙️ Processing...'})}\n\n"
            
            try:
                yield f"data: {json.dumps({'type': 'log', 'message': f'Scraping: {website}'})}\n\n"
                
                # Run sync scraper in a thread to not block asyncio
                scraped_data = await asyncio.to_thread(scrape_and_process, website)
                if not scraped_data:
                    raise Exception("Scraper failed to return data.")
                
                if req.recipient_override:
                    scraped_data["emails"] = req.recipient_override
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Generating EML for: {website}'})}\n\n"
                eml_path, trace_info = await asyncio.to_thread(
                    generate_eml_from_record, scraped_data, idx + 1, outdir, req.template
                )
                
                if not eml_path or not os.path.exists(eml_path):
                    raise Exception("EML generation failed.")
                
                # Load HTML preview
                html_path = Path(eml_path).with_suffix(".html")
                if html_path.exists():
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    yield f"data: {json.dumps({'type': 'preview_html', 'html': html_content})}\n\n"
                
                # Send RAG Trace Info
                yield f"data: {json.dumps({'type': 'rag_trace', 'data': {'company': scraped_data.get('company', website), 'trace_info': trace_info}})}\n\n"
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Sending email via GSuite for {website}'})}\n\n"
                send_success = await asyncio.to_thread(send_email_gsuite, eml_path, req.sender_email)
                
                if not send_success:
                    raise Exception("GSuite sending function returned False.")
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Email sent successfully for: {website}'})}\n\n"
                yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': '✅ Sent'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'log', 'message': f'Error processing {website}: {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': f'❌ Failed: {str(e)[:50]}...'})}\n\n"
            
            await asyncio.sleep(req.delay)
            
        yield f"data: {json.dumps({'type': 'done', 'message': 'Processing complete!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
