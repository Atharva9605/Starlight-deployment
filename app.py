"""
Starlight AI-CRM Mailer — Premium Streamlit UI.
Ultra-Modern Design System with Cyberpunk & Glassmorphism aesthetics.
"""

import streamlit as st
import pandas as pd
import os
import time
import shutil
import base64
from pathlib import Path

import chromadb

# Import project modules
from scraper import scrape_and_process
from generator_v2 import generate_eml_from_record
from send_eml_gsuite import send_email_gsuite
from rag_uploader import process_pdf_to_chroma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "starlight_vision"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Starlight AI-CRM Mailer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ultra-Premium "Starlight" Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;600;800&display=swap');

  :root {
    --primary: #8B5CF6;
    --primary-glow: rgba(139, 92, 246, 0.3);
    --secondary: #06B6D4;
    --accent: #F43F5E;
    --bg-dark: #0F172A;
    --bg-darker: #020617;
    --card-bg: rgba(30, 41, 59, 0.7);
    --border: rgba(255, 255, 255, 0.1);
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --glass: rgba(255, 255, 255, 0.03);
  }

  /* Core Layout */
  .stApp {
    background: radial-gradient(circle at 50% 0%, #1E1B4B 0%, #020617 100%);
    color: var(--text-main);
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  .block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
  }

  /* Hide Elements */
  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar - Cyberpunk Control Panel */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
    border-right: 1px solid var(--border);
    box-shadow: 10px 0 30px rgba(0,0,0,0.5);
  }
  
  [data-testid="stSidebar"] section {
    padding-top: 2rem;
  }

  /* Glassmorphism Cards */
  div[data-testid="stMetric"], .stCard {
    background: var(--card-bg);
    backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: var(--primary);
    box-shadow: 0 20px 40px var(--primary-glow);
  }

  /* Workflow Visualizer */
  .workflow-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 3rem 0;
    padding: 2rem;
    background: var(--glass);
    border-radius: 30px;
    border: 1px solid var(--border);
    position: relative;
  }
  .workflow-step {
    flex: 1;
    text-align: center;
    position: relative;
    z-index: 2;
    transition: all 0.6s ease;
    opacity: 0.3;
  }
  .workflow-step.active {
    opacity: 1;
    transform: scale(1.1);
  }
  .step-icon {
    width: 50px;
    height: 50px;
    background: #1E293B;
    border: 2px solid var(--border);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-weight: 800;
    font-size: 1.2rem;
    color: var(--text-muted);
    transition: all 0.4s ease;
  }
  .workflow-step.active .step-icon {
    background: var(--primary);
    border-color: #C084FC;
    color: white;
    box-shadow: 0 0 30px var(--primary-glow);
  }
  .step-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'Outfit', sans-serif;
  }

  /* Terminal Log */
  .terminal-window {
    background: #000;
    border: 1px solid #334155;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    margin-top: 2rem;
  }
  .terminal-header {
    background: #1E293B;
    padding: 10px 15px;
    display: flex;
    gap: 8px;
    border-bottom: 1px solid #334155;
  }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot-red { background: #FF5F56; }
  .dot-yellow { background: #FFBD2E; }
  .dot-green { background: #27C93F; }
  
  .terminal-log {
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    height: 450px;
    overflow-y: auto;
    color: #10B981;
    line-height: 1.6;
    background: linear-gradient(180deg, #000 0%, #050505 100%);
  }
  .log-entry { margin-bottom: 6px; border-left: 2px solid transparent; padding-left: 10px; }
  .log-entry:hover { border-left-color: var(--primary); background: rgba(139, 92, 246, 0.05); }
  .log-ts { color: #475569; font-size: 0.75rem; margin-right: 12px; }
  .log-success { color: #34D399; }
  .log-error { color: #F87171; }

  /* Hero Section */
  .hero-container {
    padding: 4rem 0 2rem;
    text-align: center;
    position: relative;
  }
  .hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 4.5rem;
    font-weight: 900;
    margin-bottom: 1rem;
    background: linear-gradient(to right, #FFF, #A78BFA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
    line-height: 1;
  }
  .hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-muted);
    max-width: 700px;
    margin: 0 auto 2rem;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, #6D28D9 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 700;
    transition: all 0.3s ease;
    width: 100%;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px var(--primary-glow);
  }
  .stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px var(--primary-glow);
    border: none !important;
  }

  /* Tabs Customization */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5);
    padding: 8px;
    border-radius: 16px;
    gap: 10px;
    border: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    color: var(--text-muted) !important;
    border: none !important;
    transition: all 0.3s ease;
  }
  .stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
    box-shadow: 0 4px 12px var(--primary-glow);
  }

  /* Custom Status Badges */
  .badge {
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .badge-success { background: rgba(16,185,129,0.1); color: #10B981; border: 1px solid #10B981; }
  .badge-error { background: rgba(239,68,68,0.1); color: #EF4444; border: 1px solid #EF4444; }

  /* Input Fields */
  .stTextInput input, .stSelectbox select {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: white !important;
  }
  
  .streamlit-expanderHeader {
    background: var(--glass) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 10px;
  }
</style>

<script>
  // Advanced Terminal Auto-scroll
  const scrollTerminal = () => {
    const logs = document.querySelector('.terminal-log');
    if (logs) {
      logs.scrollTo({ top: logs.scrollHeight, behavior: 'smooth' });
    }
  };
  const observer = new MutationObserver(scrollTerminal);
  document.addEventListener('DOMContentLoaded', () => {
    const target = document.body;
    observer.observe(target, { childList: true, subtree: true });
  });
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_kb_status():
    """Returns number of chunks and list of catalogue names in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
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

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "df": None,
        "processing": False,
        "run_log": [],
        "output_dir": "out_emails_streamlit",
        "latest_eml_html": "",
        "sender_email": os.getenv("GSUITE_DELEGATED_USER", "your-email@gsuite.com"),
        "recipient_override": "",
        "selected_template": "email_template.html",
        "admin_traces": [],
        "current_step": 0,   # 0=setup, 1=processing, 2=review, 3=done
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def add_log(message, type="info"):
    """Add a timestamped log message."""
    ts = time.strftime("%H:%M:%S")
    st.session_state.run_log.append({"ts": ts, "msg": message, "type": type})
    if len(st.session_state.run_log) > 100:
        st.session_state.run_log.pop(0)

# Initialize
initialize_session_state()

# ---------------------------------------------------------------------------
# SIDEBAR — Cyberpunk Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo Area
    logo_path = "starlight.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(
            '<div style="text-align:center; padding:2rem 0; background:var(--glass); border-radius:20px; border:1px solid var(--border);">'
            '<span style="font-family:Outfit; font-size:1.8rem; font-weight:900; background:linear-gradient(45deg, #8B5CF6, #06B6D4); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">STARLIGHT</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Knowledge Base Status
    st.markdown("### 📚 Knowledge Base")
    kb_chunks, kb_catalogues = get_kb_status()

    if kb_chunks > 0:
        st.success(f"**{kb_chunks}** Chunks Indexed")
        with st.expander("View Catalogues"):
            for cat in kb_catalogues:
                st.markdown(f"• {cat}")
    else:
        st.warning("KB is Empty")

    st.markdown("---")

    # Mailer Settings
    st.markdown("### ⚙️ Mailer Settings")

    uploaded_file = st.file_uploader(
        "Upload Client Leads (Excel)",
        type=["xlsx", "xls"],
        key="excel_uploader",
    )

    st.session_state.sender_email = st.text_input(
        "GSuite Sender",
        value=st.session_state.sender_email,
    )

    st.session_state.recipient_override = st.text_input(
        "Recipient Override",
        help="Redirect all emails to this address for testing.",
        value=st.session_state.recipient_override,
    )

    template_options = {
        "✨ Modern Soft": "email_template.html",
        "📄 Minimalist": "email_template_minimalist.html",
        "🔥 Bold & Vibrant": "email_template_bold.html",
    }
    selected_template_label = st.selectbox(
        "Email Template",
        options=list(template_options.keys()),
    )
    st.session_state.selected_template = template_options[selected_template_label]

# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------

# Hero Section
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Starlight AI Mailer</div>
    <div class="hero-subtitle">Next-generation CRM automation powered by Azure OpenAI. <br>Transform leads into conversations with personalized, high-conversion emails.</div>
</div>
""", unsafe_allow_html=True)

# Workflow Progress
steps = ["Configure", "Process", "Review", "Success"]
step_cols = st.columns(len(steps))
st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
for i, step in enumerate(steps):
    active_class = "active" if i == st.session_state.current_step else ""
    st.markdown(f"""
    <div class="workflow-step {active_class}">
        <div class="step-icon">{i+1}</div>
        <div class="step-label">{step}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different views
tab_main, tab_logs, tab_kb = st.tabs(["🚀 Dashboard", "📜 System Logs", "📖 Manage KB"])

with tab_main:
    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Catalogues", len(kb_catalogues))
    with m2:
        st.metric("Total Leads", len(st.session_state.df) if st.session_state.df is not None else 0)
    with m3:
        st.metric("Processed", 0) # Placeholder
    with m4:
        st.metric("Status", "Idle" if not st.session_state.processing else "Running")

    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file:
        if st.session_state.df is None:
            st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.df["Status"] = "Pending"
            add_log("Leads file uploaded successfully.")

        st.markdown("### 📋 Leads Preview")
        st.dataframe(st.session_state.df, use_container_width=True)

        col_run, col_stop = st.columns([1, 4])
        with col_run:
            if st.button("🚀 START MAILING PIPELINE"):
                st.session_state.processing = True
                st.session_state.current_step = 1
                st.rerun()

    else:
        st.info("Please upload a leads file in the sidebar to begin.")

with tab_logs:
    st.markdown("""
    <div class="terminal-window">
        <div class="terminal-header">
            <div class="dot dot-red"></div>
            <div class="dot dot-yellow"></div>
            <div class="dot dot-green"></div>
            <div style="margin-left: 10px; font-size: 0.7rem; color: #94A3B8; font-family: monospace;">system_logs.sh</div>
        </div>
        <div class="terminal-log">
    """, unsafe_allow_html=True)
    
    for log in st.session_state.run_log:
        log_type_class = f"log-{log['type']}"
        icon = "✓" if log['type'] == "success" else "⚠" if log['type'] == "error" else "ℹ"
        st.markdown(f"""
            <div class="log-entry {log_type_class}">
                <span class="log-ts">[{log['ts']}]</span>
                <span class="log-icon">{icon}</span>
                {log['msg']}
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div></div>", unsafe_allow_html=True)

with tab_kb:
    st.markdown("### 📖 Knowledge Base Management")
    st.write("Upload PDF/Doc catalogues to teach the AI about your products.")
    
    kb_files = st.file_uploader("Add to Knowledge Base", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("📥 INGEST CATALOGUES"):
        if kb_files:
            with st.spinner("Analyzing documents and indexing vector space..."):
                # Call rag_uploader
                for f in kb_files:
                    temp_path = f"temp_{f.name}"
                    with open(temp_path, "wb") as buffer:
                        buffer.write(f.read())
                    process_pdf_to_chroma(temp_path, f.name)
                    os.remove(temp_path)
                    add_log(f"Ingested: {f.name}", "success")
            st.success("Ingestion Complete!")
            st.rerun()

# ---------------------------------------------------------------------------
# PIPELINE LOGIC (Runs if processing is True)
# ---------------------------------------------------------------------------

if st.session_state.processing:
    # This is a simplified version of the processing loop
    # In a real app, you'd run this in a fragment or with better state management
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    df = st.session_state.df
    total = len(df)
    
    for index, row in df.iterrows():
        website = row.get("Website", row.get("website", ""))
        if not website: continue
        
        status_text.markdown(f"**Currently Processing:** `{website}`")
        progress_bar.progress((index + 1) / total)
        
        try:
            # Step 1: Scrape & Process
            add_log(f"Scraping website: {website}")
            scrape_data = scrape_and_process(website)
            
            # Step 2: Generate Email
            add_log(f"Generating personalized email for: {website}")
            eml_path, eml_html = generate_eml_from_record(
                row, 
                scrape_data, 
                template_name=st.session_state.selected_template
            )
            
            # Step 3: Send Email
            add_log(f"Sending email for: {website}")
            send_success = send_email_gsuite(
                eml_path,
                sender_email=st.session_state.sender_email,
                recipient_email=st.session_state.recipient_override if st.session_state.recipient_override else row.get("Email", "")
            )
            
            if send_success:
                add_log(f"Success: Email sent to {website}", "success")
                df.at[index, "Status"] = "Sent"
            else:
                add_log(f"Failed to send email to {website}", "error")
                df.at[index, "Status"] = "Error"
                
        except Exception as e:
            add_log(f"Error processing {website}: {str(e)}", "error")
            df.at[index, "Status"] = f"Error: {str(e)[:20]}"
        
        time.sleep(1) # Small delay for UX
        
    st.session_state.processing = False
    st.session_state.current_step = 3
    st.success("Pipeline Execution Finished!")
    time.sleep(2)
    st.rerun()
