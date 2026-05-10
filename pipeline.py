import os
import time
import pandas as pd
from pathlib import Path

from scraper import scrape_and_process
from generator_v2 import generate_eml_from_record
from send_eml_gsuite import send_email_gsuite

INPUT_EXCEL = "test_input.xlsx"
OUTPUT_DIR = "out_emails_streamlit"
WAIT_TIMEOUT = 60  # seconds to wait for .eml generation
SLEEP_BETWEEN_RECORDS = 3

def wait_for_eml(output_dir, before_files, timeout=WAIT_TIMEOUT):
    """Wait until a new .eml file appears in output_dir."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        eml_files = sorted(Path(output_dir).glob("*.eml"), key=os.path.getmtime)
        new_files = [f for f in eml_files if f not in before_files]
        if new_files:
            return new_files[-1]
        time.sleep(2)
    return None

def process_record(website, record_index):
    print(f"\n🟢 Processing Record {record_index+1}: {website}")

    # Step 1: Scraping
    print("🔹 Step 1: Scraping...")
    scraped = scrape_and_process(website)
    if not scraped:
        print("❌ Scraper failed, skipping record.")
        return

    client_name = scraped.get("website").split("//")[-1].split(".")[0].title()
    recipient_email = scraped.get("emails") or scraped.get("contact_email") or None
    print(f"➡ Client: {client_name}")
    print(f"➡ Recipient Email: {recipient_email}")

    # Step 2: Generate .eml
    print("🔹 Step 2: Generating .eml via Azure OpenAI...")
    eml_path = generate_eml_from_record(scraped, record_index + 1, OUTPUT_DIR, "email_template.html")

    if not eml_path or not Path(eml_path).exists():
        print("❌ .eml generation failed.")
        return

    print(f"✅ .eml generated: {eml_path.name if hasattr(eml_path, 'name') else Path(eml_path).name}")

    # Step 3: Send email via GSuite
    print("🔹 Step 3: Sending email via GSuite...")
    # FIXED: Changed to match send_eml_gsuite.py parameter name
    send_email_gsuite(eml_path, sender_email="vivek@starlightlinearled.com")
    print(f"✅ Email sent successfully to {recipient_email}")

def main():
    print("\n🚀 Starting AI-CRM-Mailer Full Pipeline...\n")

    if not os.path.exists(INPUT_EXCEL):
        print("❌ Excel file not found.")
        return

    df = pd.read_excel(INPUT_EXCEL)
    if df.empty:
        print("❌ No data found in Excel file.")
        return

    websites = df['website'].dropna().astype(str).tolist()
    print(f"✅ Total records found: {len(websites)}")

    for i, website in enumerate(websites):
        process_record(website, i)
        time.sleep(SLEEP_BETWEEN_RECORDS)

    print("\n🎯 All records processed.\n")

if __name__ == "__main__":
    main()