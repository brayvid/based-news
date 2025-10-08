# based-news/backfill_history.py

import os
import json
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo

print("--- HISTORY BACKFILL SCRIPT STARTED ---")

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("FATAL: DATABASE_URL not found in your .env file. Exiting.")
    exit()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "public", "digest-manifest.json")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

if not os.path.exists(MANIFEST_PATH):
    print(f"FATAL: Manifest file not found at {MANIFEST_PATH}. Cannot proceed.")
    exit()

# --- 2. CONNECT TO DATABASE AND CLEAR EXISTING DATA (IMPORTANT!) ---
try:
    print("Connecting to the database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("SUCCESS: Connected to the database.")

    # This command clears all existing data to prevent duplicates if you run this script multiple times.
    # RESTART IDENTITY resets the ID counter (1, 2, 3...).
    # CASCADE removes dependent data in the 'articles' table.
    print("Clearing existing 'digests' and 'articles' tables to prevent duplicates...")
    cur.execute("TRUNCATE TABLE digests, articles RESTART IDENTITY CASCADE;")
    print("SUCCESS: Tables cleared.")

except Exception as e:
    print(f"FATAL: Could not connect to or clear the database. Error: {e}")
    exit()


# --- 3. READ AND PARSE OLD DIGESTS ---
try:
    print(f"Reading manifest file from {MANIFEST_PATH}")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # The manifest is newest-to-oldest. We reverse it to insert oldest-to-newest.
    manifest.reverse()
    print(f"Found {len(manifest)} historical digests to process.")

    total_articles_migrated = 0

    # Loop through each historical digest, from oldest to newest
    for entry in manifest:
        html_path = os.path.join(PUBLIC_DIR, entry['file'])
        digest_timestamp = datetime.fromisoformat(entry['timestamp'])
        
        if not os.path.exists(html_path):
            print(f"  -> WARNING: File not found, skipping: {html_path}")
            continue

        print(f"\nProcessing digest from {digest_timestamp.strftime('%Y-%m-%d %H:%M')}...")
        
        with open(html_path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'lxml')

        # Create the digest entry in the DB and get its new ID
        cur.execute("INSERT INTO digests (created_at) VALUES (%s) RETURNING id", (digest_timestamp,))
        digest_id = cur.fetchone()[0]
        print(f"  -> Created digest entry with new ID: {digest_id}")
        
        articles_to_insert = []
        display_order = 0
        current_topic = "General"

        # Find all topic headers (h3) and article paragraphs (p)
        for element in soup.find_all(['h3', 'p']):
            if element.name == 'h3':
                current_topic = element.get_text(strip=True)
            
            elif element.name == 'p':
                link_tag = element.find('a')
                if not link_tag:
                    continue
                
                title = link_tag.get_text(strip=True)
                link = link_tag.get('href', '#')

                # We use the digest's timestamp for all articles within it for consistency
                pub_date = digest_timestamp

                articles_to_insert.append((
                    digest_id,
                    current_topic,
                    title,
                    link,
                    pub_date,
                    display_order
                ))
                display_order += 1
        
        if articles_to_insert:
            extras.execute_values(
                cur,
                "INSERT INTO articles (digest_id, topic, title, link, pub_date, display_order) VALUES %s",
                articles_to_insert
            )
            print(f"  -> Inserted {len(articles_to_insert)} articles into the database.")
            total_articles_migrated += len(articles_to_insert)

    # --- 4. FINALIZE ---
    print("\nCommitting all changes to the database...")
    conn.commit()
    print("SUCCESS: Commit complete.")
    print(f"\n--- BACKFILL COMPLETE ---")
    print(f"Successfully migrated {len(manifest)} digests and {total_articles_migrated} articles.")

except Exception as e:
    print(f"\nFATAL: An error occurred during processing. Rolling back changes. Error: {e}")
    conn.rollback()
finally:
    if conn:
        cur.close()
        conn.close()
        print("Database connection closed.")