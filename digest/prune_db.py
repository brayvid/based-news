# Author: Blake Rayvid <https://github.com/brayvid/based-news>

import os
import sys
import logging
from dotenv import load_dotenv
import psycopg2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

DATABASE_URL = os.environ.get("DATABASE_URL")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", 30))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def prune_database():
    if not DATABASE_URL:
        logging.error("DATABASE_URL environment variable is missing.")
        sys.exit(1)

    conn = None
    try:
        logging.info(f"Connecting to database to prune records older than {RETENTION_DAYS} days...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # 1. Delete articles tied to digests older than RETENTION_DAYS (or published before cutoff)
        cur.execute(
            """
            WITH deleted_articles AS (
                DELETE FROM articles
                WHERE digest_id IN (
                    SELECT id FROM digests WHERE created_at < NOW() - (%s || ' days')::INTERVAL
                )
                OR pub_date < NOW() - (%s || ' days')::INTERVAL
                RETURNING id
            )
            SELECT COUNT(*) FROM deleted_articles;
            """,
            (RETENTION_DAYS, RETENTION_DAYS),
        )
        articles_deleted = cur.fetchone()[0]

        # 2. Delete digests older than RETENTION_DAYS
        cur.execute(
            """
            WITH deleted_digests AS (
                DELETE FROM digests
                WHERE created_at < NOW() - (%s || ' days')::INTERVAL
                RETURNING id
            )
            SELECT COUNT(*) FROM deleted_digests;
            """,
            (RETENTION_DAYS,),
        )
        digests_deleted = cur.fetchone()[0]

        # 3. Clean up any empty/orphaned digests
        cur.execute(
            """
            WITH empty_digests AS (
                DELETE FROM digests
                WHERE id NOT IN (SELECT DISTINCT digest_id FROM articles WHERE digest_id IS NOT NULL)
                RETURNING id
            )
            SELECT COUNT(*) FROM empty_digests;
            """
        )
        orphaned_digests_deleted = cur.fetchone()[0]

        # Commit the transaction so deletions are permanent
        conn.commit()
        cur.close()

        logging.info(
            f"Pruning complete: "
            f"Deleted {articles_deleted} articles, "
            f"{digests_deleted} expired digests, and "
            f"{orphaned_digests_deleted} empty digests."
        )

    except Exception as e:
        if conn:
            conn.rollback()
        logging.critical(f"Database pruning failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    prune_database()
    sys.exit(0)