"""Dry-run: fetch Zotero corpus from cloud and compare against local 527.

Only reads — does NOT fetch arXiv, run reranker, or send email.
Usage: uv run --project . scripts/dryrun_zotero.py
"""
import os
import sys
from collections import Counter
from datetime import datetime

import dotenv
from loguru import logger
from pyzotero import zotero

dotenv.load_dotenv()

LOCAL_BASELINE = 527  # from local SQLite, the three paper types with abstracts


def main():
    user_id = os.environ.get("ZOTERO_ID")
    api_key = os.environ.get("ZOTERO_KEY")
    if not user_id or not api_key:
        logger.error("ZOTERO_ID / ZOTERO_KEY not set in env")
        sys.exit(1)

    logger.info(f"Connecting to Zotero cloud as user {user_id}")
    zot = zotero.Zotero(user_id, "user", api_key)

    # Same filter as executor.fetch_zotero_corpus()
    logger.info("Fetching items of type journalArticle/conferencePaper/preprint ...")
    corpus = zot.everything(zot.items(itemType="conferencePaper || journalArticle || preprint"))
    logger.info(f"Cloud returned {len(corpus)} items (before abstract filter)")

    # Per-type breakdown
    type_counts = Counter(c["data"]["itemType"] for c in corpus)
    logger.info(f"By type: {dict(type_counts)}")

    # Abstract filter (same as executor)
    with_abstract = [c for c in corpus if c["data"]["abstractNote"] != ""]
    without_abstract = [c for c in corpus if c["data"]["abstractNote"] == ""]
    logger.info(f"With abstract:    {len(with_abstract)}")
    logger.info(f"Without abstract: {len(without_abstract)} (these are dropped by executor)")

    # DateAdded range (sanity)
    dates = []
    for c in with_abstract:
        try:
            dates.append(datetime.strptime(c["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, ValueError):
            pass
    if dates:
        logger.info(f"dateAdded range: {min(dates).date()}  ..  {max(dates).date()}")

    # Keys
    keys = {c["key"] for c in with_abstract}
    logger.info(f"Unique item keys: {len(keys)}")

    print()
    print("=" * 60)
    print(f"  CLOUD (post-abstract-filter): {len(with_abstract)}")
    print(f"  LOCAL BASELINE              : {LOCAL_BASELINE}")
    diff = len(with_abstract) - LOCAL_BASELINE
    verdict = "MATCH ✓" if diff == 0 else (f"cloud +{diff} more" if diff > 0 else f"cloud {diff} fewer")
    print(f"  VERDICT                     : {verdict}")
    print("=" * 60)

    if abs(diff) > 0 and abs(diff) <= 50:
        # Show recent cloud items not in local count window — informational only
        recent = sorted(with_abstract, key=lambda c: c["data"]["dateAdded"], reverse=True)[:5]
        logger.info("5 most recent cloud items:")
        for c in recent:
            logger.info(f"  {c['data']['dateAdded']} | {c['data']['itemType']:18s} | {c['data']['title'][:60]}")


if __name__ == "__main__":
    main()
