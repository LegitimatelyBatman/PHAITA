#!/usr/bin/env python3
"""Command line interface for scraping forum data.

The script supports scraping Reddit and Patient.info forums using the
:class:`phaita.data.forum_scraper.ForumScraper`. Posts are cached to disk so they
can be reused by offline components and the integration tests.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from phaita.data.forum_scraper import ForumScraper

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape health forum posts")
    parser.add_argument(
        "--source",
        choices=["reddit", "patient", "all"],
        default="all",
        help="Which forum source to scrape",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=25,
        help="Maximum number of posts to fetch per source",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory where scraped posts should be cached",
    )
    parser.add_argument(
        "--reddit-cache",
        default="reddit_posts.json",
        help="Filename used to persist Reddit posts",
    )
    parser.add_argument(
        "--patient-cache",
        default="patient_info_posts.json",
        help="Filename used to persist Patient.info posts",
    )
    return parser.parse_args()


def scrape(scraper: ForumScraper, source: str, max_posts: int, filename: str) -> None:
    if source == "reddit":
        posts = scraper.scrape_reddit_health(max_posts=max_posts)
    else:
        posts = scraper.scrape_patient_info(max_posts=max_posts)
    scraper.save_posts(posts, filename)
    logger.info("Scraped %s posts from %s", len(posts), source)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    scraper = ForumScraper(cache_dir=args.cache_dir)

    if args.source in ("reddit", "all"):
        scrape(scraper, "reddit", args.max_posts, args.reddit_cache)

    if args.source in ("patient", "all"):
        scrape(scraper, "patient", args.max_posts, args.patient_cache)

    cache_dir: Optional[Path] = scraper.cache_dir if hasattr(scraper, "cache_dir") else None
    if cache_dir is not None:
        logger.info("Cached posts in %s", cache_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
