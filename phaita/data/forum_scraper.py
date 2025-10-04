"""Forum scraping and lay language mapping utilities.

This module provides production-ready clients for harvesting forum data from
Reddit and Patient.info. The :class:`ForumScraper` coordinates the scraping
pipeline, normalises the raw posts into :class:`ForumPost` records and offers a
simple caching layer so downstream components can operate on persisted data.

The implementation keeps the original dataclass shape that the rest of the
project expects while replacing the previous template-driven mock logic with
real HTTP/API clients. The clients honour rate limiting and authentication
requirements and can be swapped out (e.g. during tests) via dependency
injection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .icd_conditions import RespiratoryConditions

logger = logging.getLogger(__name__)

try:  # Optional dependency, only required when the Reddit client is used.
    import praw  # type: ignore
except Exception:  # pragma: no cover - handled dynamically at runtime.
    praw = None  # type: ignore

try:  # Optional dependency, only required when the Patient.info client is used.
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:  # Optional dependency, only required when the Patient.info client is used.
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


@dataclass
class ForumPost:
    """Represents a scraped forum post."""

    id: str
    title: str
    content: str
    timestamp: str
    forum_source: str
    lay_terms: List[str]
    extracted_symptoms: List[str]
    confidence_score: float


class LayLanguageMapper:
    """Maps between medical terminology and lay language."""

    def __init__(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Initialize with predefined mappings."""
        self._base_mappings = {
            "can't breathe": "dyspnea",
            "short of breath": "dyspnea",
            "breathless": "dyspnea",
            "wheezy": "wheezing",
            "whistling breath": "wheezing",
            "tight chest": "chest_tightness",
            "chest feels tight": "chest_tightness",
            "coughing up stuff": "productive_cough",
            "bringing up phlegm": "sputum_production",
            "chest hurts": "chest_pain",
            "burning chest": "chest_pain",
            "really tired": "fatigue",
            "wiped out": "fatigue",
            "exhausted": "fatigue",
            "stuffy nose": "nasal_congestion",
            "blocked nose": "nasal_congestion",
            "scratchy throat": "sore_throat",
            "throat hurts": "sore_throat",
            "can't catch my breath": "dyspnea",
            "gasping for air": "severe_dyspnea",
            "drowning feeling": "orthopnea",
            "hacking cough": "persistent_cough",
            "dry cough": "dry_cough",
            "hot": "fever",
            "burning up": "fever",
            "aching all over": "myalgia",
        }
        self.mappings: Dict[str, str] = {}
        self.reload_from_conditions(conditions or RespiratoryConditions.get_all_conditions())
        RespiratoryConditions.register_reload_hook(self.reload_from_conditions)

    def _build_reverse_mapping(self) -> None:
        self.reverse_mappings: Dict[str, List[str]] = {}
        for lay, medical in self.mappings.items():
            self.reverse_mappings.setdefault(medical, []).append(lay)

    def reload_from_conditions(self, conditions: Dict[str, Dict]) -> None:
        """Synchronise lay language mappings with the active condition set."""

        combined = dict(self._base_mappings)
        for condition in conditions.values():
            canonical_condition = condition["name"].lower().replace(" ", "_")
            combined.setdefault(condition["name"].lower(), canonical_condition)
            combined.setdefault(canonical_condition.replace("_", " "), canonical_condition)

            for symptom in condition.get("symptoms", []):
                combined.setdefault(symptom.replace("_", " "), symptom)
            for indicator in condition.get("severity_indicators", []):
                combined.setdefault(indicator.replace("_", " "), indicator)
            for lay_term in condition.get("lay_terms", []):
                combined.setdefault(lay_term.lower(), canonical_condition)

        self.mappings = combined
        self._build_reverse_mapping()

    def get_medical_term(self, lay_term: str) -> Optional[str]:
        return self.mappings.get(lay_term.lower())

    def get_lay_terms_for_medical(self, medical_term: str) -> List[str]:
        return self.reverse_mappings.get(medical_term, [])

    def update_mappings_from_posts(self, posts: Sequence[ForumPost]) -> None:
        for post in posts:
            for lay_term, medical_term in zip(post.lay_terms, post.extracted_symptoms):
                if lay_term.lower() not in self.mappings:
                    self.mappings[lay_term.lower()] = medical_term
        self._build_reverse_mapping()

    def save_mappings(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.mappings, handle, indent=2)

    def load_mappings(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as handle:
            self.mappings = json.load(handle)
        self._build_reverse_mapping()


class RateLimiter:
    """Simple time based rate limiter used by the HTTP/API clients."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self._last_invocation: float = 0.0

    def wait(self) -> None:
        """Sleep if the previous invocation happened too recently."""
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_invocation
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_invocation = time.monotonic()


class BaseForumClient:
    """Interface for forum clients."""

    def fetch_posts(self, max_posts: int) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError


class RedditClient(BaseForumClient):
    """Client that fetches posts from Reddit using :mod:`praw`."""

    def __init__(
        self,
        subreddit: str = "AskDocs",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        rate_limit_seconds: float = 1.0,
        praw_client: Optional[Any] = None,
    ) -> None:
        if praw_client is not None:
            self._reddit = praw_client
        else:
            if praw is None:  # pragma: no cover - depends on optional dependency
                raise RuntimeError(
                    "praw is required to scrape Reddit. Install the optional dependency "
                    "and provide API credentials via the ForumScraper configuration."
                )

            client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
            client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = user_agent or os.getenv("REDDIT_USER_AGENT")
            username = username or os.getenv("REDDIT_USERNAME")
            password = password or os.getenv("REDDIT_PASSWORD")

            missing = [
                ("REDDIT_CLIENT_ID", client_id),
                ("REDDIT_CLIENT_SECRET", client_secret),
                ("REDDIT_USER_AGENT", user_agent),
            ]
            missing_keys = [name for name, value in missing if not value]
            if missing_keys:
                raise RuntimeError(
                    "Missing Reddit API credentials: " + ", ".join(missing_keys)
                )

            self._reddit = praw.Reddit(  # type: ignore[call-arg]
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=username,
                password=password,
                check_for_async=False,
            )

        self.subreddit = subreddit
        self._rate_limiter = RateLimiter(rate_limit_seconds)

    def fetch_posts(self, max_posts: int) -> List[Dict[str, Any]]:
        """Fetch posts from the configured subreddit."""

        posts: List[Dict[str, Any]] = []
        subreddit = self._reddit.subreddit(self.subreddit)

        for submission in subreddit.hot(limit=max_posts * 2):
            # Honour rate limits and avoid stickied announcements.
            self._rate_limiter.wait()
            if getattr(submission, "stickied", False):
                continue

            created = getattr(submission, "created_utc", None)
            created_dt = (
                datetime.utcfromtimestamp(created)
                if created is not None
                else datetime.utcnow()
            )

            posts.append(
                {
                    "id": str(submission.id),
                    "title": submission.title or "",
                    "content": submission.selftext or submission.title or "",
                    "created_at": created_dt,
                }
            )

            if len(posts) >= max_posts:
                break

        logger.debug("Fetched %s Reddit posts", len(posts))
        return posts


class PatientInfoClient(BaseForumClient):
    """Client that scrapes Patient.info forum listings using :mod:`requests`."""

    def __init__(
        self,
        forum_paths: Optional[Sequence[str]] = None,
        base_url: str = "https://patient.info/forums",
        session: Optional[Any] = None,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        if requests is None or BeautifulSoup is None:  # pragma: no cover - optional deps
            raise RuntimeError(
                "requests and beautifulsoup4 are required to scrape Patient.info."
            )

        self.base_url = base_url.rstrip("/")
        self.forum_paths = list(forum_paths or ["/forums/categories/asthma"])
        self.session = session or requests.Session()
        self._rate_limiter = RateLimiter(rate_limit_seconds)

    def fetch_posts(self, max_posts: int) -> List[Dict[str, Any]]:
        """Fetch thread previews from the configured Patient.info forums."""

        records: List[Dict[str, Any]] = []
        for path in self.forum_paths:
            if len(records) >= max_posts:
                break

            url = f"{self.base_url}/{path.lstrip('/')}"
            self._rate_limiter.wait()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            records.extend(self._parse_thread_list(html))

        logger.debug("Fetched %s Patient.info posts", len(records))
        return records[:max_posts]

    @staticmethod
    def _parse_thread_list(html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        threads: List[Dict[str, Any]] = []

        for article in soup.select("[data-thread-id], article.thread"):
            thread_id = article.get("data-thread-id") or article.get("id")
            title_el = article.select_one("a[data-thread-title], h2 a, a.thread-link")
            summary_el = article.select_one(
                "[data-thread-excerpt], p, .thread-excerpt, .message"
            )
            time_el = article.find("time")

            title = (title_el.get_text(strip=True) if title_el else "").strip()
            summary = (summary_el.get_text(" ", strip=True) if summary_el else "").strip()
            timestamp = datetime.utcnow()
            if time_el and time_el.has_attr("datetime"):
                timestamp = _parse_datetime(time_el["datetime"]) or timestamp

            if not summary:
                summary = title

            if title or summary:
                threads.append(
                    {
                        "id": thread_id or title or summary,
                        "title": title,
                        "content": summary,
                        "created_at": timestamp,
                    }
                )

        return threads


def _parse_datetime(value: str) -> Optional[datetime]:
    """Parse ISO-like datetime strings safely."""

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class ForumScraper:
    """Scrapes health forums for lay language examples."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        reddit_client: Optional[BaseForumClient] = None,
        patient_info_client: Optional[BaseForumClient] = None,
        mapper: Optional[LayLanguageMapper] = None,
        conditions: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("forum_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._reddit_client = reddit_client
        self._patient_info_client = patient_info_client
        condition_catalogue = conditions or RespiratoryConditions.get_all_conditions()
        self.mapper = mapper or LayLanguageMapper(condition_catalogue)

    def _normalise_posts(self, records: Iterable[Dict[str, Any]], source: str) -> List[ForumPost]:
        posts: List[ForumPost] = []
        for record in records:
            content = (record.get("content") or "").strip()
            title = (record.get("title") or "").strip()
            created_at = record.get("created_at")
            if isinstance(created_at, datetime):
                timestamp = created_at.isoformat()
            else:
                timestamp = str(created_at or datetime.utcnow().isoformat())

            lay_terms, medical_terms = self._extract_terms(content)
            confidence = self._estimate_confidence(lay_terms, content)

            posts.append(
                ForumPost(
                    id=str(record.get("id")),
                    title=title,
                    content=content,
                    timestamp=timestamp,
                    forum_source=source,
                    lay_terms=lay_terms,
                    extracted_symptoms=medical_terms,
                    confidence_score=confidence,
                )
            )
        return posts

    def _extract_terms(self, content: str) -> Tuple[List[str], List[str]]:
        lay_terms: List[str] = []
        medical_terms: List[str] = []
        lowered = content.lower()
        for lay, medical in self.mapper.mappings.items():
            if lay in lowered:
                lay_terms.append(lay)
                medical_terms.append(medical)
        return lay_terms, medical_terms

    @staticmethod
    def _estimate_confidence(lay_terms: Sequence[str], content: str) -> float:
        if not lay_terms:
            return 0.3
        diversity = len(set(lay_terms))
        base = 0.5 + 0.1 * min(diversity, 3)
        length_factor = min(len(content) / 500.0, 0.5)
        return round(min(base + length_factor, 0.95), 2)

    def _get_reddit_client(self) -> BaseForumClient:
        if self._reddit_client is None:
            self._reddit_client = RedditClient()
        return self._reddit_client

    def _get_patient_info_client(self) -> BaseForumClient:
        if self._patient_info_client is None:
            self._patient_info_client = PatientInfoClient()
        return self._patient_info_client

    def scrape_reddit_health(self, max_posts: int = 10) -> List[ForumPost]:
        records = self._get_reddit_client().fetch_posts(max_posts)
        return self._normalise_posts(records, "reddit")

    def scrape_patient_info(self, max_posts: int = 10) -> List[ForumPost]:
        records = self._get_patient_info_client().fetch_posts(max_posts)
        return self._normalise_posts(records, "patient.info")

    def save_posts(self, posts: Sequence[ForumPost], filename: str) -> None:
        filepath = self.cache_dir / filename
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump([asdict(post) for post in posts], handle, indent=2)
        logger.info("Saved %s posts to %s", len(posts), filepath)

    def load_posts(self, filename: str) -> List[ForumPost]:
        filepath = self.cache_dir / filename
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [ForumPost(**post) for post in data]


class ForumDataAugmentation:
    """Augments medical complaints with forum-derived lay language."""

    def __init__(
        self,
        mapper: Optional[LayLanguageMapper] = None,
        forum_posts: Optional[Sequence[ForumPost]] = None,
        scraper: Optional[ForumScraper] = None,
        conditions: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self._conditions = conditions or RespiratoryConditions.get_all_conditions()
        self.mapper = mapper or LayLanguageMapper(self._conditions)
        self.scraper = scraper
        self._forum_posts: List[ForumPost] = list(forum_posts or [])
        self._vocabulary = RespiratoryConditions.get_vocabulary(self._conditions)
        RespiratoryConditions.register_reload_hook(self.reload_conditions)

    def reload_conditions(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Reload configuration-derived vocabulary for long running services."""

        self._conditions = conditions or RespiratoryConditions.get_all_conditions()
        self.mapper.reload_from_conditions(self._conditions)
        self._vocabulary = RespiratoryConditions.get_vocabulary(self._conditions)

    def update_forum_posts(self, posts: Sequence[ForumPost]) -> None:
        self._forum_posts = list(posts)
        if posts:
            self.mapper.update_mappings_from_posts(posts)

    def augment_complaints_with_lay_terms(
        self, complaints: Sequence[str], condition_codes: Sequence[str]
    ) -> List[str]:
        augmented: List[str] = []
        for complaint in complaints:
            augmented_complaint = complaint
            for medical, lay_terms in self.mapper.reverse_mappings.items():
                normalized_medical = medical.replace("_", " ")
                pattern = re.compile(re.escape(normalized_medical), flags=re.IGNORECASE)
                if pattern.search(augmented_complaint):
                    lay_term = lay_terms[0]
                    augmented_complaint = pattern.sub(lay_term, augmented_complaint)
            augmented.append(augmented_complaint)
        return augmented

    def get_forum_complaints_for_pretraining(
        self, max_complaints: int = 100
    ) -> List[str]:
        if not self._forum_posts and self.scraper is not None:
            try:
                self._forum_posts = self.scraper.load_posts("reddit_posts.json")
            except FileNotFoundError:
                logger.debug("No cached forum data available for augmentation")

        if self._forum_posts:
            return [post.content for post in self._forum_posts[:max_complaints]]

        # Fallback to synthetic complaints if no real data has been harvested yet.
        return self._generate_mock_forum_complaints(max_complaints)

    def _generate_mock_forum_complaints(self, max_samples: int) -> List[str]:
        templates = [
            "I've been {lay1} and {lay2} for days now",
            "Can't stop {lay1}, also have {lay2}",
            "Really worried about {lay1} and {lay2}",
            "Help! {lay1} and getting worse",
            "Been {lay1} since yesterday, now {lay2} too",
            "Having {lay1}, plus {lay2}. Getting scared.",
        ]
        lay_vocabulary = self._vocabulary["lay_terms"] or list(self.mapper.mappings.keys())
        symptoms = lay_vocabulary if lay_vocabulary else list(self.mapper.mappings.keys())
        complaints: List[str] = []
        if not symptoms:
            return complaints
        for idx in range(min(max_samples, len(symptoms))):
            lay1 = symptoms[idx % len(symptoms)]
            lay2 = symptoms[(idx + 7) % len(symptoms)]
            template = templates[idx % len(templates)]
            complaints.append(template.format(lay1=lay1, lay2=lay2))
        return complaints


# Convenience factory helpers -------------------------------------------------

def create_forum_scraper(cache_dir: Optional[str] = None) -> ForumScraper:
    conditions = RespiratoryConditions.get_all_conditions()
    return ForumScraper(cache_dir=cache_dir, conditions=conditions)


def create_lay_language_mapper(
    conditions: Optional[Dict[str, Dict]] = None,
) -> LayLanguageMapper:
    return LayLanguageMapper(conditions or RespiratoryConditions.get_all_conditions())


def create_data_augmentation(
    mapper: Optional[LayLanguageMapper] = None,
    forum_posts: Optional[Sequence[ForumPost]] = None,
    scraper: Optional[ForumScraper] = None,
) -> ForumDataAugmentation:
    conditions = RespiratoryConditions.get_all_conditions()
    mapper = mapper or LayLanguageMapper(conditions)
    return ForumDataAugmentation(
        mapper=mapper,
        forum_posts=forum_posts,
        scraper=scraper,
        conditions=conditions,
    )
