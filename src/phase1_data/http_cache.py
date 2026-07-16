"""Throttled, cached, robots.txt-aware HTTP fetcher.

Shared by all scrapers (FBref, Transfermarkt). Design goals:

- Respectful scraping: a hard minimum delay between live requests to the same
  host (default 3.5s, comfortably under Sports Reference's 20 req/min limit),
  robots.txt checked before every live fetch.
- Cache-first: every successful response body is written to disk keyed by URL;
  re-runs never hit the network. This makes the whole pipeline reproducible
  offline once the raw layer is populated.
- Honest failures: non-recoverable HTTP errors raise; nothing is silently
  substituted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "prem-fund-academic-research/0.1 (rate-limited; cached; personal MSc project)"
)


def _cache_key(url: str) -> str:
    """Filename for a URL: readable slug + short hash to guarantee uniqueness."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    parsed = urlparse(url)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", parsed.path).strip("-")[:80]
    ext = ".csv" if parsed.path.endswith(".csv") else ".html"
    return f"{slug}-{digest}{ext}"


class ThrottledCachedSession:
    def __init__(
        self,
        cache_dir: str | Path,
        min_delay: float = 3.5,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 3,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        if auth is not None:
            self.session.auth = auth  # e.g. Companies House basic auth (key, "")
        if headers:
            self.session.headers.update(headers)
        self._last_request_at: dict[str, float] = {}  # per-host
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._index_path = self.cache_dir / "index.jsonl"

    # -- robots.txt ---------------------------------------------------------

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
            try:
                resp = self.session.get(robots_url, timeout=30)
                rp.parse(resp.text.splitlines() if resp.ok else [])
            except requests.RequestException:
                # If robots.txt itself is unreachable, err on the side of caution
                # for anything that isn't a plain content page.
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.session.headers["User-Agent"], url)

    # -- throttling ---------------------------------------------------------

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_request_at.get(host, 0.0)
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request_at[host] = time.monotonic()

    # -- fetching -----------------------------------------------------------

    def cache_path(self, url: str) -> Path:
        return self.cache_dir / _cache_key(url)

    def get(self, url: str, force_refresh: bool = False) -> str:
        """Return the response body for `url`, from cache if available."""
        path = self.cache_path(url)
        if path.exists() and not force_refresh:
            return path.read_text(encoding="utf-8")

        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(url)
            log.info("GET %s (attempt %d)", url, attempt + 1)
            try:
                resp = self.session.get(url, timeout=90)
            except requests.RequestException as exc:  # timeouts, connection resets
                last_error = exc
                wait = 15 * (attempt + 1)
                log.warning("%s for %s; backing off %.0fs", type(exc).__name__, url, wait)
                time.sleep(wait)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                wait = float(resp.headers.get("Retry-After", 30 * (attempt + 1)))
                log.warning("HTTP %d for %s; backing off %.0fs", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
            with self._index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "url": url,
                    "file": path.name,
                    "status": resp.status_code,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }) + "\n")
            return resp.text

        raise RuntimeError(f"All {self.max_retries} attempts failed for {url}") from last_error
