"""GitHub code search source adapter."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

from scholaragent.utils.retry import retry_with_backoff

_http_client = httpx.Client(timeout=30.0)

GITHUB_API_URL = "https://api.github.com/search/code"


def search_github_code(
    query: str,
    language: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search GitHub for code examples.

    Returns list of {content, source_type, source_ref} dicts.
    """
    q = query
    if language:
        q += f" language:{language}"

    headers = {
        "Accept": "application/vnd.github.text-match+json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    params = {
        "q": q,
        "per_page": max_results,
    }

    try:
        response = retry_with_backoff(
            _http_client.get,
            GITHUB_API_URL,
            params=params,
            headers=headers,
            max_retries=2,
            base_delay=1.0,
            retryable_exceptions=(httpx.HTTPError,),
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning("GitHub code search failed: %s", e)
        return []

    results = []
    for item in data.get("items", []):
        fragments = []
        for match in item.get("text_matches", []):
            fragment = match.get("fragment", "")
            if fragment:
                fragments.append(fragment)

        content = "\n---\n".join(fragments) if fragments else item.get("name", "")
        repo = item.get("repository", {})
        repo_name = repo.get("full_name", "")
        file_path = item.get("path", "")

        results.append({
            "content": f"# {repo_name}/{file_path}\n\n{content}",
            "source_type": "code",
            "source_ref": item.get("html_url", ""),
        })

    return results
