import os
import urllib.parse
import uuid
from typing import List, Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

load_dotenv()

MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "4000"))
app = FastAPI(title="Lightweight Search + Summarize Proxy")

# Common browser headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    date: Optional[str] = None
    last_updated: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    id: str


async def search_duckduckgo(query: str, limit: int) -> List[SearchResult]:
    """
    Search DuckDuckGo using their HTML interface.
    More reliable than third-party libraries.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        resp = await client.get(url, headers=HEADERS)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    # DuckDuckGo HTML results are in divs with class "result"
    for result_div in soup.select(".result"):
        # Get title and URL
        title_elem = result_div.select_one(".result__a")
        snippet_elem = result_div.select_one(".result__snippet")

        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)

        # DuckDuckGo wraps URLs in a redirect, extract actual URL
        href = str(title_elem.get("href") or "")
        # Parse the uddg parameter which contains the actual URL
        if href and "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            actual_url = parsed.get("uddg", [""])[0]
        else:
            actual_url = href

        actual_url = str(actual_url or "")

        if not actual_url or not actual_url.startswith("http"):
            continue

        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        results.append(SearchResult(title=title, url=actual_url, snippet=snippet))

        if len(results) >= limit:
            break

    print(f"DuckDuckGo search for '{query}' returned {len(results)} results")
    return results


async def search_brave(query: str, limit: int) -> List[SearchResult]:
    """
    Fallback: Search using Brave Search HTML interface.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://search.brave.com/search?q={encoded_query}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        resp = await client.get(url, headers=HEADERS)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    # Brave search results - try multiple selectors
    for result_div in soup.select("[data-type='web']"):
        title_elem = result_div.select_one("a")
        snippet_elem = result_div.select_one(".snippet-description, .snippet-content")

        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        href = str(title_elem.get("href") or "")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        if not href.startswith("http"):
            continue

        results.append(SearchResult(title=title, url=href, snippet=snippet))

        if len(results) >= limit:
            break

    print(f"Brave search for '{query}' returned {len(results)} results")
    return results


async def search_google(query: str, limit: int) -> List[SearchResult]:
    """
    Fallback: Search using Google (may be rate limited).
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}&num={limit}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        resp = await client.get(url, headers=HEADERS)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    # Google search results
    for g in soup.select(".g"):
        title_elem = g.select_one("h3")
        link_elem = g.select_one("a")
        snippet_elem = g.select_one(".VwiC3b, .st")

        if not title_elem or not link_elem:
            continue

        title = title_elem.get_text(strip=True)
        href = str(link_elem.get("href") or "")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        if not href.startswith("http"):
            continue

        results.append(SearchResult(title=title, url=href, snippet=snippet))

        if len(results) >= limit:
            break

    print(f"Google search for '{query}' returned {len(results)} results")
    return results


async def search_web(
    query: str,
    limit: int,
    language: Optional[str] = None,
    country: Optional[str] = None,
) -> List[SearchResult]:
    """
    Try multiple search backends until one works.
    """
    results = []

    # Try DuckDuckGo first (most reliable)
    try:
        results = await search_duckduckgo(query, limit)
        if results:
            return results
    except Exception as e:
        print(f"DuckDuckGo failed: {e}")

    # Fallback to Brave
    try:
        results = await search_brave(query, limit)
        if results:
            return results
    except Exception as e:
        print(f"Brave failed: {e}")

    # Last resort: Google
    try:
        results = await search_google(query, limit)
        if results:
            return results
    except Exception as e:
        print(f"Google failed: {e}")

    return results


async def fetch_and_extract(url: str) -> Optional[str]:
    """Fetch a webpage and extract its main content."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=12.0) as client:
            resp = await client.get(url, headers=HEADERS)
            resp.raise_for_status()
            extracted = trafilatura.extract(
                resp.text,
                url=url,
                favor_precision=True,
                include_comments=False,
                include_images=False,
            )
            if not extracted:
                return None
            return extracted[:MAX_PAGE_CHARS]
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return None


@app.get("/search", response_model=SearchResponse)
async def search_endpoint(
    q: str = Query(..., min_length=2),
    k: int = Query(2, ge=1, le=10),
    language: Optional[str] = Query(
        None, min_length=2, max_length=10, description="e.g. en or en-US"
    ),
    country: Optional[str] = Query(
        None, min_length=2, max_length=10, description="e.g. US, IN"
    ),
    max_results: Optional[int] = Query(None, ge=1, le=10),
):
    """
    Search the web and return results in a Perplexity-like shape.
    """
    try:
        limit = max_results or k
        results = await search_web(q, limit, language=language, country=country)

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No search results found for query: {q}. Try a different query.",
            )

        # Enrich snippets by fetching page content
        enriched: List[SearchResult] = []
        for item in results:
            text = await fetch_and_extract(item.url)
            if text:
                item.snippet = text
            enriched.append(item)

        return SearchResponse(results=enriched, id=str(uuid.uuid4()))

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Search endpoint error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

