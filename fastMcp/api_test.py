# mcp_server.py
import os
import asyncio
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException, Query, Header, BackgroundTasks
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig

load_dotenv()

# Config
TARGET_BASE = "https://bks.divami.com"
API_BASE = "https://bks.divami.com/api"  # BookStack API endpoint
TOKEN_ID = os.getenv("TOKEN_ID")
TOKEN_SECRET = os.getenv("TOKEN_SECRET")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "6"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15.0"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "50"))

app = FastAPI(title="MCP Fetch + Explorer for bks.divami.com")

# Simple in-memory cache for fetched responses (avoid hammering)
fetch_cache = TTLCache(maxsize=256, ttl=60)  # 60s TTL; tune as needed

# Semaphore to limit concurrent outbound requests
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# Models
class FetchResult(BaseModel):
    url: str
    status_code: int
    content_type: Optional[str]
    snippet: Optional[str] = None

class ExploreResult(BaseModel):
    discovered: List[str]
    fetched: List[FetchResult]

# helpers
def _build_auth_headers(token_id: Optional[str], token_secret: Optional[str]) -> Dict[str, str]:
    """
    Build headers expected by BookStack API.
    BookStack expects: Authorization: Token {token_id}:{token_secret}
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    if token_id and token_secret:
        headers["Authorization"] = f"Token {token_id}:{token_secret}"
    return headers

def _same_host(url: str, base_host: str) -> bool:
    try:
        return urlparse(url).netloc == base_host
    except Exception:
        return False

async def _fetch_url(client: httpx.AsyncClient, url: str, headers: Dict[str, str]) -> httpx.Response:
    # Use caching key
    cache_key = f"GET:{url}:{headers.get('X-Token-Id','')}"
    if cache_key in fetch_cache:
        return fetch_cache[cache_key]

    async with semaphore:
        resp = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=True)
    # Cache simple tuple (status, headers['content-type'], text)
    fetch_cache[cache_key] = resp
    return resp

def _extract_links_from_html(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all(["a", "link", "script"]):
        href = tag.get("href") or tag.get("src")
        if not href:
            continue
        full = urljoin(base_url, href)
        links.add(full)
    return links

def _extract_links_from_json(data: Any, base_url: str) -> Set[str]:
    # naive JSON walk: if a value looks like a URL, collect it
    links = set()
    def walk(x):
        if isinstance(x, dict):
            for k,v in x.items():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
        elif isinstance(x, str):
            if x.startswith("http://") or x.startswith("https://") or x.startswith("/"):
                links.add(urljoin(base_url, x))
    walk(data)
    return links

# Public endpoints

@app.get("/fetch", response_model=Dict[str, Any])
async def fetch(
    path: str = Query("/api/books", description="Path to fetch (e.g. /shelves or /api/shelves)"),
    token_id: Optional[str] = Header(None, convert_underscores=False),
    token_secret: Optional[str] = Header(None, convert_underscores=False),
    use_env: bool = Query(True, description="Use env vars for tokens"),
    force_html: bool = Query(False, description="Force HTML mode, skip API"),
):
    """
    Fetch content from BookStack.
    - Tries API endpoint first (JSON)
    - Falls back to HTML page parsing if JSON fails or not authorized
    """
    tid = token_id or (TOKEN_ID if use_env else None)
    tse = token_secret or (TOKEN_SECRET if use_env else None)

    headers = _build_auth_headers(tid, tse)

    # Decide which endpoint to hit
    api_url = urljoin(API_BASE + "/", path.lstrip("/").replace("api/", ""))
    html_url = urljoin(TARGET_BASE + "/", path.lstrip("/").replace("api/", ""))

    async with httpx.AsyncClient() as client:
        # First try API if not forcing HTML
        if not force_html:
            try:
                api_resp = await client.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
                if api_resp.status_code == 200 and "application/json" in api_resp.headers.get("content-type", ""):
                    return {
                        "mode": "api",
                        "url": api_url,
                        "status": api_resp.status_code,
                        "data": api_resp.json(),
                    }
            except Exception as e:
                # fallback to HTML
                print(f"[API fetch failed, falling back to HTML] {e}")

        # Fallback: HTML parsing
        try:
            html_resp = await client.get(html_url, headers=headers, timeout=REQUEST_TIMEOUT)
            html_resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"HTML fetch failed: {e}")

        soup = BeautifulSoup(html_resp.text, "html.parser")

        # Parse shelves (BookStack standard structure)
        shelves = []
        for shelf_card in soup.select(".book-grid-item"):  # BookStack uses .book-grid-item for shelves
            title_el = shelf_card.select_one(".book-grid-title")
            desc_el = shelf_card.select_one(".book-grid-description")
            link_el = shelf_card.find("a", href=True)
            shelves.append({
                "title": title_el.get_text(strip=True) if title_el else None,
                "description": desc_el.get_text(strip=True) if desc_el else None,
                "url": urljoin(TARGET_BASE, link_el["href"]) if link_el else None
            })

        return {
            "mode": "html",
            "url": html_url,
            "status": html_resp.status_code,
            "shelves_found": len(shelves),
            "data": shelves or [{"raw_html_snippet": html_resp.text[:1500]}],
        }


@app.get("/explore", response_model=ExploreResult)
async def explore(
    start_path: str = Query("/", description="Starting path to explore (e.g. /api/shelves/{id})"),
    max_depth: int = Query(CRAWL_MAX_DEPTH, description="Max crawl depth"),
    max_pages: int = Query(CRAWL_MAX_PAGES, description="Max pages to fetch"),
    token_id: Optional[str] = Header(None, convert_underscores=False),
    token_secret: Optional[str] = Header(None, convert_underscores=False),
    use_env: bool = Query(True)
):
    """
    Explore the site, following same-host links discovered in HTML pages or JSON responses.
    Returns list of discovered endpoints and a few fetched snippets.
    """
    tid = token_id or (TOKEN_ID if use_env else None)
    tse = token_secret or (TOKEN_SECRET if use_env else None)

    if not tid or not tse:
        raise HTTPException(status_code=400, detail="Token id/secret missing. Provide headers or set env vars BKS_TOKEN_ID/BKS_TOKEN_SECRET.")

    headers = _build_auth_headers(tid, tse)
    discovered: Set[str] = set()
    fetched_results: List[FetchResult] = []

    base_host = urlparse(TARGET_BASE).netloc

    q = asyncio.Queue()
    start_url = urljoin(TARGET_BASE, start_path.lstrip("/"))
    await q.put((start_url, 0))
    visited: Set[str] = set()
    pages_fetched = 0

    async with httpx.AsyncClient() as client:
        while not q.empty() and pages_fetched < max_pages:
            url, depth = await q.get()
            if url in visited:
                continue
            if depth > max_depth:
                continue
            visited.add(url)

            try:
                resp = await _fetch_url(client, url, headers)
            except Exception as e:
                # record failed status
                fetched_results.append(FetchResult(url=url, status_code=0, content_type=None, snippet=f"fetch error: {str(e)}"))
                continue

            pages_fetched += 1
            ctype = resp.headers.get("content-type", "")
            snippet = (resp.text[:800] if resp.text else None)
            fetched_results.append(FetchResult(url=url, status_code=resp.status_code, content_type=ctype, snippet=snippet))
            discovered.add(url)

            # If JSON-like, try to extract URLs
            try:
                if "application/json" in ctype or resp.text.strip().startswith("{") or resp.text.strip().startswith("["):
                    data = resp.json()
                    urls = _extract_links_from_json(data, url)
                else:
                    urls = _extract_links_from_html(resp.text, url)
            except Exception:
                urls = set()

            for u in urls:
                # only follow same-host links
                if _same_host(u, base_host):
                    # normalize: remove fragment
                    parsed = urlparse(u)
                    norm = parsed._replace(fragment="").geturl()
                    if norm not in visited:
                        await q.put((norm, depth + 1))

    return ExploreResult(discovered=sorted(discovered), fetched=fetched_results)

# Optional: list discovered endpoints stored in cache (very basic)
@app.get("/endpoints")
def endpoints():
    keys = list(fetch_cache.keys())
    return {"cache_keys_count": len(keys), "sample_keys": keys[:20]}

# token_auth_scheme = HTTPBearer()

mcp = FastApiMCP(
    app,
    name="Protected MCP",
    # auth_config=AuthConfig(
    #     dependencies=[Depends(token_auth_scheme)],
    # ),
)

# Mount the MCP server
mcp.mount_http()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)