# mcp_server_ai.py
import os
import asyncio
from typing import List, Dict, Any, Optional, Literal
from urllib.parse import urljoin
from enum import Enum
import logging

from fastapi import Depends, FastAPI, HTTPException, Query, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig

# PydanticAI imports
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


load_dotenv()
from fastapi import APIRouter

router = APIRouter()

STATIC_TOKEN = os.getenv("STATIC_TOKEN")

# app = FastAPI()

# # Security scheme for Bearer token
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token from Authorization header"""
    if credentials.credentials != STATIC_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials



# Config
TARGET_BASE = "https://bks.divami.com"
API_BASE = "https://bks.divami.com/api"
TOKEN_ID = os.getenv("TOKEN_ID")
TOKEN_SECRET = os.getenv("TOKEN_SECRET")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15.0"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# app = FastAPI(title="Smart BookStack MCP with AI Agent")

# Cache for API responses
fetch_cache = TTLCache(maxsize=256, ttl=300)  # 5 min TTL

# ============================================================================
# Models and Enums
# ============================================================================

class EndpointType(str, Enum):
    SHELVES_LIST = "shelves_list"
    SHELF_DETAIL = "shelf_detail"
    BOOKS_LIST = "books_list"
    BOOK_DETAIL = "book_detail"
    PAGES_LIST = "pages_list"
    PAGE_DETAIL = "page_detail"
    SEARCH = "search"

class QueryIntent(BaseModel):
    """Represents the AI's understanding of the user query"""
    endpoint: EndpointType = Field(description="The best endpoint to use")
    resource_id: Optional[int] = Field(None, description="ID of specific resource if needed")
    search_query: Optional[str] = Field(None, description="Search terms if using search endpoint")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    reasoning: str = Field(description="Why this endpoint was chosen")

class BookStackContext(BaseModel):
    """Context passed to the AI agent"""
    token_id: str
    token_secret: str
    available_endpoints: List[str] = [
        "/api/shelves - List all shelves",
        "/api/shelves/{id} - Get specific shelf with books",
        "/api/books - List all books",
        "/api/books/{id} - Get specific book with pages",
        "/api/pages - List all pages",
        "/api/pages/{id} - Get specific page content",
        "/api/search?query={term}&page={n}&count={n} - Search across content"
    ]

class SmartFetchResult(BaseModel):
    intent: QueryIntent
    data: Dict[str, Any]
    source_url: str
    execution_time: float

# ============================================================================
# BookStack API Client
# ============================================================================

class BookStackClient:
    def __init__(self, token_id: str, token_secret: str):
        self.token_id = token_id
        self.token_secret = token_secret
        self.base_url = API_BASE
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.token_id}:{self.token_secret}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        cache_key = f"{endpoint}:{self.token_id}"
        if cache_key in fetch_cache:
            return fetch_cache[cache_key]
        
        async with httpx.AsyncClient() as client:
            url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
            resp = await client.get(
                url,
                headers=self._get_headers(),
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True
            )
            resp.raise_for_status()
            result = resp.json()
            fetch_cache[cache_key] = result
            return result
    
    async def list_shelves(self, count: int = 100) -> Dict[str, Any]:
        return await self._make_request(f"shelves?count={count}")
    
    async def get_shelf(self, shelf_id: int) -> Dict[str, Any]:
        return await self._make_request(f"shelves/{shelf_id}")
    
    async def list_books(self, count: int = 100, filter_params: Dict = None) -> Dict[str, Any]:
        params = f"count={count}"
        if filter_params:
            for k, v in filter_params.items():
                params += f"&{k}={v}"
        return await self._make_request(f"books?{params}")
    
    async def get_book(self, book_id: str) -> Dict[str, Any]:
        return await self._make_request(f"books/{book_id}")
    
    async def list_pages(self, count: int = 100) -> Dict[str, Any]:
        return await self._make_request(f"pages?count={count}")

    async def get_page(self, page_id: int) -> Dict[str, Any]:
        return await self._make_request(f"pages/{page_id}")
    
    async def search(self, query: str, page: int = 1, count: int = 20) -> Dict[str, Any]:
        return await self._make_request(f"search?query={query}&page={page}&count={count}")

# ============================================================================
# PydanticAI Agent Setup
# ============================================================================
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# provider = GoogleProvider(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

baseurl = os.getenv("LITELLM_BASE_URL", "https://litellm.divami.com")
litellm_key = os.getenv("LITELLM_MASTER_KEY")

# Support custom fallback models via environment variable
# Format: MODEL_FALLBACK_LIST="model1,model2,model3"
custom_models = os.getenv("MODEL_FALLBACK_LIST", "")


class ModelConfig(BaseModel):
    """Configuration for AI model providers."""
    model_name: str
    base_url: str
    api_key: Optional[str] = None  # Optional for local proxies

    @validator("base_url")
    def base_url_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("base_url must not be empty")
        return v

def _create_model(config: ModelConfig) -> OpenAIChatModel:
    """Create an OpenAIChatModel from configuration."""
    provider_kwargs = {"base_url": config.base_url}
    if config.api_key:
        provider_kwargs["api_key"] = config.api_key
    provider = OpenAIProvider(**provider_kwargs)
    return OpenAIChatModel(model_name=config.model_name, provider=provider)


if custom_models:
    # Use custom model list from environment
    model_names = [m.strip() for m in custom_models.split(",") if m.strip()]
    MODEL_CONFIGS = [
        ModelConfig(
            model_name=model_name,
            base_url=baseurl,
            api_key=litellm_key,
        )
        for model_name in model_names
    ]
    log.info(f"Using custom model fallback list: {model_names}")
else:
    # Default fallback chain
    MODEL_CONFIGS = [
        ModelConfig(
            model_name="openai/gpt-4o-mini",
            base_url=baseurl,
            api_key=litellm_key,
        ),
        ModelConfig(
            model_name="gemini/gemini-2.5-flash",
            base_url=baseurl,
            api_key=litellm_key,
        ),
        ModelConfig(
            model_name="gemini/gemini-2.5-pro",
            base_url=baseurl,
            api_key=litellm_key,
        ),
    ]

# Initialize the model (using OpenAI, but can use Anthropic/others)
# model = GoogleModel('gemini-1.5-flash', provider=provider)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# model = GoogleModel('gemini-1.5-flash', api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL_INSTANCES = [_create_model(cfg) for cfg in MODEL_CONFIGS]

# Create the agent
bookstack_agent = Agent(
    model=MODEL_INSTANCES[0],
    output_type=QueryIntent,
    system_prompt="""You are an intelligent assistant for the BookStack documentation system.
    
Your job is to analyze user queries and determine the best API endpoint to use.

BookStack hierarchy:
- Shelves: Top-level containers that hold multiple books
- Books: Collections of pages on a topic
- Pages: Individual documentation pages with content

Available endpoints:
1. /api/shelves - List all shelves (use for: "show me all shelves", "what shelves exist")
2. /api/shelves/{id} - Get shelf details with books (use for: "show me shelf X", "what's in shelf Y")
3. /api/books - List all books (use for: "list all books", "what books are there")
4. /api/books/{id} - Get book details with pages (use for: "show me book X", "what's in book Y")
5. /api/pages - List all pages (use for: "list all pages")
6. /api/pages/{id} - Get specific page content (use for: "show me page X", "content of page Y")
7. /api/search - Search across all content (use for: vague queries, keyword searches, "find X")

Decision rules:
- If query mentions specific ID (shelf/book/page), use detail endpoint
- If query is vague or asks to "find" something, use search
- If query asks for lists/overview, use list endpoints
- Extract IDs from queries like "shelf 5" or "book id 23"
- For navigation queries ("what's inside X"), use detail endpoints to get children

Provide clear reasoning for your choice."""
)

@bookstack_agent.tool
async def extract_id_from_query(ctx: RunContext[BookStackContext], query: str) -> Optional[int]:
    """Extract numeric ID from query text"""
    import re
    match = re.search(r'\b(?:id|#|number)?\s*(\d+)\b', query.lower())
    return int(match.group(1)) if match else None

# ============================================================================
# Agent Executor
# ============================================================================

async def execute_smart_query(
    user_query: str,
    client: BookStackClient,
    context: BookStackContext
) -> SmartFetchResult:
    """Use AI agent to determine intent and execute appropriate API call"""
    import time
    start = time.time()
    
    # Get intent from AI agent
    result = await bookstack_agent.run(user_query, deps=context)
    intent: QueryIntent = result.data
    
    # Execute based on intent
    data = None
    url = ""
    
    try:
        if intent.endpoint == EndpointType.SHELVES_LIST:
            data = await client.list_shelves()
            url = f"{API_BASE}/shelves"
            
        elif intent.endpoint == EndpointType.SHELF_DETAIL:
            if not intent.resource_id:
                raise ValueError("Shelf ID required but not found in query")
            data = await client.get_shelf(intent.resource_id)
            url = f"{API_BASE}/shelves/{intent.resource_id}"
            
        elif intent.endpoint == EndpointType.BOOKS_LIST:
            data = await client.list_books(filter_params=intent.filters)
            url = f"{API_BASE}/books"
            
        elif intent.endpoint == EndpointType.BOOK_DETAIL:
            if not intent.resource_id:
                raise ValueError("Book ID required but not found in query")
            data = await client.get_book(intent.resource_id)
            url = f"{API_BASE}/books/{intent.resource_id}"
            
        elif intent.endpoint == EndpointType.PAGES_LIST:
            data = await client.list_pages()
            url = f"{API_BASE}/pages"
            
        elif intent.endpoint == EndpointType.PAGE_DETAIL:
            if intent.resource_id is not None:
                intent.resource_id = str(intent.resource_id)

            if not intent.resource_id:
                raise ValueError("Page ID required but not found in query")
            data = await client.get_page(intent.resource_id)
            url = f"{API_BASE}/pages/{intent.resource_id}"
            
        elif intent.endpoint == EndpointType.SEARCH:
            if not intent.search_query:
                intent.search_query = user_query  # Fallback to original query
            data = await client.search(intent.search_query)
            url = f"{API_BASE}/search?query={intent.search_query}"
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"BookStack API error: {str(e)}")
    
    execution_time = time.time() - start
    
    return SmartFetchResult(
        intent=intent,
        data=data or {},
        source_url=url,
        execution_time=execution_time
    )

# ============================================================================
# FastAPI Endpoints
# ============================================================================

# @app.get("/smart-fetch", response_model=SmartFetchResult)
# async def smart_fetch(
#     query: str = Query(..., description="Natural language query (e.g., 'show me shelf 5', 'find pages about API')"),
#     # query: Optional[str] = Query(None),
#     token_id: Optional[str] = Header(None),
#     token_secret: Optional[str] = Header(None),
#     use_env: bool = Query(True)
# ):
#     """
#     Smart endpoint that uses AI to understand your query and fetch the right data.
    
#     Examples:
#     - "Show me all shelves"
#     - "What's in shelf 5?"
#     - "Find pages about authentication"
#     - "Get book 23"
#     - "Search for API documentation"
#     """
#     # if not model:
#     #     raise HTTPException(
#     #         status_code=503, 
#     #         detail="AI model not configured. Set OPENAI_API_KEY environment variable."
#     #     )
    
#     tid = token_id or (TOKEN_ID if use_env else None)
#     tse = token_secret or (TOKEN_SECRET if use_env else None)
    
#     if not tid or not tse:
#         raise HTTPException(
#             status_code=400,
#             detail="Authentication required. Provide token_id and token_secret headers or set environment variables."
#         )
    
#     client = BookStackClient(tid, tse)
#     context = BookStackContext(token_id=tid, token_secret=tse)
    
#     return await execute_smart_query(query, client, context)


@router.get("/fetch/{endpoint_type}/{resource_id:int}")
async def direct_fetch(
    resource_id: int,
    endpoint_type: EndpointType,
    token_id: Optional[str] = Header(None),
    token_secret: Optional[str] = Header(None),
    use_env: bool = Query(True)
):

    """
    Direct fetch endpoint for when you know exactly what you want"""
    tid = token_id or (TOKEN_ID if use_env else None)
    tse = token_secret or (TOKEN_SECRET if use_env else None)
    
    
    if not tid or not tse:
        raise HTTPException(status_code=400, detail="Authentication required")
    
    client = BookStackClient(tid, tse)
    
    if endpoint_type == EndpointType.SHELVES_LIST:
        data = await client.list_shelves()
    elif endpoint_type == EndpointType.SHELF_DETAIL:
        data = await client.get_shelf(resource_id)
    elif endpoint_type == EndpointType.BOOKS_LIST:
        data = await client.list_books()
    elif endpoint_type == EndpointType.BOOK_DETAIL:
        data = await client.get_book(resource_id)
    elif endpoint_type == EndpointType.PAGES_LIST:
        data = await client.list_pages()
    elif endpoint_type == EndpointType.PAGE_DETAIL:
        data = await client.get_page(resource_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid endpoint type")
    
    return {"data": data}


@router.get("/search")
async def search_endpoint(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1),
    count: int = Query(20, ge=1, le=100),
    token_id: Optional[str] = Header(None),
    token_secret: Optional[str] = Header(None),
    use_env: bool = Query(True)
):
    """Direct search endpoint"""
    tid = token_id or (TOKEN_ID if use_env else None)
    tse = token_secret or (TOKEN_SECRET if use_env else None)
    
    if not tid or not tse:
        raise HTTPException(status_code=400, detail="Authentication required")
    
    client = BookStackClient(tid, tse)
    return await client.search(q, page, count)


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "ai_enabled": MODEL_INSTANCES[0] is not None,
        "cache_size": len(fetch_cache)
    }
@router.get("/mcp")
async def mcp_root(token: str = Depends(verify_token)):
    """Protected MCP root endpoint"""
    return {
        "message": "✅ MCP Ready. Auth successful.",
        "status": "authenticated"
    }


# ============================================================================
# MCP Integration
# ============================================================================

# mcp = FastApiMCP(app, name="Smart BookStack MCP",  auth_config=AuthConfig(
#         dependencies=[Depends(verify_token)]
#     ))
# mcp.mount_http()


# if __name__ == "__main__":
#     import uvicorn
#     # print("gemini_key configured:", bool(GEMINI_API_KEY))
#     print("token_id configured:", bool(TOKEN_ID))
#     print("token_secret configured:", bool(TOKEN_SECRET))
#     uvicorn.run(app, host="0.0.0.0", port=8010)