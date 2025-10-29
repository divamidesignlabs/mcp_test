"""Market Research MCP Server using FastMCP, DDGS, and LiteLLM.

Refactored to remove undefined model abstractions and rely directly on LiteLLM's
`completion` interface pointed at a configurable proxy/base URL.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from fastmcp import FastMCP
from ddgs import DDGS
import os
import textwrap
import json
import re
import asyncio
from dotenv import load_dotenv
from litellm import completion
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from fastapi_mcp import FastApiMCP, AuthConfig
# -------------------------
# Environment / Configuration
# -------------------------
load_dotenv()

LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")  
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.divami.com")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini/gemini-1.5-flash") 


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

def _create_model(config) -> OpenAIChatModel:
    """Create an OpenAIChatModel from configuration."""
    provider_kwargs = {"base_url": config.base_url}
    if config.api_key:
        provider_kwargs["api_key"] = config.api_key
    provider = OpenAIProvider(**provider_kwargs)
    return OpenAIChatModel(model_name=config.model_name, provider=provider)

baseurl=os.getenv("LITELLM_BASE_URL", "https://litellm.divami.com")
litellm_key=os.getenv("LITELLM_MASTER_KEY")

gemini_config = ModelConfig(
    model_name="openai/gpt-4o-mini",
    base_url=baseurl,
    api_key=litellm_key,
)

gemini_model = _create_model(gemini_config)

# -------------------------
# Pydantic Models
# -------------------------
class SearchHit(BaseModel):
    """Individual search result citation"""
    title: Optional[str] = None
    href: Optional[str] = None
    body: Optional[str] = None


class MarketResearchResult(BaseModel):
    """Structured market research analysis"""
    query: str = Field(..., description="The original search query")
    summary: str = Field(..., description="2-4 sentence summary of findings")
    top_trends: List[str] = Field(default_factory=list, description="3-5 key market trends")
    competitors: List[str] = Field(default_factory=list, description="3-5 main competitors")
    opportunities: List[str] = Field(default_factory=list, description="3 market opportunities")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence level 0-1")
    citations: List[SearchHit] = Field(default_factory=list, description="Source citations")


# -------------------------
# FastMCP Setup
# -------------------------
# mcp = FastMCP("Market Research MCP")
app=FastAPI()

# -------------------------
# Helper Functions
# -------------------------


def format_search_results(raw_hits: list) -> List[dict]:
    """Convert DDGS raw hits into structured SearchHit format"""
    formatted = []
    for hit in raw_hits:
        formatted.append({
            "title": hit.get("title"),
            "href": hit.get("href") or hit.get("url"),
            "body": hit.get("body") or hit.get("snippet"),
        })
    return formatted


def build_context_string(hits: List[dict]) -> str:
    """Create a formatted context string from search results"""
    snippets = []
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title") or "No title"
        url = hit.get("href") or "No URL"
        body = hit.get("body") or "No snippet"
        snippets.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {body}\n")
    return "\n".join(snippets)


# -------------------------
# MCP Tool
# -------------------------
@app.get("/market_research")
async def market_research(query: str, max_results: int = 5) -> dict:
    """
    Perform comprehensive market research using DuckDuckGo search and AI analysis.
    
    Args:
        query: The market research query (e.g., "AI coding assistants market")
        max_results: Number of search results to analyze (default: 5)
    
    Returns:
        dict: Structured market research with trends, competitors, and opportunities
    """
    try:
        # 1. Perform web search
        print(f"Searching for: {query}")
        with DDGS(timeout=10) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
        
        if not raw_results:
            return {
                "error": "No search results found",
                "query": query,
                "suggestions": "Try a different search query or check your internet connection"
            }
        
        # 2. Format search results
        hits = format_search_results(raw_results)
        context = build_context_string(hits)
        
        # 3. Get configured AI model
        # model = get_configured_model()
        
        # 4. Create Pydantic AI Agent with proper syntax
        # agent = Agent(
        #     model=model,
        #     system_prompt=(
        #         "You are an expert market research analyst. "
        #         "Analyze search results and provide structured insights. "
        #         "Be factual, specific, and conservative with confidence estimates. "
        #         "Focus on actionable intelligence."
        #     )
        # )

        agent = Agent(
            model=gemini_model,
            system_prompt=(
                 "You are an expert market research analyst. "
                "Analyze search results and provide structured insights. "
                "Be factual, specific, and conservative with confidence estimates. "
                "Focus on actionable intelligence."
            ),
            name="daksh-bot",
        )

        
        # 5. Build analysis prompt
        prompt = textwrap.dedent(f"""
            Conduct market research analysis for: "{query}"
            
            Search Results ({len(hits)} sources):
            {context}
            
            Provide a structured analysis with:
            1. Summary: 2-4 sentences capturing key insights
            2. Top Trends: 3-5 significant market trends (be specific)
            3. Competitors: 3-5 main players or competitor names
            4. Opportunities: 3 potential market opportunities
            5. Confidence: Your confidence level (0.0-1.0) in this analysis
            6. Citations: Reference the sources used
            
            Return ONLY a valid JSON object matching this exact structure:
            {{
                "query": "{query}",
                "summary": "...",
                "top_trends": ["trend1", "trend2", ...],
                "competitors": ["competitor1", "competitor2", ...],
                "opportunities": ["opportunity1", "opportunity2", ...],
                "confidence": 0.0-1.0,
                "citations": [
                    {{"title": "...", "href": "...", "body": "..."}},
                    ...
                ]
            }}
        """)
        
        # 6. Run agent and get structured output
        result = await agent.run(prompt, message_history=[])
        
        # 7. Parse and validate response
        # Try to parse as MarketResearchResult
        try:
            # Extract the data field from the agent result
            data = result.data if hasattr(result, 'data') else result
            
            # If it's a string, try to parse it as JSON
            if isinstance(data, str):
                import json
                data = json.loads(data)
            
            # Validate with Pydantic model
            validated = MarketResearchResult.model_validate(data)
            return validated.model_dump()
            
        except Exception as parse_error:
            # Return raw response if parsing fails
            print(f"Parsing error: {parse_error}")
            return {
                "query": query,
                "summary": str(result.data if hasattr(result, 'data') else result),
                "top_trends": [],
                "competitors": [],
                "opportunities": [],
                "confidence": 0.3,
                "citations": hits,
                "note": "Analysis returned in raw format due to parsing issue"
            }
    
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "type": type(e).__name__
        }


# -------------------------
# Optional: Additional Tools
# -------------------------
@app.get("/quick_search")
def quick_search(query: str, max_results: int = 3) -> dict:
    """
    Quick DuckDuckGo search without AI analysis.
    
    Args:
        query: Search query
        max_results: Number of results (default: 3)
    
    Returns:
        dict: Raw search results
    """
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        return {
            "query": query,
            "results": format_search_results(results),
            "count": len(results)
        }
    except Exception as e:
        return {
            "error": str(e),
            "query": query
        }


STATIC_TOKEN = os.getenv("STATIC_TOKEN")

app = FastAPI()

# Security scheme for Bearer token
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


@app.get("/mcp")
async def mcp_root(token: str = Depends(verify_token)):
    """Protected MCP root endpoint"""
    return {
        "message": "✅ MCP Ready. Auth successful.",
        "status": "authenticated"
    }


# Mount MCP with token auth
mcp = FastApiMCP(
    app,
    name="Market Research MCP",
    auth_config=AuthConfig(
        dependencies=[Depends(verify_token)]
    )
)

mcp.mount_http()

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
    
    # Run FastMCP server
    print("\n🚀 Starting Market Research MCP Server on port 8011...")
    print("Available tools: market_research, quick_search\n")
    mcp.run(transport="streamable-http", port=8011)