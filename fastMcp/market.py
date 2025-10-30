"""
Market Research MCP Server using FastMCP, DDGS, and Pydantic AI
"""
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional
from fastmcp import FastMCP
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
import os
import textwrap
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig
from fastapi import FastAPI, Depends


# Load environment variablesx
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"Gemini API Key configured: {bool(gemini_key)}")
print(f"OpenAI API Key configured: {bool(openai_key)}")


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



##auth
token_auth_scheme = HTTPBearer()

##fastapi app
app = FastAPI()

@app.get("/private")
async def private(token=Depends(token_auth_scheme)):
    return token.credentials
# -------------------------
# FastMCP Setup
# -------------------------
# mcp = FastMCP("MarketResearchAgent")
mcp = FastApiMCP(
    app,
    name="MarketResearchAgent",
    # auth_config=AuthConfig(
    #     dependencies=[Depends(token_auth_scheme)],
    # ),
)

mcp.mount_http()
# app.mount('/mcp', mcp.sse_app())


# -------------------------
# Helper Functions
# -------------------------
def get_configured_model():
    """Return the first available AI model based on API keys"""
    if gemini_key:
        return GeminiModel("gemini-2.0-flash-exp")
    elif openai_key:
        return OpenAIModel("gpt-4o-mini")
    else:
        raise ValueError(
            "No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in your .env file"
        )


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
# @mcp.tool()
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
        model = get_configured_model()
        
        # 4. Create Pydantic AI Agent with proper syntax
        agent = Agent(
            model=model,
            system_prompt=(
                "You are an expert market research analyst. "
                "Analyze search results and provide structured insights. "
                "Be factual, specific, and conservative with confidence estimates. "
                "Focus on actionable intelligence."
            )
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
# @mcp.tool()
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


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    # Verify API keys before starting
    if not gemini_key and not openai_key:
        print("\n⚠️  WARNING: No API keys configured!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY in your .env file\n")
    
    # Run FastMCP server
    print("\n🚀 Starting Market Research MCP Server on port 8001...")
    print("Available tools: market_research, quick_search\n")
    # mcp.run(transport="streamable-http", port=8001)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)