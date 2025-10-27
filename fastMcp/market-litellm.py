"""Market Research MCP Server using FastMCP, DDGS, and LiteLLM.

Production hardening updates:
 - Separate auth port to avoid conflict
 - Optional auth skip via SKIP_STARTUP_AUTH
 - Session token secret (itsdangerous ready)
 - Reuse model providers (performance)
 - Robust JSON extraction fallback
 - Tool-level auth gating
 - Logging + removal of unused imports
"""
import os
import time
import json
import textwrap
import logging
import secrets
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from fastmcp import FastMCP
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import webbrowser
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="[%(levelname)s] %(message)s")
log = logging.getLogger("market-mcp")
# -------------------------
# Environment / Configuration
# -------------------------
load_dotenv()

# Simple Authentication Configuration
AUTH_USERNAME = os.getenv("MCP_AUTH_USERNAME","testmcp")
AUTH_PASSWORD = os.getenv("MCP_AUTH_PASSWORD","test123")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
AUTH_PORT = int(os.getenv("AUTH_PORT", str(MCP_PORT + 1)))
PUBLIC_AUTH_BASE_URL = os.getenv("PUBLIC_AUTH_BASE_URL")  # e.g. https://mydomain.com
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development or production
SKIP_STARTUP_AUTH = os.getenv("SKIP_STARTUP_AUTH", "false").lower() == "true"
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)

if not AUTH_USERNAME or not AUTH_PASSWORD:
    log.warning("Authentication credentials not set (MCP_AUTH_USERNAME/MCP_AUTH_PASSWORD). Set these for production use.")
serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="mcp-auth")

# LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")  
# LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
# MODEL_NAME = os.getenv("MODEL_NAME", "gemini/gemini-1.5-flash") 

# -------------------------
# Model Configurations with Fallback
# -------------------------
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

def get_model_with_fallback() -> OpenAIChatModel:
    """
    Try to create a model from the configured options with fallback.
    Returns the first successfully created model.
    """
    last_error = None
    
    for i, config in enumerate(MODEL_CONFIGS):
        try:
            log.info(f"Attempting to use model: {config.model_name}")
            model = _create_model(config)
            log.info(f"Successfully initialized model: {config.model_name}")
            return model
        except Exception as e:
            last_error = e
            log.error(f"Failed to initialize {config.model_name}: {str(e)}")
            continue
    
    # If all models fail, raise the last error
    raise RuntimeError(
        f"All model configurations failed. Last error: {last_error}"
    )

# Initialize the primary model with fallback
try:
    primary_model = get_model_with_fallback()
except Exception as e:
    log.critical(f"Could not initialize any model: {e}")
    primary_model = None

# Pre-initialize providers list for reuse
MODEL_PROVIDERS: List[OpenAIChatModel] = []
for cfg in MODEL_CONFIGS:
    try:
        MODEL_PROVIDERS.append(_create_model(cfg))
    except Exception as e:
        log.warning(f"Skipping provider init for {cfg.model_name}: {e}")

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
mcp = FastMCP("MarketResearchAgent")

# -------------------------
# Authentication State
# -------------------------
authenticated_user = None
authentication_complete = False

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


async def run_agent_with_fallback(prompt: str, system_prompt: str) -> any:
    """
    Try to run the agent with fallback models if the primary fails.
    
    Args:
        prompt: The analysis prompt
        system_prompt: The system prompt for the agent
        
    Returns:
        Agent result or raises exception if all models fail
    """
    last_error = None
    
    providers_chain = MODEL_PROVIDERS if MODEL_PROVIDERS else [
        _create_model(cfg) for cfg in MODEL_CONFIGS
    ]
    for i, model in enumerate(providers_chain):
        try:
            log.info(f"Trying model {i+1}/{len(providers_chain)}: {model.model_name}")
            agent = Agent(
                model=model,
                system_prompt=system_prompt,
                name="market-research-agent",
            )
            result = await agent.run(prompt, message_history=[])
            log.info(f"Got response from: {model.model_name}")
            return result
        except Exception as e:
            last_error = e
            log.error(f"Model {getattr(model,'model_name','unknown')} failed: {e}")
            if i < len(providers_chain) - 1:
                log.info("Trying next fallback model...")
            continue
    
    # If all models fail, raise the last error
    raise RuntimeError(
        f"All {len(MODEL_CONFIGS)} model configurations failed. Last error: {last_error}"
    )


# -------------------------
# MCP Tool
# -------------------------
def _require_auth() -> Optional[dict]:
    if not authentication_complete:
        return {"error": "Not authenticated", "detail": "Login required before using tools"}
    return None

def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    return None

@mcp.tool()
async def market_research(query: str, max_results: int = 5) -> dict:
    """
    Perform comprehensive market research using DuckDuckGo search and AI analysis.
    
    Args:
        query: The market research query (e.g., "AI coding assistants market")
        max_results: Number of search results to analyze (default: 5)
    
    Returns:
        dict: Structured market research with trends, competitors, and opportunities
    """
    auth_err = _require_auth()
    if auth_err:
        return auth_err
    try:
        log.info(f"Searching for: {query}")
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
        
        # 3. Build analysis prompt
        system_prompt = (
            "You are an expert market research analyst. "
            "Analyze search results and provide structured insights. "
            "Be factual, specific, and conservative with confidence estimates. "
            "Focus on actionable intelligence."
        )
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
        
        # 4. Run agent with automatic fallback if primary model fails
        result = await run_agent_with_fallback(prompt, system_prompt)
        
        # 5. Parse and validate response
        # Try to parse as MarketResearchResult
        try:
            # Extract the data field from the agent result
            data = result.data if hasattr(result, 'data') else result
            
            # If it's a string, try to parse it as JSON
            if isinstance(data, str):
                parsed = _extract_json(data)
                if parsed is None:
                    raise ValueError("Model response did not contain valid JSON object")
                data = parsed
            
            # Validate with Pydantic model
            validated = MarketResearchResult.model_validate(data)
            return validated.model_dump()
            
        except Exception as parse_error:
            # Return raw response if parsing fails
            log.error(f"Parsing error: {parse_error}")
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
@mcp.tool()
def check_available_models() -> dict:
    """
    Check which AI models are available and their configuration status.
    
    Returns:
        dict: Status of all configured models with availability info
    """
    models_status = []
    
    auth_err = _require_auth()
    if auth_err:
        return auth_err
    for i, config in enumerate(MODEL_CONFIGS, start=1):
        status = {
            "priority": i,
            "model_name": config.model_name,
            "base_url": config.base_url,
            "has_api_key": bool(config.api_key),
            "status": "unknown"
        }
        
        try:
            # Try to create the model
            model = _create_model(config)
            status["status"] = "available"
            status["message"] = "Model initialized successfully"
        except Exception as e:
            status["status"] = "unavailable"
            status["message"] = str(e)
        
        models_status.append(status)
    
    return {
        "total_models": len(MODEL_CONFIGS),
        "primary_model": MODEL_CONFIGS[0].model_name if MODEL_CONFIGS else None,
        "models": models_status,
        "note": "Models are tried in priority order. First available model is used."
    }


@mcp.tool()
def quick_search(query: str, max_results: int = 3) -> dict:
    """
    Quick DuckDuckGo search without AI analysis.
    
    Args:
        query: Search query
        max_results: Number of results (default: 3)
    
    Returns:
        dict: Raw search results
    """
    auth_err = _require_auth()
    if auth_err:
        return auth_err
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
    import threading
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import HTMLResponse, JSONResponse
    import uvicorn
    
    print("\n" + "="*60)
    print("🔐 Market Research MCP Server")
    print(f"   Environment: {ENVIRONMENT.upper()}")
    print("="*60 + "\n")
    
    # Create a temporary Starlette app just for authentication
    async def temp_auth_page(request):
        """Temporary auth page before MCP starts"""
        html = """
        <html>
        <head>
            <title>MCP Server Authentication</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .login-container {
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                    width: 300px;
                }
                h2 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }
                input {
                    width: 100%;
                    padding: 12px;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-sizing: border-box;
                    font-size: 14px;
                }
                button {
                    width: 100%;
                    padding: 12px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    margin-top: 20px;
                }
                button:hover {
                    background: #5568d3;
                }
                .error {
                    color: #e74c3c;
                    text-align: center;
                    margin-top: 10px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h2>🔐 MCP Server Login</h2>
                <p style="text-align: center; color: #666; font-size: 14px;">
                    Server will start after authentication
                </p>
                <form id="loginForm">
                    <input type="text" id="username" placeholder="Username" required>
                    <input type="password" id="password" placeholder="Password" required>
                    <button type="submit">Login & Start Server</button>
                    <div id="error" class="error"></div>
                </form>
            </div>
            
            <script>
                document.getElementById('loginForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const username = document.getElementById('username').value;
                    const password = document.getElementById('password').value;
                    
                    const response = await fetch('/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({username: username, password: password})
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.body.innerHTML = `
                            <div class="login-container" style="text-align: center;">
                                <h1 style="color: green;">✓ Authentication Successful!</h1>
                                <p>MCP Server is now starting...</p>
                                <p style="color: #666; font-size: 14px;">You can close the window now.</p>
                            </div>
                        `;
                        setTimeout(() => window.close(), 3000);
                    } else {
                        document.getElementById('error').textContent = result.error || 'Invalid username or password!';
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    async def temp_auth_login(request):
        """Handle authentication and signal to start MCP server"""
        global authenticated_user, authentication_complete
        
        try:
            data = await request.json()
            username = data.get('username', '')
            password = data.get('password', '')
            
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                authenticated_user = {
                    'username': username,
                    'authenticated': True,
                    'timestamp': time.time()
                }
                authentication_complete = True
                token = serializer.dumps({"u": username, "ts": int(time.time())})
                return JSONResponse({
                    "success": True,
                    "message": "Authentication successful. Starting MCP server...",
                    "token": token
                })
            else:
                return JSONResponse({
                    "success": False,
                    "error": "Invalid username or password"
                }, status_code=401)
                
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=400)
    
    # Create temporary auth app
    # Expose both '/' and '/login' for GET (UI) and '/login' for POST (submission)
    temp_app = Starlette(routes=[
        Route('/', temp_auth_page),
        Route('/login', temp_auth_page, methods=['GET']),
        Route('/login', temp_auth_login, methods=['POST']),
    ])
    
    print("="*60)
    if SKIP_STARTUP_AUTH:
        print("⚠️  Skipping authentication (SKIP_STARTUP_AUTH=true)")
    else:
        print(f"🔐 Starting Authentication Server on port {AUTH_PORT}...")
    print("="*60 + "\n")
    
    # Run auth server in background thread
    auth_server_ref = {"server": None}
    def run_auth_server():
        config = uvicorn.Config(temp_app, host="0.0.0.0", port=AUTH_PORT, log_level="error")
        server = uvicorn.Server(config)
        auth_server_ref["server"] = server
        server.run()
    
    if not SKIP_STARTUP_AUTH:
        auth_thread = threading.Thread(target=run_auth_server, daemon=True)
        auth_thread.start()
        time.sleep(2)  # Let auth server start

        # Determine which URL to present/open
       
            # Use provided public base if available, else fallback to localhost (warning)
        if PUBLIC_AUTH_BASE_URL:
            # Ensure trailing slash for consistency
            base = PUBLIC_AUTH_BASE_URL.rstrip('/')
            login_url = f"{base}/login"
        else:
            login_url = f"http://localhost:{AUTH_PORT}/login"
            print("⚠️  PUBLIC_AUTH_BASE_URL not set; using localhost fallback. Set PUBLIC_AUTH_BASE_URL for deployed environment.")

        print(f"🌐 Authentication page: {login_url}")
        print("   (If browser doesn't open, manually visit the URL above)\n")
        try:
            webbrowser.open(login_url)
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
        print("⏳ Waiting for authentication...")
        max_wait = int(os.getenv("AUTH_MAX_WAIT", "300"))
        for i in range(max_wait):
            if authentication_complete:
                break
            time.sleep(1)
            if (i + 1) % 10 == 0:
                print(f"   Still waiting... ({i + 1}s elapsed)")
        if not authentication_complete:
            print("\n❌ Authentication timed out!")
            print("   Proceeding without tool access (tools will return Not authenticated).")
    else:
        authentication_complete = True
        authenticated_user = {"username": "skipped", "authenticated": True, "timestamp": time.time()}
    if authentication_complete:
        print(f"\n✅ Authentication Successful!")
        print(f"   User: {authenticated_user.get('username')}\n")
    else:
        print("⚠️  Starting MCP with auth disabled (tools gated).\n")
    print("="*60)
    print(f"🚀 Starting MCP Server on port {MCP_PORT}...")
    print("="*60 + "\n")
    if not authentication_complete:
        print("⚠️  Tools will reject requests until authenticated.\n")
    else:
        print("✅ MCP Server Starting! Tools are accessible.\n")
    mcp.run(transport="streamable-http", port=MCP_PORT)