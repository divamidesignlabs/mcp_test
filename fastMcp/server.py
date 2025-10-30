
# from md_converter import mcp as md_converter_tool
# import contextlib
# from fastapi import FastAPI


# # Create a combined lifespan to manage both session managers
# @contextlib.asynccontextmanager
# async def lifespan(app: FastAPI):
#     async with contextlib.AsyncExitStack() as stack:
#         await stack.enter_async_context(md_converter_tool.session_manager.run())
#         # await stack.enter_async_context(math_mcp.session_manager.run())
#         yield

# app = FastAPI(lifespan=lifespan)
# app.mount("/convert", md_converter_tool.streamable_http_app())
# # app.mount("/math", math_mcp.streamable_http_app())


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


"""
Unified MCP Server - Combines Market Research and BookStack AI MCP Servers

This server mounts both services under different path prefixes:
- /market/* - Market Research MCP endpoints
- /bookstack/* - Smart BookStack MCP endpoints

Run with: uv run server.py
"""
import os
import logging
from fastapi import FastAPI, Depends, HTTPException, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.security import HTTPAuthorizationCredentials, HTTPAuthorizationCredentials, HTTPBearer
from fastmcp import FastMCP
# from fastMcp.market_research import verify_token
from fastapi_mcp import FastApiMCP, AuthConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Create main FastAPI app
app = FastAPI(
    title="Unified MCP Server",
    description="Combined Market Research and BookStack AI MCP Services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers from both services
try:
    # Import Market Research router
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from market_research import router as market_router
    from pydantic_bks_copy import router as bookstack_router
    
    log.info("✅ Successfully imported Market Research router")
    log.info("✅ Successfully imported BookStack router")
    
except ImportError as e:
    log.error(f"❌ Failed to import routers: {e}")
    raise

# Mount routers with prefixes
app.include_router(market_router, prefix="/market", tags=["Market Research"])
app.include_router(bookstack_router, prefix="/bookstack", tags=["BookStack AI"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint showing available services"""
    return {
        "message": "🚀 Unified MCP Server",
        "version": "1.0.0",
        "services": {
            "market_research": {
                "prefix": "/market",
                "endpoints": [
                    "GET /market/market_research?query=<query>&max_results=<n>",
                    "GET /market/quick_search?query=<query>&max_results=<n>",
                    "GET /market/mcp"
                ],
                "description": "AI-powered market research using DuckDuckGo and LLM analysis"
            },
            "bookstack": {
                "prefix": "/bookstack",
                "endpoints": [
                    "GET /bookstack/fetch/{endpoint_type}/{resource_id}",
                    "GET /bookstack/search?q=<query>&page=<n>&count=<n>",
                    "GET /bookstack/health",
                    "GET /bookstack/mcp"
                ],
                "description": "Smart BookStack documentation access with AI query understanding"
            }
        },
        "docs": {
            "interactive": "/docs",
            "redoc": "/redoc"
        },
        "authentication": "Bearer token required for all endpoints (use STATIC_TOKEN from .env)"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Combined health check for both services"""
    return {
        "status": "healthy",
        "services": {
            "market_research": "operational",
            "bookstack": "operational"
        },
        "environment": {
            "litellm_configured": bool(os.getenv("LITELLM_MASTER_KEY")),
            "bookstack_configured": bool(os.getenv("TOKEN_ID") and os.getenv("TOKEN_SECRET")),
            "auth_configured": bool(os.getenv("STATIC_TOKEN"))
        }
    }

# Configuration info endpoint
@app.get("/config")
async def config_info():
    """Display configuration status (without exposing secrets)"""
    return {
        "litellm": {
            "base_url": os.getenv("LITELLM_BASE_URL", "https://litellm.divami.com"),
            "model": os.getenv("MODEL_NAME", "gemini/gemini-1.5-flash"),
            "key_configured": bool(os.getenv("LITELLM_MASTER_KEY"))
        },
        "bookstack": {
            "target_base": os.getenv("TARGET_BASE", "https://bks.divami.com"),
            "api_base": os.getenv("API_BASE", "https://bks.divami.com/api"),
            "token_configured": bool(os.getenv("TOKEN_ID") and os.getenv("TOKEN_SECRET"))
        },
        "authentication": {
            "method": "Bearer Token",
            "token_configured": bool(os.getenv("STATIC_TOKEN"))
        }
    }

STATIC_TOKEN = os.getenv("STATIC_TOKEN")

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

mcp = FastApiMCP(
    app,
    name="unified-mcp-server",
    auth_config=AuthConfig(
        dependencies=[Depends(verify_token)]
    )
)

mcp.mount_http()


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8010"))
    
    log.info("=" * 60)
    log.info("🚀 Starting Unified MCP Server")
    log.info("=" * 60)
    log.info(f"📍 Host: {host}")
    log.info(f"🔌 Port: {port}")
    log.info(f"📚 Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    log.info("=" * 60)
    log.info("Available Services:")
    log.info("  📊 Market Research MCP: /market/*")
    log.info("  📖 BookStack AI MCP: /bookstack/*")
    log.info("=" * 60)
    
    # Check critical environment variables
    if not os.getenv("STATIC_TOKEN"):
        log.warning("⚠️  STATIC_TOKEN not set - authentication will fail!")
    
    if not os.getenv("LITELLM_MASTER_KEY"):
        log.warning("⚠️  LITELLM_MASTER_KEY not set - LLM calls may fail!")
    
    if not (os.getenv("TOKEN_ID") and os.getenv("TOKEN_SECRET")):
        log.warning("⚠️  BookStack credentials not set - BookStack endpoints will fail!")
    
    log.info("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )