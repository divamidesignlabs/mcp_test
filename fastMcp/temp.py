"""Production-Ready MCP Server with Authentication Integration.

Features:
- FastMCP integration with proper tool registration
- Environment-based authentication (dev/prod)
- Session token management with itsdangerous
- Middleware for authentication
- Proper error handling and logging
- Health check endpoints
"""
import os
import sys
import time
import json
import logging
import secrets
import webbrowser
import threading
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import FastMCP for proper MCP server setup
from fastmcp import FastMCP

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mcp-server")

# === LOAD ENVIRONMENT ===
load_dotenv()

# === CONFIG ===
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
TOKEN_EXPIRY = int(os.getenv("TOKEN_EXPIRY", 3600))  # seconds
SKIP_AUTH = os.getenv("SKIP_STARTUP_AUTH", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development or production
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
AUTH_USERNAME = os.getenv("MCP_AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("MCP_AUTH_PASSWORD", "admin123")
PUBLIC_AUTH_BASE_URL = os.getenv("PUBLIC_AUTH_BASE_URL", f"http://localhost:{MCP_PORT}")
SESSION_FILE = ".session_token"

# API Keys (example for external services)
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# === AUTHENTICATION STATE ===
authenticated_users: Dict[str, dict] = {}  # {token: {username, timestamp}}
authentication_complete = False

# === FASTMCP SETUP ===
mcp = FastMCP("ProductionMCPServer")

# === PYDANTIC MODELS ===
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenRequest(BaseModel):
    token: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    
class HealthResponse(BaseModel):
    status: str
    authenticated: bool
    environment: str
    tools_available: int

# === TOKEN UTILITIES ===
def get_serializer():
    """Get configured serializer for token generation/verification."""
    return URLSafeTimedSerializer(SESSION_SECRET)

def generate_session_token(username: str) -> str:
    """Generate a new session token for authenticated user."""
    s = get_serializer()
    token = s.dumps({"user": username, "timestamp": time.time()})
    return token

def verify_session_token(token: str) -> dict:
    """Verify session token and return user data."""
    s = get_serializer()
    try:
        data = s.loads(token, max_age=TOKEN_EXPIRY)
        return data
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Token expired")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token verification failed: {str(e)}")

def save_token(token: str):
    """Save token to file for persistence."""
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(token)
        logger.info(f"Session token saved to {SESSION_FILE}")
    except Exception as e:
        logger.error(f"Failed to save token: {e}")

def load_token() -> Optional[str]:
    """Load token from file if exists."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load token: {e}")
    return None

# === AUTH MIDDLEWARE ===
async def get_current_user(request: Request) -> Optional[dict]:
    """Extract and verify user from request token."""
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            user_data = verify_session_token(token)
            return user_data
        except HTTPException:
            pass
    
    # Check query parameter
    token = request.query_params.get("token")
    if token:
        try:
            user_data = verify_session_token(token)
            return user_data
        except HTTPException:
            pass
    
    return None

def require_auth(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency that requires authentication."""
    if not user and not SKIP_AUTH and not authentication_complete:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please login at /auth"
        )
    return user or {"username": "anonymous", "authenticated": False}

# === FASTAPI APP WITH LIFESPAN ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting MCP Server...")
    # Startup logic
    if not gemini_key and not openai_key:
        logger.warning("⚠️  No API keys configured (GEMINI_API_KEY or OPENAI_API_KEY)")
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down MCP Server...")

app = FastAPI(
    title="Production MCP Server",
    version="1.0.0",
    description="MCP Server with integrated authentication and tool management",
    lifespan=lifespan
)

# Add CORS middleware for browser-based authentication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AUTH ROUTES ===
AUTH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Server Authentication</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 400px;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 14px;
        }
        input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 15px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            margin-top: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 15px;
            font-size: 14px;
            padding: 10px;
            background: #fee;
            border-radius: 4px;
            display: none;
        }
        .error.show {
            display: block;
        }
        .success {
            text-align: center;
            color: #27ae60;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>🔐 MCP Server</h2>
        <p class="subtitle">Authenticate to access MCP tools</p>
        
        <form id="loginForm">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" placeholder="Enter username" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="Enter password" required>
            </div>
            <button type="submit" id="loginBtn">Login & Start Server</button>
            <div id="error" class="error"></div>
        </form>
    </div>
    
    <script>
        const form = document.getElementById('loginForm');
        const errorDiv = document.getElementById('error');
        const loginBtn = document.getElementById('loginBtn');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            loginBtn.disabled = true;
            errorDiv.classList.remove('show');
            
            try {
                const response = await fetch('/auth/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password})
                });
                
                const result = await response.json();
                
                if (response.ok && result.success) {
                    document.body.innerHTML = `
                        <div class="login-container" style="text-align: center;">
                            <div class="success">
                                <h1 style="color: #27ae60; font-size: 48px; margin: 20px 0;">✓</h1>
                                <h2>Authentication Successful!</h2>
                                <p style="color: #666; margin-top: 20px;">MCP Server is now ready</p>
                                <p style="color: #999; font-size: 12px; margin-top: 20px;">You can close this window</p>
                            </div>
                        </div>
                    `;
                } else {
                    throw new Error(result.message || 'Authentication failed');
                }
            } catch (error) {
                loginBtn.disabled = false;
                errorDiv.textContent = error.message || 'Invalid username or password';
                errorDiv.classList.add('show');
            }
        });
    </script>
</body>
</html>
"""

@app.get("/auth", response_class=HTMLResponse)
async def auth_page():
    """Authentication page for browser-based login."""
    return HTMLResponse(content=AUTH_HTML)

@app.post("/auth/login")
async def login(request: Request):
    """Handle login."""
    global authentication_complete, authenticated_users
    
    try:
        body = await request.json()
        username = body.get("username")
        password = body.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Missing username or password")
        
        # Verify credentials
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            token = generate_session_token(username)
            save_token(token)
            
            # Update authentication state
            authentication_complete = True
            authenticated_users[token] = {
                "username": username,
                "timestamp": time.time()
            }
            
            logger.info(f"✅ User '{username}' authenticated successfully")
            
            return JSONResponse({
                "success": True,
                "message": "Login successful",
                "token": token
            })
        else:
            logger.warning(f"❌ Failed login attempt for user '{username}'")
            raise HTTPException(status_code=401, detail="Invalid username or password")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/verify")
async def verify(request: Request):
    """Verify an existing token."""
    body = await request.json()
    token = body.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing 'token'")
    data = verify_session_token(token)
    return {"message": "Token valid", "data": data}

@app.post("/auth/refresh")
async def refresh(request: Request):
    """Refresh a valid token and issue a new one."""
    body = await request.json()
    token = body.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing 'token'")
    data = verify_session_token(token)
    new_token = generate_session_token(data["user"])
    save_token(new_token)
    return {"message": "Token refreshed", "token": new_token}

# === HEALTH & STATUS ROUTES ===
@app.get("/")
async def root():
    """Root endpoint with server status."""
    return {
        "status": "ok",
        "message": "Production MCP Server is running",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "authenticated": authentication_complete,
        "auth_required": not SKIP_AUTH
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        authenticated=authentication_complete,
        environment=ENVIRONMENT,
        tools_available=len(mcp._tools) if hasattr(mcp, '_tools') else 0
    )

# === MCP TOOL EXAMPLES ===
@mcp.tool()
def market_research(query: str, max_results: int = 5, user: dict = Depends(require_auth)) -> dict:
    """
    Perform market research using DuckDuckGo search.
    
    Args:
        query: The research query
        max_results: Number of results to return
    
    Returns:
        dict: Research results
    """
    if not authentication_complete and not SKIP_AUTH:
        return {"error": "Not authenticated", "detail": "Login required at /auth"}
    
    logger.info(f"Market research query: {query} (user: {user.get('username', 'unknown')})")
    
    # Example implementation
    return {
        "tool": "market_research",
        "query": query,
        "max_results": max_results,
        "status": "available",
        "message": "Tool ready for use",
        "user": user.get("username")
    }

@mcp.tool()
def quick_search(query: str, user: dict = Depends(require_auth)) -> dict:
    """
    Quick search tool.
    
    Args:
        query: Search query
    
    Returns:
        dict: Search results
    """
    if not authentication_complete and not SKIP_AUTH:
        return {"error": "Not authenticated", "detail": "Login required at /auth"}
    
    logger.info(f"Quick search: {query} (user: {user.get('username', 'unknown')})")
    
    return {
        "tool": "quick_search",
        "query": query,
        "status": "available",
        "user": user.get("username")
    }

# === STARTUP AUTHENTICATION ===
def run_startup_authentication():
    """Run authentication flow before starting MCP server."""
    global authentication_complete, authenticated_users
    
    logger.info("="*60)
    logger.info("🔐 Production MCP Server - Authentication")
    logger.info(f"   Environment: {ENVIRONMENT.upper()}")
    logger.info("="*60)
    
    if SKIP_AUTH:
        logger.warning("⚠️  Skipping authentication (SKIP_STARTUP_AUTH=true)")
        authentication_complete = True
        authenticated_users["skipped"] = {"username": "admin", "timestamp": time.time()}
        return True
    
    # Check for existing valid token
    existing_token = load_token()
    if existing_token:
        try:
            data = verify_session_token(existing_token)
            logger.info(f"✅ Found valid session token for user: {data['user']}")
            authentication_complete = True
            authenticated_users[existing_token] = {"username": data["user"], "timestamp": time.time()}
            return True
        except HTTPException as e:
            logger.warning(f"⚠️  Existing token invalid: {e.detail}")
    
    # Need to authenticate via browser
    auth_url = f"{PUBLIC_AUTH_BASE_URL.rstrip('/')}/auth"
    
    logger.info(f"\n🔐 Authentication Required")
    logger.info(f"   URL: {auth_url}")
    logger.info(f"   Credentials: {AUTH_USERNAME} / {AUTH_PASSWORD}")
    
    if ENVIRONMENT == "development":
        logger.info(f"\n🌐 Opening browser to: {auth_url}")
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            logger.warning(f"⚠️  Could not open browser: {e}")
    else:
        logger.info(f"\n⚠️  PRODUCTION MODE")
        logger.info(f"   Visit {auth_url} to authenticate")
    
    logger.info(f"\n⏳ Waiting for authentication...")
    
    max_wait = int(os.getenv("AUTH_TIMEOUT", "300"))
    for elapsed in range(max_wait):
        if authentication_complete:
            logger.info(f"\n✅ Authentication successful after {elapsed}s")
            return True
        time.sleep(1)
        if (elapsed + 1) % 30 == 0:
            logger.info(f"   Still waiting... ({elapsed + 1}s elapsed)")
    
    logger.error("\n❌ Authentication timeout!")
    logger.warning("   Server will start but tools will be gated")
    return False

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    import uvicorn
    
    # Start FastAPI server in background thread for authentication
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=MCP_PORT, log_level="error")
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Wait for FastAPI to start
    time.sleep(2)
    
    # Run authentication flow
    auth_success = run_startup_authentication()
    
    if not auth_success and not SKIP_AUTH:
        logger.warning("⚠️  Proceeding without authentication - tools will be gated")
    
    logger.info("\n" + "="*60)
    logger.info("🚀 MCP Server Ready")
    logger.info(f"   FastAPI: http://0.0.0.0:{MCP_PORT}")
    logger.info(f"   Auth: {PUBLIC_AUTH_BASE_URL.rstrip('/')}/auth")
    logger.info(f"   Health: http://localhost:{MCP_PORT}/health")
    logger.info(f"   Authenticated: {authentication_complete}")
    logger.info("="*60 + "\n")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down MCP Server...")
        sys.exit(0)
