import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig

load_dotenv()
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
    name="TEST",
    auth_config=AuthConfig(
        dependencies=[Depends(verify_token)]
    )
)

mcp.mount_http()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    has_token = bool(STATIC_TOKEN and STATIC_TOKEN != "mysecrettoken")
    print(f"🔐 STATIC_TOKEN configured: {'✅ Yes' if has_token else '❌ No (using default)'}")
    print(f"🚀 Starting server on http://0.0.0.0:{port}")
    print(f"🔑 Use Bearer token: {STATIC_TOKEN}")
    print(f"📝 Example: curl -H 'Authorization: Bearer {STATIC_TOKEN}' http://localhost:{port}/mcp")
    