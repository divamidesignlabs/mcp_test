# """
# This example shows how to reject any request without a valid token passed in the Authorization header.

# In order to configure the auth header, the config file for the MCP server should looks like this:
# ```json
# {
#   "mcpServers": {
#     "remote-example": {
#       "command": "npx",
#       "args": [
#         "mcp-remote",
#         "http://localhost:8000/mcp",
#         "--header",
#         "Authorization:${AUTH_HEADER}"
#       ]
#     },
#     "env": {
#       "MCP_ACCESS_TOKEN": "Bearer test_token_123456789"
#     }
#   }
# }
# ```
# """
# # from examples.shared.apps.items import app  # The FastAPI app
# # from examples.shared.setup import setup_logging

# from fastapi import Depends, FastAPI
# from fastapi.security import HTTPBearer

# from fastapi_mcp import FastApiMCP, AuthConfig


# # Scheme for the Authorization header
# token_auth_scheme = HTTPBearer()

# app = FastAPI()
    
# # Create a private endpoint
# @app.get("/private")
# async def private(token=Depends(token_auth_scheme)):
#     return token.credentials


# # Create the MCP server with the token auth scheme
# mcp = FastApiMCP(
#     app,
#     name="Protected MCP",
#     auth_config=AuthConfig(
#         dependencies=[Depends(token_auth_scheme)],
#     ),
# )

# # Mount the MCP server
# mcp.mount_http()


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

# # from fastapi import FastAPI, Depends
# # from fastapi.security import HTTPBearer
# # from fastapi_mcp import FastApiMCP, AuthConfig

# # token_auth_scheme = HTTPBearer()

# # # Your FastAPI app
# # app = FastAPI()

# # mcp = FastApiMCP(
# #     app,
# #     name="Protected MCP",
# #     auth_config=AuthConfig(
# #         dependencies=[Depends(token_auth_scheme)],
# #     ),
# # )
# # mcp.mount_http()
# """
# This example shows how to reject any request without a valid token passed in the Authorization header.

# In order to configure the auth header, the config file for the MCP server should looks like this:
# ```json
# {
#   "mcpServers": {
#     "remote-example": {
#       "command": "npx",
#       "args": [
#         "mcp-remote",
#         "http://localhost:8000/mcp",
#         "--header",
#         "Authorization:${AUTH_HEADER}"
#       ]
#     },
#     "env": {
#       "AUTH_HEADER": "Bearer <your-token>"
#     }
#   }
# }
# ```
# """

# from examples.shared.apps.items import app  # The FastAPI app
# from examples.shared.setup import setup_logging

# from fastapi import Depends
# from fastapi.security import HTTPBearer

# from fastapi_mcp import FastApiMCP, AuthConfig

# setup_logging()

# # Scheme for the Authorization header
# token_auth_scheme = HTTPBearer()


# # Create a private endpoint
# @app.get("/private")
# async def private(token=Depends(token_auth_scheme)):
#     return token.credentials


# # Create the MCP server with the token auth scheme
# mcp = FastApiMCP(
#     app,
#     name="Protected MCP",
#     auth_config=AuthConfig(
#         dependencies=[Depends(token_auth_scheme)],
#     ),
# )

# # Mount the MCP server
# mcp.mount_http()


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)



"""
fastmcp_auth_handshake.py

Example: FastAPI + WebSocket MCP-style server with an "auth handshake before tools load"
Runtime: Python 3.10+
Requirements: fastapi, uvicorn, python-multipart (optional), itsdangerous (optional)

This example expects a STATIC_TOKEN (env var) and performs a handshake on the same port.
Client MUST authenticate first by sending a JSON message:
    {"type": "auth", "token": "MY_STATIC_TOKEN"}

If token is valid, server replies with {"type": "auth_ok", "session_id": "..."}
and will accept further MCP-style messages (here: simple tool calls).
If invalid or missing, server replies {"type": "auth_failed"} and closes the connection.

This is intentionally small, easy to adapt for FastMCP frameworks: the "auth gating" lives
in the WebSocket accept flow and controls whether the server will process tool messages.

NOTES:
 - Replace STATIC_TOKEN validation with any external validator if needed.
 - For production, use TLS (uvicorn --ssl-keyfile/--ssl-certfile or reverse proxy).
 - Consider rate-limiting and replay protections for tokens.

"""

import os
import asyncio
import json
import secrets
from typing import Callable, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer
from fastmcp import FastMCP
import uvicorn
from fastapi_mcp import FastApiMCP, AuthConfig

# Configuration
STATIC_TOKEN = os.getenv("STATIC_TOKEN", "change-me-now")
SESSION_TTL = 60 * 60  # seconds

app = FastAPI(title="MCP-with-Auth-Handshake (example)")

# Simple in-memory session store (session_id -> meta)
_sessions: Dict[str, Dict[str, Any]] = {}

# Simple tool registry
Tools = Dict[str, Callable[..., Any]]
TOOL_REGISTRY: Tools = {}


def register_tool(name: str):
    def _decor(fn: Callable[..., Any]):
        TOOL_REGISTRY[name] = fn
        return fn
    return _decor

# mcp=FastMCP(app, name="MCP-with-Auth-Handshake")




@app.get("/add")
async def tool_add(a: int, b: int):
    return {"ok": True, "result": a + b}


async def validate_token(token: str) -> bool:
    # Static token check; replace with DB/JWT/remote validator as needed
    await asyncio.sleep(0)  # placeholder for async validators
    return bool(token) and token == STATIC_TOKEN


def create_session_meta() -> Dict[str, Any]:
    sid = secrets.token_urlsafe(16)
    meta = {"sid": sid, "created_at": asyncio.get_event_loop().time()}
    _sessions[sid] = meta
    return meta


async def handle_message(ws: WebSocket, session_meta: Dict[str, Any], msg: dict):
    """Process an MCP-style message after auth."""
    typ = msg.get("type")
    if typ == "call_tool":
        tool_name = msg.get("tool")
        payload = msg.get("payload", {})
        if tool_name not in TOOL_REGISTRY:
            await ws.send_text(json.dumps({"type": "error", "message": f"Unknown tool {tool_name}"}))
            return
        try:
            result = await TOOL_REGISTRY[tool_name](payload)
            await ws.send_text(json.dumps({"type": "tool_result", "tool": tool_name, "result": result}))
        except Exception as e:
            await ws.send_text(json.dumps({"type": "tool_error", "tool": tool_name, "message": str(e)}))
    else:
        await ws.send_text(json.dumps({"type": "error", "message": "Unsupported message type"}))


@app.get("/")
async def index():
    html = """
    <html>
        <head>
            <title>MCP Auth Handshake Example</title>
        </head>
        <body>
            <h3>MCP Auth Handshake Example</h3>
            <p>Use a WebSocket client to connect to <code>/ws</code> and send an auth message.</p>
        </body>
    </html>
    """
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Step 1: ask for auth
        await websocket.send_text(json.dumps({"type": "auth_required", "message": "send: {type: 'auth', token: '...' }"}))

        # Give client a timeout to authenticate (10s here)
        auth_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        try:
            auth_msg = json.loads(auth_raw)
        except Exception:
            await websocket.send_text(json.dumps({"type": "auth_failed", "message": "invalid-json"}))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if auth_msg.get("type") != "auth":
            await websocket.send_text(json.dumps({"type": "auth_failed", "message": "expected auth message"}))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        token = auth_msg.get("token")
        valid = await validate_token(token)
        if not valid:
            await websocket.send_text(json.dumps({"type": "auth_failed", "message": "invalid token"}))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Auth succeeded -> create session and enable tools
        session_meta = create_session_meta()
        await websocket.send_text(json.dumps({"type": "auth_ok", "session_id": session_meta["sid"]}))

        # Now enter normal MCP message loop
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid-json"}))
                continue
            await handle_message(websocket, session_meta, msg)

    except WebSocketDisconnect:
        # client disconnected
        return
    except asyncio.TimeoutError:
        await websocket.send_text(json.dumps({"type": "auth_failed", "message": "auth_timeout"}))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

# TOKEN = os.getenv("MCP_TOKEN", "my-secret")

# app = FastAPI()
    
token_auth_scheme = HTTPBearer()

@app.get("/private")
async def private(token=Depends(token_auth_scheme)):
    return token.credentials

mcp = FastApiMCP(
    app,
    name="TEST",
    # auth_config=AuthConfig(
    #     dependencies=[Depends(token_auth_scheme)],
    # ),
)

# Mount the MCP server
mcp.mount_http()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print("STATIC_TOKEN (configured):", "***" if STATIC_TOKEN else "(not set)")
    # uvicorn.run(app, host="0.0.0.0", port=8003)
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
