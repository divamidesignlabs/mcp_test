"""Authentication server module for Market Research MCP.

Encapsulates the temporary Starlette-based login flow used to gate MCP startup.
"""
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse
import threading
import time
import uvicorn
import webbrowser
from typing import Optional, Dict
from fastapi_mcp import FastApiMCP, AuthConfig


def run_authentication(port: int, environment: str, username: str, password: str, timeout: int = 300) -> Dict:
    """Run a temporary authentication server and block until user authenticates.

    Args:
        port: Port to run the temporary auth server.
        environment: 'development' or 'production'. In development opens browser automatically.
        username: Expected username.
        password: Expected password.
        timeout: Seconds to wait before giving up (default 300).

    Returns:
        Dict with authenticated user info: { 'username': str, 'authenticated': bool, 'timestamp': float }

    Raises:
        TimeoutError: If authentication not completed within timeout.
    """
    authenticated_user: Optional[Dict] = None
    authentication_complete = False

    # ----------------------
    # Route handlers
    # ----------------------
    async def temp_auth_page(request):
        html = """
        <html>
        <head>
            <title>MCP Server Authentication</title>
            <style>
                body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .login-container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); width: 300px; }
                h2 { text-align: center; color: #333; margin-bottom: 30px; }
                input { width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; font-size: 14px; }
                button { width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; }
                button:hover { background: #5568d3; }
                .error { color: #e74c3c; text-align: center; margin-top: 10px; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h2>🔐 MCP Server Login</h2>
                <p style="text-align: center; color: #666; font-size: 14px;">Server will start after authentication</p>
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
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({username: username, password: password})
                    });
                    const result = await response.json();
                    if (response.ok) {
                        document.body.innerHTML = `
                            <div class="login-container" style="text-align: center;">
                                <h1 style="color: green;">✓ Authentication Successful!</h1>
                                <p>MCP Server is now starting...</p>
                                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
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
        nonlocal authenticated_user, authentication_complete
        try:
            data = await request.json()
            in_username = data.get('username', '')
            in_password = data.get('password', '')
            if in_username == username and in_password == password:
                authenticated_user = {
                    'username': in_username,
                    'authenticated': True,
                    'timestamp': time.time()
                }
                authentication_complete = True
                return JSONResponse({"success": True, "message": "Authentication successful. Starting MCP server..."})
            return JSONResponse({"success": False, "error": "Invalid username or password"}, status_code=401)
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=400)

    # ----------------------
    # Build Starlette app
    # ----------------------
    temp_app = Starlette(routes=[
        Route('/', temp_auth_page),
        Route('/login', temp_auth_login, methods=['POST']),
    ])

    print("=" * 60)
    print(f"🔐 Starting Authentication Server on port {port}...")
    print("=" * 60 + "\n")

    server_config = uvicorn.Config(temp_app, host="0.0.0.0", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(server_config)

    def run_server():
        uvicorn_server.run()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    time.sleep(2)  # allow server startup

    if environment == "development":
        auth_url = f"http://localhost:{port}"
        print(f"🌐 Opening browser to: {auth_url}")
        print("   (If browser doesn't open, manually visit the URL above)\n")
        try:
            webbrowser.open(auth_url)
        except Exception:
            print("   (Browser open failed; continue manually)\n")
    else:
        print("⚠️  PRODUCTION MODE")
        print("   Visit the server URL to authenticate and start MCP\n")

    print("⏳ Waiting for authentication...")

    for elapsed in range(timeout):
        if authentication_complete:
            break
        time.sleep(1)
        if (elapsed + 1) % 10 == 0:
            print(f"   Still waiting... ({elapsed + 1}s elapsed)")

    if not authentication_complete:
        print("\n❌ Authentication timed out!\n   Server will not start.\n")
        # Signal server exit and raise
        uvicorn_server.should_exit = True
        raise TimeoutError("Authentication timed out")

    print(f"\n✅ Authentication Successful!\n   User: {authenticated_user.get('username')}\n")

    # Stop auth server cleanly before returning
    uvicorn_server.should_exit = True
    # Give server loop a moment to exit
    time.sleep(0.5)

    return authenticated_user  # type: ignore
