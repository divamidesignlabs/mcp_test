"""
MCP Server for fetching content from links using API authentication.
Supports token ID and token secret authentication for bks.divami.com API.
Automatically discovers and explores nested endpoints.
"""

from fastmcp import FastMCP
import requests
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import json
import re
from urllib.parse import urljoin, urlparse

# Load environment variables
load_dotenv()

# Default base URL for the API
DEFAULT_BASE_URL = "https://bks.divami.com"

mcp = FastMCP("Hyperlink Content Fetcher")

def extract_endpoints_from_json(data: Any, base_url: str) -> List[str]:
    """
    Recursively extract potential endpoint URLs from JSON response.
    Looks for links, hrefs, urls, and path-like strings.
    """
    endpoints = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Look for common link fields
            if key.lower() in ['url', 'href', 'link', 'endpoint', 'path', 'uri']:
                if isinstance(value, str):
                    # Convert relative paths to full URLs
                    if value.startswith('/'):
                        endpoints.append(urljoin(base_url, value))
                    elif value.startswith('http'):
                        endpoints.append(value)
            
            # Recursively search nested structures
            endpoints.extend(extract_endpoints_from_json(value, base_url))
    
    elif isinstance(data, list):
        for item in data:
            endpoints.extend(extract_endpoints_from_json(item, base_url))
    
    return list(set(endpoints))  # Remove duplicates

def extract_endpoints_from_html(html: str, base_url: str) -> List[str]:
    """
    Extract API endpoint links from HTML.
    Looks for anchor tags and potential API paths.
    """
    endpoints = []
    
    # Find all href attributes
    href_pattern = r'href=["\']([^"\']+)["\']'
    hrefs = re.findall(href_pattern, html)
    
    for href in hrefs:
        # Skip non-API links (javascript, anchors, etc.)
        if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
            continue
        
        # Convert relative URLs to absolute
        if href.startswith('/'):
            full_url = urljoin(base_url, href)
            endpoints.append(full_url)
        elif href.startswith('http'):
            # Only include URLs from the same domain
            if urlparse(href).netloc == urlparse(base_url).netloc:
                endpoints.append(href)
    
    return list(set(endpoints))

@mcp.tool
def fetch_content(
    endpoint: str,
    token_id: Optional[str] = None,
    token_secret: Optional[str] = None,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    explore_nested: bool = True,
    max_depth: int = 2
) -> str:
    """
    Fetches content from bks.divami.com API and discovers nested endpoints.
    
    Args:
        endpoint: The API endpoint path (e.g., "/shelves" or full URL)
        token_id: API token ID (can also use TOKEN_ID env var)
        token_secret: API token secret (can also use TOKEN_SECRET env var)
        base_url: Base URL for the API (defaults to https://bks.divami.com)
        headers: Optional additional headers as a dictionary
        explore_nested: Whether to discover and fetch nested endpoints (default: True)
        max_depth: Maximum depth for nested exploration (default: 2)
    
    Returns:
        JSON string containing the fetched content and discovered endpoints
    
    Examples:
        - fetch_content("/shelves")
        - fetch_content("/shelves", explore_nested=True, max_depth=3)
    """
    # Get credentials from environment if not provided
    token_id = token_id or os.getenv("TOKEN_ID")
    token_secret = token_secret or os.getenv("TOKEN_SECRET")
    base_url =  DEFAULT_BASE_URL

    if not token_id or not token_secret:
        return json.dumps({
            "error": "Both token_id and token_secret are required. Provide them as parameters or set TOKEN_ID and TOKEN_SECRET environment variables."
        })
    
    # Ensure base_url has proper scheme
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    
    # Construct full URL if endpoint is relative
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        url = endpoint
    else:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    # Prepare headers
    request_headers = headers.copy() if headers else {}
    request_headers["X-Token-ID"] = token_id
    request_headers["X-Token-Secret"] = token_secret
    
    result = {
        "main_endpoint": url,
        "content": None,
        "discovered_endpoints": [],
        "nested_content": {}
    }
    
    try:
        # Fetch main endpoint
        response = requests.get(url, headers=request_headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        
        # Parse and store main content
        if "application/json" in content_type:
            json_data = response.json()
            result["content"] = json_data
            
            # Discover endpoints from JSON
            if explore_nested and max_depth > 0:
                discovered = extract_endpoints_from_json(json_data, base_url)
                result["discovered_endpoints"] = discovered
                
                # Fetch nested endpoints
                for nested_url in discovered[:10]:  # Limit to first 10 to avoid overload
                    try:
                        nested_response = requests.get(nested_url, headers=request_headers, timeout=30)
                        nested_response.raise_for_status()
                        
                        if "application/json" in nested_response.headers.get("Content-Type", ""):
                            result["nested_content"][nested_url] = nested_response.json()
                        else:
                            result["nested_content"][nested_url] = nested_response.text[:500]  # Truncate HTML
                    except Exception as e:
                        result["nested_content"][nested_url] = f"Error: {str(e)}"
        
        else:
            # HTML or other content
            result["content"] = response.text
            
            # Discover endpoints from HTML
            if explore_nested and max_depth > 0:
                discovered = extract_endpoints_from_html(response.text, base_url)
                result["discovered_endpoints"] = discovered[:20]  # Limit to 20 endpoints
        
        return json.dumps(result, indent=2)
            
    except requests.exceptions.RequestException as e:
        result["error"] = f"Error fetching content: {str(e)}"
        return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")