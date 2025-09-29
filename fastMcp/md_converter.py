# mcp_server.py
from fastmcp import FastMCP
from markitdown import MarkItDown

mcp = FastMCP("File Converter MCP Server", host="127.0.0.1", port=8001)

@mcp.tool
def convert(file_path: str) -> str:
    """Converts a file to markdown format."""
    md = MarkItDown(enable_plugins=False)
    result = md.convert(file_path)
    output_file = f"{file_path.split('.')[0]}.md"
    with open(output_file, "w") as f:
        f.write(result.text_content)
    return f"Markdown output saved to {output_file}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
