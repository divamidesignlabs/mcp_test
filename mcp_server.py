from fastmcp import FastMCP
from markitdown import MarkItDown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import threading
import uvicorn
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP and FastAPI
mcp = FastMCP("My MCP Server")
app = FastAPI(title="File Converter API", description="API for converting files to markdown")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure environment variables with defaults
HOST = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost for safety
PORT = int(os.getenv("API_PORT", "8000"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ConvertRequest(BaseModel):
    file_path: str

class ConvertResponse(BaseModel):
    message: str
    output_file: str

def is_safe_path(file_path: str) -> bool:
    """Check if the file path is safe to use."""
    abs_path = os.path.abspath(file_path)
    try:
        return os.path.exists(abs_path) and not os.path.islink(abs_path)
    except (TypeError, ValueError, OSError):
        return False

@mcp.tool
def convert(file_path: str) -> str:
    """Converts a file to markdown format."""
    md = MarkItDown(enable_plugins=False)
    result = md.convert(file_path)
    output_file = f"{file_path.split('.')[0]}.md" # Save with .md extension
    with open(output_file, "w") as f:
        f.write(result.text_content)
    return f"Markdown output saved to {output_file}"

@app.post("/convert", response_model=ConvertResponse)
async def convert_file(request: ConvertRequest):
    """API endpoint to convert a file to markdown format."""
    try:
        # Validate file path
        if not is_safe_path(request.file_path):
            raise HTTPException(status_code=400, detail="Invalid or unsafe file path")
        
        # Generate safe output path
        input_filename = os.path.basename(request.file_path)
        output_filename = f"{os.path.splitext(input_filename)[0]}.md"
        output_file = os.path.join(UPLOAD_DIR, output_filename)
        
        # Convert the file
        try:
            md = MarkItDown(enable_plugins=False)
            result = md.convert(request.file_path)
            
            with open(output_file, "w") as f:
                f.write(result.text_content)
            
            logger.info(f"Successfully converted {input_filename} to markdown")
            return ConvertResponse(
                message="File successfully converted to markdown",
                output_file=output_file
            )
        except Exception as e:
            logger.error(f"Error converting file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error converting file")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "File Converter API"}

def run_api_server():
    """Run the FastAPI server in a separate thread."""
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except Exception as e:
        logger.error(f"API server error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Start the API server in a separate thread
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        logger.info(f"Starting API Server on http://{HOST}:{PORT}")
        logger.info(f"API Documentation available at http://{HOST}:{PORT}/docs")
        logger.info("Starting MCP Server...")
        
        # Run the MCP server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise