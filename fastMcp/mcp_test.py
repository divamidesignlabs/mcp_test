from fastapi import FastAPI, UploadFile, File
from markitdown import MarkItDown
import uvicorn
import os

app = FastAPI(title="MCP File-to-Markdown Server")

@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    """Converts an uploaded file to Markdown format."""
    content = await file.read()
    md = MarkItDown(enable_plugins=False)
    result = md.convert((file.filename, content))

    return {
        "filename": file.filename,
        "markdown": result.text_content
    }

@app.get("/")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
