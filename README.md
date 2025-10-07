# MCP Test - File Converter with S3 Integration

A Model Context Protocol (MCP) server implementation that provides file conversion capabilities with AWS S3 integration.

## Features

- **File to Markdown Conversion**: Convert various file formats to Markdown using MarkItDown
- **S3 Integration**: Upload original files and converted markdown to AWS S3
- **Batch Processing**: Handle single files or entire directories
- **FastMCP Server**: RESTful API endpoints for file conversion operations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp_test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET=your_bucket_name
S3_PREFIX=optional_prefix
```

## Usage

### Running the MCP Server

```bash
python fastMcp/md_converter_s3.py
```

The server will start on `http://127.0.0.1:8001`

### Available Tools

#### 1. convert
Converts a single file to markdown format locally.

```python
# Example usage
convert("path/to/your/file.pdf")
```

#### 2. upload_and_convert
Uploads files to S3, converts them to markdown, and uploads the markdown back to S3.

```python
# Convert single file
upload_and_convert("path/to/file.pdf", "custom_output_name")

# Convert entire directory
upload_and_convert("path/to/directory")
```

## Project Structure

```
mcp_test/
├── fastMcp/
│   └── md_converter_s3.py    # Main MCP server implementation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── .env                     # Environment variables (not tracked)
└── README.md               # This file
```

## Dependencies

- `fastmcp` - FastMCP framework for building MCP servers
- `markitdown` - File to markdown conversion library
- `boto3` - AWS SDK for Python
- `python-dotenv` - Load environment variables from .env files
- `fastapi` - Web framework for building APIs
- `uvicorn` - ASGI server implementation
- `pydantic` - Data validation and settings management
- `requests` - HTTP library
- `click` - Command line interface creation toolkit

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key ID | Yes | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | Yes | - |
| `AWS_REGION` | AWS region | No | us-east-1 |
| `S3_BUCKET` | S3 bucket name | Yes | - |
| `S3_PREFIX` | Optional prefix for S3 keys | No | - |

## Development

This project is part of the DAKSH workflow system and serves as a testing ground for MCP server implementations with file processing capabilities.

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]