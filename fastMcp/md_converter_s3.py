# mcp_server.py
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from fastmcp import FastMCP
from markitdown import MarkItDown

# Load environment variables
load_dotenv()

mcp = FastMCP("File Converter MCP Server", host="127.0.0.1", port=8001)

# S3 Configuration
def get_s3_client():
    """Initialize and return S3 client with error handling"""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
    except NoCredentialsError:
        \
        raise Exception("AWS credentials not found. Please check your .env file.")

def upload_to_s3(file_path: str, s3_key: str) -> str:
    """Upload file to S3 and return the S3 URL"""
    s3_client = get_s3_client()
    bucket = os.getenv('S3_BUCKET')
    if not bucket:
        raise Exception("S3_BUCKET not configured in .env file")
    
    s3_prefix = os.getenv('S3_PREFIX', '')
    if s3_prefix:
        s3_key = f"{s3_prefix}/{s3_key}"
    
    try:
        s3_client.upload_file(file_path, bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"
    except ClientError as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")

def process_folder(folder_path: str, temp_dir: str) -> list:
    """Process all files in a folder and return list of processed files"""
    processed_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise Exception(f"Folder does not exist: {folder_path}")
    
    for file_path in folder_path.rglob('*'):
        if file_path.is_file():
            # Copy file to temp directory maintaining structure
            relative_path = file_path.relative_to(folder_path)
            temp_file_path = Path(temp_dir) / relative_path
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, temp_file_path)
            processed_files.append(str(temp_file_path))
    
    return processed_files

@mcp.tool
def convert(file_path: str) -> str:
    """Converts a file to markdown format."""
    md = MarkItDown(enable_plugins=False)
    result = md.convert(file_path)
    output_file = f"{file_path.split('.')[0]}.md"
    with open(output_file, "w") as f:
        f.write(result.text_content)
    return f"Markdown output saved to {output_file}"

@mcp.tool
def upload_and_convert(
    input_path: str, 
    output_filename: Optional[str] = None
) -> str:
    """
    Upload a file or folder to S3, convert all files to markdown using MarkItDown,
    save markdown files locally and upload them to S3.
    
    Args:
        input_path: Path to the file or folder to process
        output_filename: Optional custom name for the output file (without extension)
    
    Returns:
        Status message with details of processed files and S3 URLs
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            return f"Error: Path does not exist: {input_path}"
        
        results = []
        md = MarkItDown(enable_plugins=False)
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            files_to_process = []
            
            if input_path.is_file():
                # Single file processing
                temp_file = Path(temp_dir) / input_path.name
                shutil.copy2(input_path, temp_file)
                files_to_process.append(str(temp_file))
            else:
                # Folder processing
                files_to_process = process_folder(str(input_path), temp_dir)
            
            for file_path in files_to_process:
                file_path_obj = Path(file_path)
                relative_path = file_path_obj.relative_to(temp_dir)
                
                try:
                    # Upload original file to S3
                    original_s3_key = f"uploads/originals/{relative_path}"
                    original_s3_url = upload_to_s3(file_path, original_s3_key)
                    
                    # Convert to markdown
                    result = md.convert(file_path)
                    
                    # Determine output filename
                    if output_filename and len(files_to_process) == 1:
                        md_filename = f"{output_filename}.md"
                    else:
                        md_filename = f"{file_path_obj.stem}.md"
                    
                    # Save markdown locally
                    if len(files_to_process) == 1:
                        local_md_path = Path(md_filename)
                    else:
                        # Maintain folder structure for multiple files
                        local_md_dir = Path("converted_markdown") / relative_path.parent
                        local_md_dir.mkdir(parents=True, exist_ok=True)
                        local_md_path = local_md_dir / md_filename
                    
                    with open(local_md_path, "w", encoding="utf-8") as f:
                        f.write(result.text_content)
                    
                    # Upload markdown to S3
                    md_s3_key = f"uploads/markdown/{relative_path.parent}/{md_filename}" if relative_path.parent != Path('.') else f"uploads/markdown/{md_filename}"
                    md_s3_url = upload_to_s3(str(local_md_path), md_s3_key)
                    
                    results.append({
                        "original_file": str(relative_path),
                        "original_s3_url": original_s3_url,
                        "markdown_file": str(local_md_path),
                        "markdown_s3_url": md_s3_url,
                        "status": "success"
                    })
                    
                except Exception as e:
                    results.append({
                        "original_file": str(relative_path),
                        "status": "error",
                        "error": str(e)
                    })
        
        # Format response
        if not results:
            return "No files were processed."
        
        response_lines = ["File processing completed:"]
        success_count = 0
        error_count = 0
        
        for result in results:
            if result["status"] == "success":
                success_count += 1
                response_lines.extend([
                    f"\n✅ {result['original_file']}:",
                    f"   📁 Original: {result['original_s3_url']}",
                    f"   📝 Markdown: {result['markdown_file']} -> {result['markdown_s3_url']}"
                ])
            else:
                error_count += 1
                response_lines.extend([
                    f"\n❌ {result['original_file']}: {result['error']}"
                ])
        
        response_lines.insert(1, f"\nSummary: {success_count} successful, {error_count} errors")
        return "\n".join(response_lines)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
