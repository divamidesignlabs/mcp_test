
from md_converter import mcp as md_converter_tool
import contextlib
from fastapi import FastAPI


# Create a combined lifespan to manage both session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(md_converter_tool.session_manager.run())
        # await stack.enter_async_context(math_mcp.session_manager.run())
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/convert", md_converter_tool.streamable_http_app())
# app.mount("/math", math_mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
