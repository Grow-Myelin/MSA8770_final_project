"""
main.py - FastAPI application for MSA8770 Legal Demo Dashboard

This module provides the web interface for the multi-agent legal analysis system.
"""

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api import router as api_router
from app.logger import setup_logging, get_logger, generate_request_id, set_request_id

# Initialize logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)

logger = get_logger("app.main")

# Initialize FastAPI app
app = FastAPI(
    title="MSA8770 Legal Demo",
    description="Multi-Agent Legal Case Analysis Dashboard",
    version="1.0.0",
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all incoming requests with timing."""
    request_id = generate_request_id()
    set_request_id(request_id)

    # Add request_id to request state for access in routes
    request.state.request_id = request_id

    start_time = time.time()
    method = request.method
    path = request.url.path
    query = str(request.url.query) if request.url.query else ""

    logger.info(
        f"Request: {method} {path}",
        extra={"extra_data": {"query": query, "client": request.client.host if request.client else "unknown"}}
    )

    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Response: {response.status_code} ({duration_ms}ms)",
            extra={"extra_data": {"status": response.status_code, "duration_ms": duration_ms}}
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Request failed: {str(e)}",
            extra={"extra_data": {"duration_ms": duration_ms}},
            exc_info=True
        )
        raise

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "app" / "static"),
    name="static",
)

# Setup templates
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info("Application starting up", extra={"extra_data": {"log_level": LOG_LEVEL}})


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("Application shutting down")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "legal-demo"}
