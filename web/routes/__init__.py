"""Web routes package."""

from .api import router as api_router
from .websocket import router as websocket_router
from .pages import router as pages_router

__all__ = ["api_router", "websocket_router", "pages_router"]
