"""
WebSocket Chat Handler
======================
Handles real-time chat via WebSocket with streaming responses.
"""

import json
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for chat."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "").strip()

            if not message:
                continue

            logger.info(f"Received: {message[:100]}...")

            # Send thinking indicator
            await manager.send_json(websocket, {"type": "thinking"})

            try:
                from web.agent_wrapper import get_agent_session

                session = get_agent_session()

                # Callback for streaming
                async def stream_callback(event_type: str, content: str, **kwargs):
                    msg = {"type": event_type, "content": content}
                    msg.update(kwargs)
                    await manager.send_json(websocket, msg)

                # Process message
                response = await session.process_message(message, stream_callback)

                # Send complete
                await manager.send_json(websocket, {
                    "type": "complete",
                    "content": response
                })

            except Exception as e:
                logger.exception(f"Error: {e}")
                await manager.send_json(websocket, {
                    "type": "error",
                    "content": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        manager.disconnect(websocket)
