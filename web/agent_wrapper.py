"""
Agent Wrapper for Web Interface
===============================
Wraps the LangChain agent for WebSocket streaming.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict
from queue import Queue

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from config import CONFIG, AGENT_SYSTEM_PROMPT
from memory import get_memory
from era5_tool import era5_tool
from repl_tool import SuperbPythonREPLTool

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Manages a single agent session with streaming support.
    """

    def __init__(self):
        self._agent = None
        self._repl_tool: Optional[SuperbPythonREPLTool] = None
        self._messages: List[Dict] = []
        self._initialized = False

        # Queue for captured plots (thread-safe)
        self._plot_queue: Queue = Queue()

        self._initialize()

    def _initialize(self):
        """Initialize the agent and tools."""
        logger.info("Initializing agent session...")

        if not os.environ.get("ARRAYLAKE_API_KEY"):
            logger.warning("ARRAYLAKE_API_KEY not found")

        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found")
            return

        try:
            # Initialize tools
            logger.info("Starting Python kernel...")
            self._repl_tool = SuperbPythonREPLTool(working_dir=os.getcwd())

            # Set up plot callback using the proper method
            def on_plot_captured(base64_data: str, filepath: str):
                logger.info(f"Plot captured, adding to queue: {filepath}")
                self._plot_queue.put((base64_data, filepath))

            self._repl_tool._executor.set_plot_callback(on_plot_captured)
            logger.info("Plot callback registered")

            tools = [era5_tool, self._repl_tool]

            # Initialize LLM
            logger.info("Connecting to LLM...")
            llm = ChatOpenAI(
                model=CONFIG.model_name,
                temperature=CONFIG.temperature
            )

            # Load memory context
            memory = get_memory()
            context_summary = memory.get_context_summary()
            enhanced_prompt = AGENT_SYSTEM_PROMPT

            if context_summary and context_summary != "No context available.":
                enhanced_prompt += f"\n\n## CURRENT CONTEXT\n{context_summary}"

            # Create agent
            logger.info("Creating agent...")
            self._agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=enhanced_prompt,
                debug=False
            )

            # Load recent messages from memory
            recent_messages = memory.get_langchain_messages(n_messages=10)
            self._messages = recent_messages.copy()

            self._initialized = True
            logger.info("Agent session initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize agent: {e}")
            self._initialized = False

    def is_ready(self) -> bool:
        """Check if the agent is ready."""
        return self._initialized and self._agent is not None

    def clear_messages(self):
        """Clear conversation messages."""
        self._messages = []

    def get_pending_plots(self) -> List[tuple]:
        """Get all pending plots from queue."""
        plots = []
        while not self._plot_queue.empty():
            try:
                plots.append(self._plot_queue.get_nowait())
            except:
                break
        return plots

    async def process_message(
        self,
        user_message: str,
        stream_callback: Callable
    ) -> str:
        """
        Process a user message and stream the response.
        """
        if not self.is_ready():
            raise RuntimeError("Agent not initialized")

        # Clear any old plots from queue
        self.get_pending_plots()

        # Add user message to history
        memory = get_memory()
        memory.add_message("user", user_message)
        self._messages.append({"role": "user", "content": user_message})

        try:
            # Invoke the agent in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._agent.invoke({"messages": self._messages})
            )

            # Update messages
            self._messages = result["messages"]

            # Extract response
            last_message = self._messages[-1]

            if hasattr(last_message, 'content') and last_message.content:
                response_text = last_message.content
            elif isinstance(last_message, dict) and last_message.get('content'):
                response_text = last_message['content']
            else:
                response_text = str(last_message)

            # Stream the response in chunks
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                await stream_callback("chunk", chunk)
                await asyncio.sleep(0.01)

            # Send any captured plots
            plots = self.get_pending_plots()
            logger.info(f"Sending {len(plots)} plots to client")
            for base64_data, filepath in plots:
                await stream_callback("plot", "", data=base64_data, path=filepath)

            # Save to memory
            memory.add_message("assistant", response_text)

            return response_text

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            raise

    def close(self):
        """Clean up resources."""
        logger.info("Closing agent session...")
        if self._repl_tool:
            try:
                self._repl_tool.close()
            except Exception as e:
                logger.error(f"Error closing REPL: {e}")


# Global session instance
_session: Optional[AgentSession] = None


def get_agent_session() -> AgentSession:
    """Get or create the global agent session."""
    global _session
    if _session is None:
        _session = AgentSession()
    return _session


def shutdown_agent_session():
    """Shutdown the global agent session."""
    global _session
    if _session:
        _session.close()
        _session = None
