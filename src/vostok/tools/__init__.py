"""
Vostok Tools Registry
=====================
Central hub for all agent skills (tools).

Tools are organized by category:
- Data Retrieval: ERA5 data access
- Analysis: Python REPL for custom analysis
- Climate Science: Statistical methods for attribution & discovery
- Routing: Maritime navigation (optional)
"""

from typing import List
from langchain_core.tools import BaseTool

# Import existing core tools
from .era5 import era5_tool
from .repl import SuperbPythonREPLTool
from .routing import routing_tool

# Import climate science tools (the "Physics Brain")
from .climate_science import (
    diagnostics_tool,      # Z-scores & anomalies
    detrend_tool,          # Remove warming trend
    eof_tool,              # Pattern discovery (EOF/PCA)
    compound_tool,         # Ocean Oven detection
    trend_tool,            # Trend analysis with significance
    correlation_tool,      # Teleconnection analysis
    percentile_tool,       # Extreme event detection
    index_tool,            # Climate indices (ENSO, NAO, etc.)
    return_period_tool,    # Extreme Value Theory (GEV)
    composite_tool,        # Composite maps for mechanism discovery
    granger_tool,          # Granger causality (X -> Y)
    SCIENCE_TOOLS          # All science tools as a list
)

# Optional dependency check for routing
try:
    import scgraph
    HAS_ROUTING_DEPS = True
except ImportError:
    HAS_ROUTING_DEPS = False


def get_all_tools(
    enable_routing: bool = False,
    enable_science: bool = True
) -> List[BaseTool]:
    """
    Return a list of all available tools for the agent.

    Args:
        enable_routing: If True, includes the maritime routing and risk assessment tools.
        enable_science: If True, includes climate science statistical tools (default: True).

    Returns:
        List of LangChain tools for the agent.
    """
    # Core tools: data retrieval + Python analysis
    tools = [
        era5_tool,
        SuperbPythonREPLTool(working_dir=".")
    ]

    # Science tools: statistical analysis for publication-grade research
    if enable_science:
        tools.extend(SCIENCE_TOOLS)

    # Routing tools: maritime navigation (optional)
    if enable_routing:
        if HAS_ROUTING_DEPS:
            tools.append(routing_tool)
        else:
            print("WARNING: Routing tools requested but dependencies (scgraph, global-land-mask) are missing.")

    return tools


def get_science_tools() -> List[BaseTool]:
    """Return only the climate science analysis tools."""
    return SCIENCE_TOOLS.copy()


# Alias for backward compatibility
get_tools = get_all_tools