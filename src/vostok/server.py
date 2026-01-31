#!/usr/bin/env python3
"""
ERA5 MCP Server
===============

Model Context Protocol server for ERA5 climate data retrieval.

Usage:
    vostok-mcp                          # If installed as package
    python -m vostok.server         # Direct execution

Configuration via environment variables:
    ARRAYLAKE_API_KEY    - Required for data access
    ERA5_DATA_DIR        - Data storage directory (default: ./data)
    ERA5_MEMORY_DIR      - Memory storage directory (default: ./.memory)
    ERA5_MAX_RETRIES     - Download retry attempts (default: 3)
    ERA5_LOG_LEVEL       - Logging level (default: INFO)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Configure logging
log_level = os.environ.get("ERA5_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import MCP components
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
    )
except ImportError:
    logger.error("MCP library not found. Install with: pip install mcp")
    sys.exit(1)

# Import ERA5 components
from vostok.config import (
    GEOGRAPHIC_REGIONS,
    list_available_variables,
    list_regions,
)
from vostok.memory import get_memory
from vostok.retrieval import retrieve_era5_data

# Import Climate Science tools for full scientific capability
from vostok.tools.climate_science import (
    # Functions
    calculate_climate_diagnostics,
    perform_eof_analysis,
    detect_compound_extremes,
    calculate_trends,
    calculate_correlation,
    detect_percentile_extremes,
    fetch_climate_index,
    calculate_return_periods,
    # Nature-tier additions
    detrend_climate_data,
    perform_composite_analysis,
    analyze_granger_causality,
    # Argument schemas for MCP
    DiagnosticsArgs,
    EOFArgs,
    CompoundExtremeArgs,
    TrendArgs,
    CorrelationArgs,
    PercentileArgs,
    IndexArgs,
    ReturnPeriodArgs,
    DetrendArgs,
    CompositeArgs,
    GrangerArgs,
)

# Create MCP server
server = Server("era5-climate-data")

# Alias for compatibility
app = server


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="retrieve_era5_data",
            description=(
                "Retrieve ERA5 climate reanalysis data from Earthmover's cloud archive.\n\n"
                "QUERY TYPES:\n"
                "- 'temporal': For time series (long time periods, small geographic area)\n"
                "- 'spatial': For spatial maps (large geographic area, short time periods)\n\n"
                "CLIMATOLOGY & RISK ASSESSMENT:\n"
                "- For risk assessment or climatological studies, retrieve at least 20-30 years of data.\n"
                "- Use 'temporal' with focused geographic bounds to manage size for long periods.\n\n"
                "VARIABLES:\n"
                "- sst: Sea Surface Temperature (K)\n"
                "- t2: 2m Air Temperature (K)\n"
                "- u10, v10: 10m Wind Components (m/s)\n"
                "- mslp: Mean Sea Level Pressure (Pa)\n"
                "- tcc: Total Cloud Cover (0-1)\n"
                "- tp: Total Precipitation (m)\n\n"
                "REGIONS (optional, overrides lat/lon):\n"
                "north_atlantic, north_pacific, california_coast, mediterranean,\n"
                "gulf_of_mexico, caribbean, nino34, nino3, nino4, arctic, antarctic\n\n"
                "Returns the file path. Load with: xr.open_dataset('PATH', engine='zarr')"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["temporal", "spatial"],
                        "description": "Use 'temporal' for time series, 'spatial' for maps"
                    },
                    "variable_id": {
                        "type": "string",
                        "description": "ERA5 variable (sst, skt, stl1, swvl1, t2, d2, u10, v10, u100, v100, sp, mslp, blh, tcc, tcw, tcwv, cp, lsp, tp, ssr, ssrd, cape, sd)"
                    },
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "End date (YYYY-MM-DD)"
                    },
                    "min_latitude": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "default": -90.0,
                        "description": "Southern bound (-90 to 90)"
                    },
                    "max_latitude": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "default": 90.0,
                        "description": "Northern bound (-90 to 90)"
                    },
                    "min_longitude": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Western bound (0 to 360, negative allowed)"
                    },
                    "max_longitude": {
                        "type": "number",
                        "default": 359.75,
                        "description": "Eastern bound (0 to 360, negative allowed)"
                    },
                    "region": {
                        "type": "string",
                        "description": "Predefined region name (overrides lat/lon bounds)"
                    }
                },
                "required": ["query_type", "variable_id", "start_date", "end_date"]
            }
        ),
        Tool(
            name="list_era5_variables",
            description=(
                "List all available ERA5 variables with their descriptions, units, "
                "and short names for use with retrieve_era5_data."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_cached_datasets",
            description=(
                "List all ERA5 datasets that have been downloaded and cached locally. "
                "Shows variable, date range, file path, and size."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_regions",
            description=(
                "List all predefined geographic regions that can be used with retrieve_era5_data. "
                "Includes ocean basins, El Niño regions, and coastal areas."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        # ========== CLIMATE SCIENCE TOOLS ==========
        Tool(
            name="compute_climate_diagnostics",
            description=(
                "Calculate anomalies and Z-scores from raw ERA5 data. "
                "ESSENTIAL first step for any scientific analysis. "
                "Z-scores enable detection of statistically significant extremes (Z > 2σ)."
            ),
            inputSchema=DiagnosticsArgs.model_json_schema()
        ),
        Tool(
            name="analyze_climate_modes_eof",
            description=(
                "Perform EOF/PCA analysis to discover dominant spatial patterns. "
                "Reveals climate modes like El Niño, marine heatwave patterns, blocking. "
                "Returns spatial patterns and principal component time series."
            ),
            inputSchema=EOFArgs.model_json_schema()
        ),
        Tool(
            name="detect_compound_extremes",
            description=(
                "Detect 'Ocean Ovens' - compound events where hot SST coincides with stagnant winds. "
                "These compound extremes cause severe marine ecosystem stress. "
                "REQUIRES Z-score data from compute_climate_diagnostics."
            ),
            inputSchema=CompoundExtremeArgs.model_json_schema()
        ),
        Tool(
            name="calculate_climate_trends",
            description=(
                "Calculate linear trends with statistical significance testing. "
                "Essential for climate change attribution and long-term analysis. "
                "Returns trend maps with p-values and significance masking."
            ),
            inputSchema=TrendArgs.model_json_schema()
        ),
        Tool(
            name="calculate_correlation",
            description=(
                "Calculate temporal correlation between two climate variables or indices. "
                "Useful for teleconnection analysis and ocean-atmosphere coupling. "
                "Supports lag analysis for lead-lag relationships."
            ),
            inputSchema=CorrelationArgs.model_json_schema()
        ),
        Tool(
            name="detect_percentile_extremes",
            description=(
                "Detect extreme events using percentile thresholds (e.g., 90th, 95th). "
                "Good for marine heatwave and cold spell identification. "
                "Alternative to Z-score method for extreme detection."
            ),
            inputSchema=PercentileArgs.model_json_schema()
        ),
        Tool(
            name="fetch_climate_index",
            description=(
                "Fetch standard climate indices (Nino3.4, NAO, PDO, AMO, SOI) from NOAA. "
                "ESSENTIAL for attribution - correlate local events with large-scale modes. "
                "Returns monthly time series for correlation analysis."
            ),
            inputSchema=IndexArgs.model_json_schema()
        ),
        Tool(
            name="calculate_return_periods",
            description=(
                "Fit GEV distribution to calculate Return Periods (e.g., '1-in-100 year event'). "
                "CRITICAL for Nature papers - quantifies rarity beyond Z-scores. "
                "Uses Extreme Value Theory. Requires 20+ years of data."
            ),
            inputSchema=ReturnPeriodArgs.model_json_schema()
        ),
        # ========== NATURE-TIER TOOLS (Reviewer #2 Defense) ==========
        Tool(
            name="detrend_climate_data",
            description=(
                "Remove global warming trend to isolate internal climate modes. "
                "MANDATORY before correlation analysis on multi-decadal data. "
                "Reviewer #2 will reject papers that skip this step."
            ),
            inputSchema=DetrendArgs.model_json_schema()
        ),
        Tool(
            name="perform_composite_analysis",
            description=(
                "Create composite maps showing average atmospheric state during events. "
                "Use to discover MECHANISMS - e.g., 'What does pressure look like during heatwaves?' "
                "Essential for explaining WHY extremes occur."
            ),
            inputSchema=CompositeArgs.model_json_schema()
        ),
        Tool(
            name="analyze_granger_causality",
            description=(
                "Perform Granger Causality test to prove X drives Y (not just correlated). "
                "Answers: Does NAO CAUSE SST anomalies, or is it the other way around? "
                "Critical for Nature papers - moves from correlation to causation."
            ),
            inputSchema=GrangerArgs.model_json_schema()
        ),
    ]


# ============================================================================
# TOOL HANDLERS
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""

    try:
        if name == "retrieve_era5_data":
            # Run synchronous function in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retrieve_era5_data(
                    query_type=arguments["query_type"],
                    variable_id=arguments["variable_id"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    min_latitude=arguments.get("min_latitude", -90.0),
                    max_latitude=arguments.get("max_latitude", 90.0),
                    min_longitude=arguments.get("min_longitude", 0.0),
                    max_longitude=arguments.get("max_longitude", 359.75),
                    region=arguments.get("region"),
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_era5_variables":
            result = list_available_variables()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_cached_datasets":
            memory = get_memory()
            result = memory.list_datasets()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_regions":
            result = list_regions()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        # ========== CLIMATE SCIENCE TOOL HANDLERS ==========
        elif name == "compute_climate_diagnostics":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: calculate_climate_diagnostics(
                    dataset_path=arguments["dataset_path"],
                    baseline_start=arguments.get("baseline_start", "1991"),
                    baseline_end=arguments.get("baseline_end", "2020")
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "analyze_climate_modes_eof":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: perform_eof_analysis(
                    dataset_path=arguments["dataset_path"],
                    n_modes=arguments.get("n_modes", 3)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "detect_compound_extremes":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: detect_compound_extremes(
                    sst_path=arguments["sst_path"],
                    wind_path=arguments["wind_path"],
                    heat_threshold=arguments.get("heat_threshold", 1.5),
                    stagnation_threshold=arguments.get("stagnation_threshold", -1.0)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "calculate_climate_trends":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: calculate_trends(
                    dataset_path=arguments["dataset_path"],
                    confidence_level=arguments.get("confidence_level", 0.95)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "calculate_correlation":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: calculate_correlation(
                    dataset_path_1=arguments["dataset_path_1"],
                    dataset_path_2=arguments["dataset_path_2"],
                    lag_hours=arguments.get("lag_hours", 0)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "detect_percentile_extremes":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: detect_percentile_extremes(
                    dataset_path=arguments["dataset_path"],
                    percentile=arguments.get("percentile", 95.0),
                    extreme_type=arguments.get("extreme_type", "above")
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "fetch_climate_index":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_climate_index(
                    index_name=arguments["index_name"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"]
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "calculate_return_periods":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: calculate_return_periods(
                    dataset_path=arguments["dataset_path"],
                    block_size=arguments.get("block_size", "year"),
                    fit_type=arguments.get("fit_type", "maxima")
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        # ========== NATURE-TIER TOOL HANDLERS ==========
        elif name == "detrend_climate_data":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: detrend_climate_data(
                    dataset_path=arguments["dataset_path"],
                    method=arguments.get("method", "linear")
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "perform_composite_analysis":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: perform_composite_analysis(
                    target_path=arguments["target_path"],
                    index_path=arguments["index_path"],
                    threshold=arguments.get("threshold", 1.0)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "analyze_granger_causality":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analyze_granger_causality(
                    dataset_x=arguments["dataset_x"],
                    dataset_y=arguments["dataset_y"],
                    max_lag=arguments.get("max_lag", 5)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True
            )

    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


# ============================================================================
# SERVER STARTUP
# ============================================================================

async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    logger.info("Starting ERA5 MCP Server...")

    # Check for API key
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        logger.warning(
            "ARRAYLAKE_API_KEY not set. Data retrieval will fail. "
            "Set it via environment variable or .env file."
        )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
