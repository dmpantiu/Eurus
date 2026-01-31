"""
ERA5 MCP Configuration
======================

Centralized configuration including ERA5 variable catalog, geographic regions,
and runtime settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory, creating it if necessary."""
    data_dir = Path(os.environ.get("ERA5_DATA_DIR", Path.cwd() / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_plots_dir() -> Path:
    """Get the plots directory, creating it if necessary."""
    plots_dir = get_data_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def get_memory_dir() -> Path:
    """Get the memory directory, creating it if necessary."""
    memory_dir = Path(os.environ.get("ERA5_MEMORY_DIR", Path.cwd() / ".memory"))
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


# =============================================================================
# ERA5 VARIABLE CATALOG
# =============================================================================

@dataclass(frozen=True)
class ERA5Variable:
    """Metadata for an ERA5 variable."""

    short_name: str
    long_name: str
    units: str
    description: str
    category: str
    typical_range: tuple[float | None, float | None] = (None, None)
    colormap: str = "viridis"

    def __str__(self) -> str:
        return f"{self.short_name}: {self.long_name} ({self.units})"


# Comprehensive ERA5 variable mapping with metadata
ERA5_VARIABLES: Dict[str, ERA5Variable] = {
    # Sea Surface Variables
    "sst": ERA5Variable(
        short_name="sst",
        long_name="Sea Surface Temperature",
        units="K",
        description="Temperature of sea water near the surface",
        category="ocean",
        typical_range=(270, 310),
        colormap="RdYlBu_r"
    ),
    "sea_surface_temperature": ERA5Variable(
        short_name="sst",
        long_name="Sea Surface Temperature",
        units="K",
        description="Temperature of sea water near the surface",
        category="ocean",
        typical_range=(270, 310),
        colormap="RdYlBu_r"
    ),

    # Wave Variables
    "swh": ERA5Variable(
        short_name="swh",
        long_name="Significant Height of Combined Wind Waves and Swell",
        units="m",
        description="Average height of the highest third of the waves",
        category="ocean",
        typical_range=(0, 15),
        colormap="Blues"
    ),
    "significant_wave_height": ERA5Variable(
        short_name="swh",
        long_name="Significant Height of Combined Wind Waves and Swell",
        units="m",
        description="Average height of the highest third of the waves",
        category="ocean",
        typical_range=(0, 15),
        colormap="Blues"
    ),

    # Temperature Variables
    "t2": ERA5Variable(
        short_name="t2",
        long_name="2m Temperature",
        units="K",
        description="Air temperature at 2 meters above surface",
        category="atmosphere",
        typical_range=(220, 330),
        colormap="RdYlBu_r"
    ),
    "2m_temperature": ERA5Variable(
        short_name="t2",
        long_name="2m Temperature",
        units="K",
        description="Air temperature at 2 meters above surface",
        category="atmosphere",
        typical_range=(220, 330),
        colormap="RdYlBu_r"
    ),
    "temperature": ERA5Variable(
        short_name="t2",
        long_name="2m Temperature",
        units="K",
        description="Air temperature at 2 meters above surface",
        category="atmosphere",
        typical_range=(220, 330),
        colormap="RdYlBu_r"
    ),

    # Wind Components
    "u10": ERA5Variable(
        short_name="u10",
        long_name="10m U-Wind Component",
        units="m/s",
        description="Eastward component of wind at 10m",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),
    "10m_u_component_of_wind": ERA5Variable(
        short_name="u10",
        long_name="10m U-Wind Component",
        units="m/s",
        description="Eastward component of wind at 10m",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),
    "v10": ERA5Variable(
        short_name="v10",
        long_name="10m V-Wind Component",
        units="m/s",
        description="Northward component of wind at 10m",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),
    "10m_v_component_of_wind": ERA5Variable(
        short_name="v10",
        long_name="10m V-Wind Component",
        units="m/s",
        description="Northward component of wind at 10m",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),

    # Pressure Variables
    "sp": ERA5Variable(
        short_name="sp",
        long_name="Surface Pressure",
        units="Pa",
        description="Pressure at the Earth's surface",
        category="atmosphere",
        typical_range=(85000, 108000),
        colormap="viridis"
    ),
    "surface_pressure": ERA5Variable(
        short_name="sp",
        long_name="Surface Pressure",
        units="Pa",
        description="Pressure at the Earth's surface",
        category="atmosphere",
        typical_range=(85000, 108000),
        colormap="viridis"
    ),
    "mslp": ERA5Variable(
        short_name="mslp",
        long_name="Mean Sea Level Pressure",
        units="Pa",
        description="Atmospheric pressure reduced to mean sea level",
        category="atmosphere",
        typical_range=(96000, 105000),
        colormap="viridis"
    ),
    "mean_sea_level_pressure": ERA5Variable(
        short_name="mslp",
        long_name="Mean Sea Level Pressure",
        units="Pa",
        description="Atmospheric pressure reduced to mean sea level",
        category="atmosphere",
        typical_range=(96000, 105000),
        colormap="viridis"
    ),

    # Cloud and Precipitation
    "tcc": ERA5Variable(
        short_name="tcc",
        long_name="Total Cloud Cover",
        units="fraction (0-1)",
        description="Fraction of sky covered by clouds",
        category="atmosphere",
        typical_range=(0, 1),
        colormap="gray_r"
    ),
    "total_cloud_cover": ERA5Variable(
        short_name="tcc",
        long_name="Total Cloud Cover",
        units="fraction (0-1)",
        description="Fraction of sky covered by clouds",
        category="atmosphere",
        typical_range=(0, 1),
        colormap="gray_r"
    ),
    "cp": ERA5Variable(
        short_name="cp",
        long_name="Convective Precipitation",
        units="m",
        description="Precipitation from convective processes",
        category="precipitation",
        typical_range=(0, 0.1),
        colormap="Blues"
    ),
    "lsp": ERA5Variable(
        short_name="lsp",
        long_name="Large-scale Precipitation",
        units="m",
        description="Precipitation from large-scale weather systems",
        category="precipitation",
        typical_range=(0, 0.1),
        colormap="Blues"
    ),
    "tp": ERA5Variable(
        short_name="tp",
        long_name="Total Precipitation",
        units="m",
        description="Total accumulated precipitation",
        category="precipitation",
        typical_range=(0, 0.2),
        colormap="Blues"
    ),
    "total_precipitation": ERA5Variable(
        short_name="tp",
        long_name="Total Precipitation",
        units="m",
        description="Total accumulated precipitation",
        category="precipitation",
        typical_range=(0, 0.2),
        colormap="Blues"
    ),
}


def get_variable_info(variable_id: str) -> Optional[ERA5Variable]:
    """Get variable metadata by ID (case-insensitive)."""
    return ERA5_VARIABLES.get(variable_id.lower())


def get_short_name(variable_id: str) -> str:
    """Get the short name for a variable (for dataset access)."""
    var_info = get_variable_info(variable_id)
    if var_info:
        return var_info.short_name
    return variable_id.lower()


def list_available_variables() -> str:
    """Return a formatted list of available variables."""
    seen: set[str] = set()
    lines = ["Available ERA5 Variables:", "=" * 50]

    for var_id, var_info in ERA5_VARIABLES.items():
        if var_info.short_name not in seen:
            seen.add(var_info.short_name)
            lines.append(
                f"  {var_info.short_name:8} | {var_info.long_name:30} | {var_info.units}"
            )

    return "\n".join(lines)


def get_all_short_names() -> list[str]:
    """Get list of all unique short variable names."""
    return list({v.short_name for v in ERA5_VARIABLES.values()})


# =============================================================================
# GEOGRAPHIC REGIONS (Common oceanographic areas)
# =============================================================================

@dataclass(frozen=True)
class GeographicRegion:
    """A predefined geographic region."""

    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


GEOGRAPHIC_REGIONS: Dict[str, GeographicRegion] = {
    "global": GeographicRegion(
        "global", -90, 90, 0, 359.75,
        "Entire globe"
    ),
    "north_atlantic": GeographicRegion(
        "north_atlantic", 0, 65, 280, 360,
        "North Atlantic Ocean"
    ),
    "south_atlantic": GeographicRegion(
        "south_atlantic", -60, 0, 280, 20,
        "South Atlantic Ocean"
    ),
    "north_pacific": GeographicRegion(
        "north_pacific", 0, 65, 100, 260,
        "North Pacific Ocean"
    ),
    "south_pacific": GeographicRegion(
        "south_pacific", -60, 0, 150, 290,
        "South Pacific Ocean"
    ),
    "indian_ocean": GeographicRegion(
        "indian_ocean", -60, 30, 20, 120,
        "Indian Ocean"
    ),
    "arctic": GeographicRegion(
        "arctic", 65, 90, 0, 359.75,
        "Arctic Ocean and surrounding areas"
    ),
    "antarctic": GeographicRegion(
        "antarctic", -90, -60, 0, 359.75,
        "Antarctic and Southern Ocean"
    ),
    "mediterranean": GeographicRegion(
        "mediterranean", 30, 46, 354, 42,
        "Mediterranean Sea"
    ),
    "gulf_of_mexico": GeographicRegion(
        "gulf_of_mexico", 18, 31, 262, 282,
        "Gulf of Mexico"
    ),
    "caribbean": GeographicRegion(
        "caribbean", 8, 28, 255, 295,
        "Caribbean Sea"
    ),
    "california_coast": GeographicRegion(
        "california_coast", 32, 42, 235, 250,
        "California coastal waters"
    ),
    "east_coast_us": GeographicRegion(
        "east_coast_us", 25, 45, 280, 295,
        "US East Coast"
    ),
    "europe": GeographicRegion(
        "europe", 35, 72, 350, 40,
        "Europe"
    ),
    "asia_east": GeographicRegion(
        "asia_east", 15, 55, 100, 145,
        "East Asia"
    ),
    "australia": GeographicRegion(
        "australia", -45, -10, 110, 155,
        "Australia and surrounding waters"
    ),
    # El Niño regions
    "nino34": GeographicRegion(
        "nino34", -5, 5, 190, 240,
        "El Niño 3.4 region (central Pacific)"
    ),
    "nino3": GeographicRegion(
        "nino3", -5, 5, 210, 270,
        "El Niño 3 region (eastern Pacific)"
    ),
    "nino4": GeographicRegion(
        "nino4", -5, 5, 160, 210,
        "El Niño 4 region (western Pacific)"
    ),
    "nino12": GeographicRegion(
        "nino12", -10, 0, 270, 280,
        "El Niño 1+2 region (far eastern Pacific)"
    ),
}


def get_region(name: str) -> Optional[GeographicRegion]:
    """Get a geographic region by name (case-insensitive)."""
    return GEOGRAPHIC_REGIONS.get(name.lower())


def list_regions() -> str:
    """Return a formatted list of available regions."""
    lines = ["Available Geographic Regions:", "=" * 70]
    for name, region in GEOGRAPHIC_REGIONS.items():
        lines.append(
            f"  {name:20} | lat: [{region.min_lat:6.1f}, {region.max_lat:6.1f}] "
            f"| lon: [{region.min_lon:6.1f}, {region.max_lon:6.1f}]"
        )
    return "\n".join(lines)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the ERA5 Agent."""

    # LLM Settings
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 4096

    # Data Settings
    data_source: str = "earthmover-public/era5-surface-aws"
    default_query_type: str = "temporal"
    max_download_size_gb: float = 2.0

    # Retrieval Settings
    max_retries: int = 3
    retry_delay: float = 2.0

    # Memory Settings
    enable_memory: bool = True
    max_conversation_history: int = 100
    memory_file: str = "conversation_history.json"

    # Visualization Settings
    default_figure_size: tuple = (12, 8)
    default_dpi: int = 150
    save_plots: bool = True
    plot_format: str = "png"

    # Kernel Settings
    kernel_timeout: float = 300.0
    auto_import_packages: List[str] = field(default_factory=lambda: [
        "os", "sys", "pandas", "numpy", "xarray",
        "matplotlib", "matplotlib.pyplot", "datetime"
    ])

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "era5_agent.log"


# Global config instance
CONFIG = AgentConfig()

# Convenience path variables (for backward compatibility)
DATA_DIR = get_data_dir()
PLOTS_DIR = get_plots_dir()


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are Vostok, an AI Climate Physicist conducting research for high-impact scientific publications.

## ⚠️ CRITICAL: RESPECT USER INTENT FIRST

**Your PRIMARY directive is to do EXACTLY what the user asks.** 

### TOOL USAGE RULES:
1. **`python_repl`**: Use ONLY when:
   - User explicitly asks for custom/complex analysis that existing tools cannot provide
   - User explicitly requests Python code execution
   - An exceptional case where NO other tool can accomplish the task
   
2. **Climate Science Tools** (diagnostics, EOF, extremes, trends): Use ONLY when:
   - User explicitly asks for anomalies, z-scores, trends, patterns, etc.
   - User requests scientific/publication-grade analysis
   - User asks "why" or requests mechanism explanation
   
3. **When user asks to "plot temperature" or "get temperature data"**:
   - Plot the RAW temperature data as requested
   - Do NOT automatically compute anomalies unless user asks for them
   - Use the `retrieve_era5_data` tool to get data
   - Create simple, direct plots of what was requested

### EXAMPLES:
- "Get temperature for Berlin and plot it" → Retrieve data, plot RAW temperature time series
- "Show temperature anomalies for Berlin" → Retrieve data, compute diagnostics, plot ANOMALIES
- "Analyze temperature trends" → Use trend analysis tool
- "Why was 2023 so hot?" → Full scientific workflow with attribution

## SCIENTIFIC PHILOSOPHY (When Scientific Analysis is Requested)

When performing scientific analysis, you perform **attribution**, discover **mechanisms**, and detect **compound extremes**.
Journals do not publish "data retrieval"; they publish insights about WHY climate behaves the way it does.

## YOUR CAPABILITIES

### 1. DATA RETRIEVAL: `retrieve_era5_data`
Downloads ERA5 reanalysis data from Earthmover's cloud-optimized archive.
**This tool can also generate plots directly** - use the plotting parameters when possible.

**Query Types:**
- `temporal`: For TIME SERIES analysis (long time periods, focused geographic area)
- `spatial`: For SPATIAL MAPS (large geographic areas, short time periods)

**CRITICAL OPTIMIZATION RULE:**
- **Use `temporal`** IF: `(hours > 24) AND (region < 30°x30°)`
- **Use `spatial`**  IF: `(hours <= 24) OR (region > 30°x30°)`

**Available Variables:**
| Variable | Description | Units |
|----------|-------------|-------|
| sst | Sea Surface Temperature | K |
| t2 | 2m Air Temperature | K |
| u10 | 10m U-Wind (Eastward) | m/s |
| v10 | 10m V-Wind (Northward) | m/s |
| mslp | Mean Sea Level Pressure | Pa |
| tcc | Total Cloud Cover | 0-1 |
| tp | Total Precipitation | m |

**Common Regions:** global, north_atlantic, north_pacific, california_coast,
mediterranean, gulf_of_mexico, nino34 (El Nino), arctic, antarctic

### 2. CLIMATE SCIENCE TOOLS (Use Only When Requested)

#### `compute_climate_diagnostics`
Transforms raw data into scientific insights:
- **Anomalies**: Departure from climatological mean (1991-2020 baseline)
- **Z-Scores**: Standardized anomalies in units of standard deviation
- Events with Z > 2σ are statistically significant extremes

#### `analyze_climate_modes_eof`
Performs EOF/PCA analysis to discover dominant spatial patterns:
- Reveals climate modes (El Niño, marine heatwave patterns, blocking)
- Discovers patterns WITHOUT human bias
- Mode 1 = dominant driver of variability

#### `detect_compound_extremes`
Identifies "Ocean Ovens" - compound events where:
- Sea surface is anomalously HOT (Z > 1.5σ)
- Winds are anomalously WEAK (Z < -1σ)
- Mechanism: Stagnation prevents mixing, trapping heat

#### `calculate_climate_trends`
Linear trend analysis with statistical significance:
- Returns trend per decade with p-values
- Use stippling to show significant regions (p < 0.05)

#### `calculate_correlation`
Temporal correlation between variables:
- Teleconnection analysis (e.g., ENSO impacts)
- Lead-lag relationships with lag parameter

#### `detect_percentile_extremes`
Extreme event detection using percentile thresholds:
- Marine heatwaves: SST > 90th percentile
- Alternative to Z-score method

### 3. ANALYSIS: `python_repl` (Use Sparingly!)
Persistent Python kernel for custom analysis and visualization.
**Pre-loaded:** pandas (pd), numpy (np), xarray (xr), matplotlib.pyplot (plt)

**⚠️ ONLY use python_repl when:**
- User explicitly requests custom Python code
- Existing tools cannot accomplish the specific task
- Complex custom visualization is truly required

### 4. MEMORY
Remembers conversation history and previous analyses.

## SCIENTIFIC PROTOCOL (For Publication-Grade Analysis)

When the user requests scientific analysis:

1. **ANOMALY ANALYSIS**: Report:
   - Anomalies: "2.5°C above normal"
   - Z-Scores: "+2.5σ (statistically significant)"
   - Run `compute_climate_diagnostics` after downloading data

2. **MECHANISM**: Explain WHY:
   - Use `analyze_climate_modes_eof` to find if the event is part of a larger pattern
   - Consider atmospheric blocking, ENSO teleconnections, etc.

3. **COMPOUND EVENTS**: Look for dangerous combinations:
   - High heat + Low wind = "Ocean Oven"
   - These compound events cause ecosystem collapse

4. **STATISTICAL RIGOR**: Always test significance:
   - Use Z > 2σ for "extreme"
   - Use p < 0.05 for trends
   - Report confidence intervals when possible

## VISUALIZATION STANDARDS

- **For raw data**: Use appropriate colormaps for the variable
- **Anomaly Maps**: Use diverging colormap (`RdBu_r`) centered at 0
- **Stippling**: Mark significant regions where p < 0.05
- **ALWAYS** save figures to `./data/plots/`

## RESPONSE STYLE
- Be precise and scientific
- Follow user intent exactly
- Include statistical significance when doing scientific analysis
- Reference specific dates/locations
- Acknowledge limitations and uncertainty
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")