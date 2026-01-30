"""
ERA5 Agent Configuration Module
================================
Centralized configuration for all agent components.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
MEMORY_DIR = BASE_DIR / ".memory"
PLOTS_DIR = DATA_DIR / "plots"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, MEMORY_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ERA5 VARIABLE CATALOG
# ============================================================================

@dataclass
class ERA5Variable:
    """Metadata for an ERA5 variable."""
    short_name: str
    long_name: str
    units: str
    description: str
    category: str
    typical_range: tuple = field(default=(None, None))
    colormap: str = "viridis"


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
    "u_component_of_wind": ERA5Variable(
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
    "v_component_of_wind": ERA5Variable(
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
    "convective_precipitation": ERA5Variable(
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
    "large_scale_precipitation": ERA5Variable(
        short_name="lsp",
        long_name="Large-scale Precipitation",
        units="m",
        description="Precipitation from large-scale weather systems",
        category="precipitation",
        typical_range=(0, 0.1),
        colormap="Blues"
    ),

    # Additional commonly used variables
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
    seen = set()
    lines = ["Available ERA5 Variables:", "=" * 50]

    for var_id, var_info in ERA5_VARIABLES.items():
        if var_info.short_name not in seen:
            seen.add(var_info.short_name)
            lines.append(f"  {var_info.short_name:8} | {var_info.long_name:30} | {var_info.units}")

    return "\n".join(lines)


# ============================================================================
# GEOGRAPHIC REGIONS (Common oceanographic areas)
# ============================================================================

GEOGRAPHIC_REGIONS = {
    "global": {"min_lat": -90, "max_lat": 90, "min_lon": 0, "max_lon": 359.75},
    "north_atlantic": {"min_lat": 0, "max_lat": 65, "min_lon": 280, "max_lon": 360},
    "south_atlantic": {"min_lat": -60, "max_lat": 0, "min_lon": 280, "max_lon": 20},
    "north_pacific": {"min_lat": 0, "max_lat": 65, "min_lon": 100, "max_lon": 260},
    "south_pacific": {"min_lat": -60, "max_lat": 0, "min_lon": 150, "max_lon": 290},
    "indian_ocean": {"min_lat": -60, "max_lat": 30, "min_lon": 20, "max_lon": 120},
    "arctic": {"min_lat": 65, "max_lat": 90, "min_lon": 0, "max_lon": 359.75},
    "antarctic": {"min_lat": -90, "max_lat": -60, "min_lon": 0, "max_lon": 359.75},
    "mediterranean": {"min_lat": 30, "max_lat": 46, "min_lon": 354, "max_lon": 42},
    "gulf_of_mexico": {"min_lat": 18, "max_lat": 31, "min_lon": 262, "max_lon": 282},
    "caribbean": {"min_lat": 8, "max_lat": 28, "min_lon": 255, "max_lon": 295},
    "california_coast": {"min_lat": 32, "max_lat": 42, "min_lon": 235, "max_lon": 250},
    "east_coast_us": {"min_lat": 25, "max_lat": 45, "min_lon": 280, "max_lon": 295},
    "europe": {"min_lat": 35, "max_lat": 72, "min_lon": 350, "max_lon": 40},
    "asia_east": {"min_lat": 15, "max_lat": 55, "min_lon": 100, "max_lon": 145},
    "australia": {"min_lat": -45, "max_lat": -10, "min_lon": 110, "max_lon": 155},
    "nino34": {"min_lat": -5, "max_lat": 5, "min_lon": 190, "max_lon": 240},  # El Nino region
    "nino3": {"min_lat": -5, "max_lat": 5, "min_lon": 210, "max_lon": 270},
    "nino4": {"min_lat": -5, "max_lat": 5, "min_lon": 160, "max_lon": 210},
}


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

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

    # Web Interface Settings
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "era5_agent.log"


# Global config instance
CONFIG = AgentConfig()


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are Vostok, an expert Oceanographer & Climate Data Analyst.

## YOUR CAPABILITIES

### 1. DATA RETRIEVAL: `retrieve_era5_data`
Downloads ERA5 reanalysis data from Earthmover's cloud-optimized archive.

**Query Types:**
- `temporal`: For TIME SERIES analysis (long time periods, focused geographic area)
- `spatial`: For SPATIAL MAPS (large geographic areas, short time periods)

**Available Variables:**
| Variable | Description | Units |
|----------|-------------|-------|
| sst | Sea Surface Temperature | K |
| t2 | 2m Air Temperature | K |
| u10 | 10m U-Wind (Eastward) | m/s |
| v10 | 10m V-Wind (Northward) | m/s |
| sp | Surface Pressure | Pa |
| mslp | Mean Sea Level Pressure | Pa |
| tcc | Total Cloud Cover | 0-1 |
| cp | Convective Precipitation | m |
| lsp | Large-scale Precipitation | m |
| tp | Total Precipitation | m |

**Common Regions:** global, north_atlantic, north_pacific, california_coast,
mediterranean, gulf_of_mexico, nino34 (El Nino), arctic, antarctic

### 2. ANALYSIS: `python_repl`
A PERSISTENT Jupyter kernel with state preserved between calls.

**Pre-loaded:** pandas (pd), numpy (np), xarray (xr), matplotlib.pyplot (plt)

**Data Loading:**
```python
ds = xr.open_dataset('PATH_FROM_TOOL', engine='zarr')
print(ds)
```

**Visualization Best Practices:**
```python
# Create figure and plot
fig, ax = plt.subplots(figsize=(12, 8))
# ... plotting code ...
plt.tight_layout()
plt.show()  # ALWAYS call plt.show() to display the plot
```

### 3. MEMORY
I remember our conversation history and can recall previous analyses.
Ask me to "recall" or "remember" what we discussed.

## WORKFLOW

1. **UNDERSTAND** the question - clarify if needed
2. **RETRIEVE** data with appropriate query_type and bounds
3. **ANALYZE** using Python - always print() results
4. **VISUALIZE** when helpful - use plt.show() to display plots
5. **SYNTHESIZE** findings into clear answers

## TIPS

- Convert temperatures: `(temp_K - 273.15)` for Celsius
- Wind speed: `np.sqrt(u10**2 + v10**2)`
- Time averaging: `ds.mean(dim='time')` or `ds.resample(time='1M').mean()`
- Geographic mean: `ds.mean(dim=['latitude', 'longitude'])`
- For trends: `scipy.stats.linregress` or `xr.polyfit`

## RESPONSE STYLE
- Be precise and scientific
- Include units in all values
- Reference specific dates/locations
- Explain methodology briefly
- If uncertain, acknowledge limitations
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
