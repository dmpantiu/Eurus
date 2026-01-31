"""
Extremes Detection Module
=========================
Compound events, percentile-based extremes, and return period analysis.
"""

import numpy as np
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from ._utils import _import_xarray, _import_scipy_stats


# =============================================================================
# ARGUMENT SCHEMAS
# =============================================================================

class CompoundExtremeArgs(BaseModel):
    """Arguments for compound extreme detection."""
    sst_path: str = Field(
        description="Path to SST dataset with Z-scores (run diagnostics first)"
    )
    wind_path: str = Field(
        description="Path to wind dataset with Z-scores (run diagnostics first)"
    )
    heat_threshold: float = Field(
        default=1.5,
        description="Z-score threshold for 'hot' (default: 1.5 ~ 90th percentile)"
    )
    stagnation_threshold: float = Field(
        default=-1.0,
        description="Z-score threshold for 'stagnant' wind (default: -1.0)"
    )


class PercentileArgs(BaseModel):
    """Arguments for percentile/extreme detection."""
    dataset_path: str = Field(
        description="Path to the Zarr dataset"
    )
    percentile: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Percentile threshold (e.g., 95 for 95th percentile extremes)"
    )
    extreme_type: Literal["above", "below"] = Field(
        default="above",
        description="'above' for hot extremes, 'below' for cold extremes"
    )


class ReturnPeriodArgs(BaseModel):
    """Arguments for Extreme Value Analysis."""
    dataset_path: str = Field(
        description="Path to Zarr dataset (needs multi-year data for reliable fits)"
    )
    block_size: str = Field(
        default="year",
        description="Block size for block maxima: 'year', 'month', or 'season'"
    )
    fit_type: Literal["maxima", "minima"] = Field(
        default="maxima",
        description="Fit to block maxima (heat extremes) or minima (cold extremes)"
    )

    @field_validator('dataset_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.endswith('.zarr'):
            raise ValueError("Dataset path must be a .zarr directory")
        return v


# =============================================================================
# FUNCTIONS
# =============================================================================

def detect_compound_extremes(
    sst_path: str,
    wind_path: str,
    heat_threshold: float = 1.5,
    stagnation_threshold: float = -1.0
) -> str:
    """
    Detect 'Ocean Ovens': compound events where extreme heat coincides
    with atmospheric stagnation (low wind).

    Scientific Mechanism:
    - Low wind prevents ocean mixing
    - Heat accumulates at the surface
    - Marine ecosystems experience lethal stress

    This is the type of compound extreme analysis that leads to
    high-impact publications.
    """
    xr = _import_xarray()

    try:
        # Load datasets
        ds_sst = xr.open_dataset(sst_path, engine='zarr')
        ds_wind = xr.open_dataset(wind_path, engine='zarr')

        # Find Z-score variables
        sst_var = None
        for v in ds_sst.data_vars:
            if 'zscore' in v.lower():
                sst_var = v
                break
        if sst_var is None:
            return "ERROR: SST dataset must contain Z-scores. Run 'compute_climate_diagnostics' first."

        wind_var = None
        for v in ds_wind.data_vars:
            if 'zscore' in v.lower():
                wind_var = v
                break
        if wind_var is None:
            return "ERROR: Wind dataset must contain Z-scores. Run 'compute_climate_diagnostics' first."

        sst_z = ds_sst[sst_var]
        wind_z = ds_wind[wind_var]

        # Align grids (interpolate if needed)
        sst_z, wind_z = xr.align(sst_z, wind_z, join='inner')

        if len(sst_z.time) == 0:
            return ("ERROR: No overlapping time period between SST and Wind datasets.\n"
                   "Ensure both datasets cover the same date range.")

        # Define extreme conditions
        hot = sst_z > heat_threshold
        stagnant = wind_z < stagnation_threshold
        compound = hot & stagnant

        # Calculate statistics
        total_hours = len(sst_z.time)
        compound_hours = compound.sum(dim='time').astype(float)
        compound_hours.name = 'compound_event_hours'

        # Intensity during compound events
        intensity = sst_z.where(compound).mean(dim='time')
        intensity.name = 'mean_sst_zscore_during_compound'

        # Build output dataset
        ds_out = xr.Dataset({
            'compound_event_hours': compound_hours,
            'mean_sst_zscore_during_compound': intensity
        })
        ds_out.attrs['analysis'] = 'Compound Extreme Detection (Ocean Oven)'
        ds_out.attrs['heat_threshold'] = f'SST Z-score > {heat_threshold}'
        ds_out.attrs['stagnation_threshold'] = f'Wind Z-score < {stagnation_threshold}'
        ds_out.attrs['total_hours_analyzed'] = total_hours

        out_path = sst_path.replace(".zarr", "_COMPOUND_EXTREMES.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)

        # Summary stats
        total_compound_hours = int(compound_hours.sum().values)
        max_hours = int(compound_hours.max().values)
        mean_intensity = float(intensity.mean().values) if not np.isnan(intensity.mean().values) else 0.0

        # Find hotspot location
        if max_hours > 0:
            lat_dim = 'latitude' if 'latitude' in compound_hours.dims else 'lat'
            lon_dim = 'longitude' if 'longitude' in compound_hours.dims else 'lon'
            max_idx = compound_hours.argmax(dim=[lat_dim, lon_dim])
            max_lat = float(compound_hours[lat_dim][max_idx[lat_dim]].values)
            max_lon = float(compound_hours[lon_dim][max_idx[lon_dim]].values)
            hotspot_info = f"  - Hotspot location: {max_lat:.1f}°N, {max_lon:.1f}°E\n"
        else:
            hotspot_info = ""

        return (
            f"COMPOUND EXTREME DETECTION COMPLETE\n"
            f"{'='*50}\n"
            f"Output saved to: {out_path}\n\n"
            f"CRITERIA USED:\n"
            f"  - Heat: SST Z-score > {heat_threshold}σ (top ~{100*(1-0.5*(1+np.math.erf(heat_threshold/np.sqrt(2)))):.1f}%)\n"
            f"  - Stagnation: Wind Z-score < {stagnation_threshold}σ\n"
            f"  - Compound: BOTH conditions simultaneously\n\n"
            f"RESULTS:\n"
            f"  - Time period analyzed: {total_hours} hours\n"
            f"  - Total compound grid-point-hours: {total_compound_hours:,}\n"
            f"  - Maximum at any location: {max_hours} hours\n"
            f"  - Mean intensity during events: {mean_intensity:.2f}σ\n"
            f"{hotspot_info}\n"
            f"SCIENTIFIC INTERPRETATION:\n"
            f"  High compound_event_hours indicates 'Ocean Oven' zones where:\n"
            f"  1. Surface waters are anomalously warm\n"
            f"  2. Wind mixing is suppressed\n"
            f"  3. Marine ecosystems face compounded thermal stress\n\n"
            f"RECOMMENDED:\n"
            f"  - Plot 'compound_event_hours' to map ecosystem risk zones\n"
            f"  - Overlay with coral reef / fishery locations for impact assessment"
        )

    except Exception as e:
        return f"COMPOUND DETECTION FAILED: {str(e)}"


def detect_percentile_extremes(
    dataset_path: str,
    percentile: float = 95.0,
    extreme_type: str = "above"
) -> str:
    """
    Detect extreme events based on percentile thresholds.

    Use cases:
    - Marine heatwaves (SST > 90th percentile)
    - Cold spells (Temperature < 10th percentile)
    - Extreme wind events (Wind > 95th percentile)

    Returns a mask of extreme events and their statistics.
    """
    xr = _import_xarray()

    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Calculate threshold
        if extreme_type == "above":
            threshold = da.quantile(percentile / 100.0, dim='time')
            extreme_mask = da > threshold
            description = f'Events above {percentile}th percentile'
        else:
            threshold = da.quantile(percentile / 100.0, dim='time')
            extreme_mask = da < threshold
            description = f'Events below {percentile}th percentile'

        # Count extreme hours
        extreme_hours = extreme_mask.sum(dim='time').astype(float)
        extreme_hours.name = 'extreme_event_hours'

        # Calculate percentage
        total_hours = len(da.time)
        extreme_pct = (extreme_hours / total_hours * 100)
        extreme_pct.name = 'extreme_percentage'

        # Intensity during extremes
        intensity = da.where(extreme_mask).mean(dim='time')
        intensity.name = 'extreme_intensity'

        # Build output
        ds_out = xr.Dataset({
            'extreme_event_hours': extreme_hours,
            'extreme_percentage': extreme_pct,
            'extreme_intensity': intensity,
            'threshold': threshold.drop_vars('quantile', errors='ignore')
        })
        ds_out.attrs['analysis'] = f'Percentile Extreme Detection ({description})'
        ds_out.attrs['percentile'] = percentile
        ds_out.attrs['extreme_type'] = extreme_type

        out_path = dataset_path.replace(".zarr", f"_EXTREMES_p{int(percentile)}.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)

        total_extreme_hours = int(extreme_hours.sum().values)
        mean_intensity = float(intensity.mean().values)

        return (
            f"EXTREME DETECTION COMPLETE\n"
            f"{'='*50}\n"
            f"Output saved to: {out_path}\n\n"
            f"CRITERIA:\n"
            f"  - Type: {extreme_type} {percentile}th percentile\n"
            f"  - Expected frequency: ~{100-percentile if extreme_type=='above' else percentile:.1f}%\n\n"
            f"RESULTS:\n"
            f"  - Total extreme hours: {total_extreme_hours:,}\n"
            f"  - Mean intensity: {mean_intensity:.2f}\n\n"
            f"OUTPUT VARIABLES:\n"
            f"  - extreme_event_hours: Count at each location\n"
            f"  - extreme_percentage: Fraction of time in extreme state\n"
            f"  - threshold: The threshold value at each location"
        )

    except Exception as e:
        return f"EXTREME DETECTION FAILED: {str(e)}"


def calculate_return_periods(
    dataset_path: str,
    block_size: str = "year",
    fit_type: str = "maxima"
) -> str:
    """
    Fit a Generalized Extreme Value (GEV) distribution to calculate Return Periods.

    Scientific Context:
    - Answers: "Is this a 1-in-100 year event?"
    - CRITICAL for Nature papers: Quantifies rarity beyond just "sigma"
    - Uses Block Maxima approach (standard for climate extremes)

    Requirements:
    - At least 20 years of data for reliable GEV fits
    - Data is spatially aggregated if gridded (regional risk assessment)

    Returns:
    - GEV parameters (shape, location, scale)
    - Return levels for 10, 50, 100, 500 year events
    """
    xr = _import_xarray()
    stats = _import_scipy_stats()

    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Get units for output
        units = da.attrs.get('units', 'units')

        # Spatial aggregation if gridded data
        if 'latitude' in da.dims and len(da.latitude) > 1:
            da_agg = da.mean(dim=['latitude', 'longitude'])
            spatial_note = "Data spatially averaged for regional analysis."
        else:
            da_agg = da
            spatial_note = "Single point or already aggregated data."

        # Block maxima/minima extraction
        if block_size == "year":
            grouper = "time.year"
            block_desc = "annual"
        elif block_size == "month":
            grouper = "time.month"
            block_desc = "monthly"
        elif block_size == "season":
            grouper = "time.season"
            block_desc = "seasonal"
        else:
            return f"Error: Unknown block_size '{block_size}'. Use 'year', 'month', or 'season'."

        if fit_type == "maxima":
            block_extremes = da_agg.groupby(grouper).max()
            extreme_desc = "maxima"
        else:
            block_extremes = da_agg.groupby(grouper).min()
            extreme_desc = "minima"

        # Extract values and remove NaNs
        data = block_extremes.values.flatten()
        data = data[~np.isnan(data)]

        n_blocks = len(data)
        if n_blocks < 15:
            return (
                f"Error: Only {n_blocks} blocks available. GEV fitting requires at least 15-20 blocks.\n"
                f"TIP: For annual maxima, you need ~20 years of data.\n"
                f"Try using 'month' block_size for more samples, or retrieve longer time series."
            )

        # Fit GEV distribution
        c, loc, scale = stats.genextreme.fit(data)

        # Calculate return levels
        return_periods = [10, 20, 50, 100, 200, 500]
        return_levels = {}

        for rp in return_periods:
            prob = 1 - (1.0 / rp)
            level = stats.genextreme.ppf(prob, c, loc, scale)
            return_levels[rp] = level

        # Current observed maximum for context
        obs_max = float(np.max(data))
        obs_min = float(np.min(data))

        # Estimate return period of observed maximum
        obs_prob = stats.genextreme.cdf(obs_max, c, loc, scale)
        if obs_prob < 1.0:
            obs_return_period = 1 / (1 - obs_prob)
        else:
            obs_return_period = float('inf')

        # Build summary
        summary = f"EXTREME VALUE ANALYSIS (GEV Fit)\n{'='*55}\n"
        summary += f"Variable: {var_name}\n"
        summary += f"Block size: {block_desc} {extreme_desc}\n"
        summary += f"Number of blocks: {n_blocks}\n"
        summary += f"{spatial_note}\n\n"

        summary += f"GEV PARAMETERS:\n"
        summary += f"  - Shape (ξ): {-c:.4f}\n"
        summary += f"  - Location (μ): {loc:.4f}\n"
        summary += f"  - Scale (σ): {scale:.4f}\n"

        # Interpret shape parameter
        if c < -0.1:
            shape_interp = "Heavy-tailed (Frechet): Extreme events more likely than normal"
        elif c > 0.1:
            shape_interp = "Bounded (Weibull): Finite upper limit to extremes"
        else:
            shape_interp = "Light-tailed (Gumbel): Classic extreme value behavior"
        summary += f"  - Interpretation: {shape_interp}\n\n"

        summary += f"RETURN LEVELS ({units}):\n"
        for rp in return_periods:
            level = return_levels[rp]
            summary += f"  - {rp:3d}-Year Event: {level:.3f}\n"

        summary += f"\nOBSERVED DATA:\n"
        summary += f"  - Maximum observed: {obs_max:.3f}\n"
        summary += f"  - Minimum observed: {obs_min:.3f}\n"
        if obs_return_period < 10000:
            summary += f"  - Estimated return period of max: ~{obs_return_period:.0f} years\n"
        else:
            summary += f"  - Estimated return period of max: >10,000 years\n"

        summary += f"\nSCIENTIFIC INTERPRETATION:\n"
        summary += f"  - A '{return_periods[2]}-year event' has {100/return_periods[2]:.1f}% annual probability\n"
        summary += f"  - If observed max exceeds 100-year level, it's a rare extreme\n"
        summary += f"  - Compare recent events to historical return levels for attribution\n"

        # Save results
        out_path = dataset_path.replace(".zarr", "_EVT_ANALYSIS.txt")
        with open(out_path, 'w') as f:
            f.write(summary)

        summary += f"\nResults saved to: {out_path}"

        return summary

    except Exception as e:
        return f"GEV Analysis Failed: {str(e)}"


# =============================================================================
# TOOL EXPORTS
# =============================================================================

compound_tool = StructuredTool.from_function(
    func=detect_compound_extremes,
    name="detect_compound_extremes",
    description=(
        "Detect 'Ocean Ovens' - compound events where hot SST coincides with stagnant winds. "
        "These compound extremes cause severe marine ecosystem stress. "
        "REQUIRES Z-score data from compute_climate_diagnostics."
    ),
    args_schema=CompoundExtremeArgs
)

percentile_tool = StructuredTool.from_function(
    func=detect_percentile_extremes,
    name="detect_percentile_extremes",
    description=(
        "Detect extreme events using percentile thresholds (e.g., 90th, 95th). "
        "Alternative to Z-score method for extreme detection. "
        "Good for marine heatwave and cold spell identification."
    ),
    args_schema=PercentileArgs
)

return_period_tool = StructuredTool.from_function(
    func=calculate_return_periods,
    name="calculate_return_periods",
    description=(
        "Fit GEV distribution to calculate Return Periods (e.g., '1-in-100 year event'). "
        "CRITICAL for Nature papers - quantifies rarity beyond Z-scores. "
        "Requires multi-year data (20+ years recommended)."
    ),
    args_schema=ReturnPeriodArgs
)

EXTREME_TOOLS = [compound_tool, percentile_tool, return_period_tool]
