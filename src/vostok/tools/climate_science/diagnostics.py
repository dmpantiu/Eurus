"""
Climate Diagnostics Module
===========================
Core preprocessing tools: anomalies, Z-scores, and detrending.
"""

import numpy as np
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from ._utils import _import_xarray


# =============================================================================
# ARGUMENT SCHEMAS
# =============================================================================

class DiagnosticsArgs(BaseModel):
    """Arguments for climate diagnostics calculation."""
    dataset_path: str = Field(
        description="Path to the downloaded Zarr dataset (from retrieve_era5_data)"
    )
    baseline_start: str = Field(
        default="1991",
        description="Start year for climatological baseline (default: 1991 for WMO standard)"
    )
    baseline_end: str = Field(
        default="2020",
        description="End year for climatological baseline (default: 2020 for WMO standard)"
    )

    @field_validator('dataset_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.endswith('.zarr'):
            raise ValueError("Dataset path must be a .zarr directory")
        return v


class DetrendArgs(BaseModel):
    """Arguments for detrending climate data."""
    dataset_path: str = Field(description="Path to the Zarr dataset to detrend")
    method: Literal["linear", "constant"] = Field(
        default="linear", 
        description="Remove 'linear' trend (warming) or 'constant' mean"
    )


# =============================================================================
# FUNCTIONS
# =============================================================================

def calculate_climate_diagnostics(
    dataset_path: str,
    baseline_start: str = "1991",
    baseline_end: str = "2020"
) -> str:
    """
    Transform raw climate data into scientific insights.

    Calculates:
    1. Daily Climatology (seasonal cycle removal)
    2. Anomalies (departure from normal)
    3. Z-Scores (standardized anomalies for extreme detection)

    Scientific Note:
    - Z-Score > 2.0: Statistically significant extreme (~2.3% probability)
    - Z-Score > 3.0: Rare extreme (~0.1% probability)
    - This is the FOUNDATION of all climate science analysis.
    """
    xr = _import_xarray()

    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Check if we have enough baseline data
        time_range = ds.time.values
        baseline_data = da.sel(time=slice(baseline_start, baseline_end))

        if len(baseline_data.time) == 0:
            return (
                f"ERROR: No data in baseline period {baseline_start}-{baseline_end}.\n"
                f"Dataset time range: {str(time_range[0])[:10]} to {str(time_range[-1])[:10]}\n"
                "TIP: Adjust baseline_start/baseline_end to match your data, or retrieve a longer time series."
            )

        # Calculate daily climatology (mean for each day of year)
        climatology = baseline_data.groupby("time.dayofyear").mean("time")
        std_dev = baseline_data.groupby("time.dayofyear").std("time")

        # Prevent division by zero
        std_dev = xr.where(std_dev > 0, std_dev, 1e-10)

        # Calculate anomaly and Z-score for ALL data
        anomalies = da.groupby("time.dayofyear") - climatology
        z_scores = anomalies / std_dev

        # Build output dataset
        ds_out = xr.Dataset({
            f"{var_name}_anom": anomalies.drop_vars('dayofyear', errors='ignore'),
            f"{var_name}_zscore": z_scores.drop_vars('dayofyear', errors='ignore')
        })

        # Preserve coordinates and attributes
        ds_out.attrs['description'] = f'Climatological Anomalies & Z-Scores (baseline: {baseline_start}-{baseline_end})'
        ds_out.attrs['baseline_period'] = f"{baseline_start}-{baseline_end}"
        ds_out.attrs['source_variable'] = var_name

        # Save output
        out_path = dataset_path.replace(".zarr", "_DIAGNOSTICS.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)

        # Calculate summary statistics
        max_zscore = float(z_scores.max().values)
        min_zscore = float(z_scores.min().values)
        extreme_count = int((np.abs(z_scores) > 2.0).sum().values)

        return (
            f"CLIMATE DIAGNOSTICS COMPLETE\n"
            f"{'='*50}\n"
            f"Output saved to: {out_path}\n\n"
            f"Variables created:\n"
            f"  - {var_name}_anom: Raw anomalies (departure from {baseline_start}-{baseline_end} mean)\n"
            f"  - {var_name}_zscore: Standardized anomalies (units: standard deviations)\n\n"
            f"Summary Statistics:\n"
            f"  - Max Z-Score: {max_zscore:.2f}σ\n"
            f"  - Min Z-Score: {min_zscore:.2f}σ\n"
            f"  - Extreme events (|Z| > 2σ): {extreme_count:,} grid-point-hours\n\n"
            f"INTERPRETATION:\n"
            f"  - Z > +2σ: Unusually HOT (top 2.3%)\n"
            f"  - Z < -2σ: Unusually COLD (bottom 2.3%)\n"
            f"  - Z > +3σ: EXTREME heat (top 0.1%)\n\n"
            f"NEXT STEP: Use 'analyze_climate_modes_eof' to find spatial patterns,\n"
            f"or 'detect_compound_extremes' to find Ocean Ovens."
        )

    except Exception as e:
        return f"DIAGNOSTICS FAILED: {str(e)}\nTIP: Ensure the dataset path is correct and data is not corrupted."


def detrend_climate_data(dataset_path: str, method: str = "linear") -> str:
    """
    Removes the long-term trend from the dataset using polynomial fitting.

    SCIENTIFIC NECESSITY:
    - Before correlating variables (e.g., SST vs Index), you MUST detrend.
    - Otherwise, the global warming trend creates FALSE positives (r > 0.9).
    - This isolates "Internal Variability" from "Forced Trends."
    """
    xr = _import_xarray()
    
    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        print(f"Detrending {var_name} using {method} fit...")
        
        if method == "linear":
            # Polyfit handles NaNs and Dask arrays efficiently
            coeffs = da.polyfit(dim="time", deg=1, skipna=True)
            fit = xr.polyval(da.time, coeffs.polyfit_coefficients)
            detrended = da - fit
        else:
            mean = da.mean(dim="time", skipna=True)
            detrended = da - mean
            
        # Metadata
        detrended.name = f"{var_name}_detrended"
        detrended.attrs = da.attrs.copy()
        detrended.attrs['processing'] = f"Detrended ({method})"
        
        # Save
        ds_out = detrended.to_dataset()
        out_path = dataset_path.replace(".zarr", f"_DETRENDED.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)
        
        return (
            f"DETRENDING COMPLETE\n{'='*50}\n"
            f"Saved to: {out_path}\n\n"
            f"INSTRUCTION: Use THIS file for 'calculate_correlation' or 'analyze_granger_causality'\n"
            f"to prove that relationships are physical, not just shared warming trends."
        )

    except Exception as e:
        return f"Detrending Failed: {str(e)}"


# =============================================================================
# TOOL EXPORTS
# =============================================================================

diagnostics_tool = StructuredTool.from_function(
    func=calculate_climate_diagnostics,
    name="compute_climate_diagnostics",
    description=(
        "Calculate climate anomalies and Z-scores from raw data. "
        "ESSENTIAL first step for any scientific analysis. "
        "Z-scores enable detection of statistically significant extremes."
    ),
    args_schema=DiagnosticsArgs
)

detrend_tool = StructuredTool.from_function(
    func=detrend_climate_data,
    name="detrend_climate_data",
    description=(
        "Remove long-term warming trend to isolate internal variability. "
        "MANDATORY before correlation analysis on multi-decadal data. "
        "Reviewer #2 will reject papers that skip this step."
    ),
    args_schema=DetrendArgs
)

DIAGNOSTICS_TOOLS = [diagnostics_tool, detrend_tool]
