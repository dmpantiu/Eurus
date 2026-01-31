"""
Attribution Analysis Module
===========================
Climate indices, composite analysis, and Granger causality.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from ._utils import _import_xarray, _import_granger


# =============================================================================
# ARGUMENT SCHEMAS
# =============================================================================

class IndexArgs(BaseModel):
    """Arguments for fetching climate indices."""
    index_name: str = Field(
        description="Name of the index: 'nino34', 'nao', 'pdo', 'amo', 'soi'"
    )
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")

    @field_validator('index_name')
    @classmethod
    def validate_index(cls, v: str) -> str:
        valid = ['nino34', 'nao', 'pdo', 'amo', 'soi']
        if v.lower() not in valid:
            raise ValueError(f"Unknown index '{v}'. Supported: {valid}")
        return v.lower()


class CompositeArgs(BaseModel):
    """Arguments for Composite Analysis."""
    target_path: str = Field(description="Path to target variable (e.g., Pressure/Wind)")
    index_path: str = Field(description="Path to index/mask (e.g., Nino3.4 or Extreme Events)")
    threshold: float = Field(default=1.0, description="Event threshold (e.g., Index > 1.0)")


class GrangerArgs(BaseModel):
    """Arguments for Granger Causality."""
    dataset_x: str = Field(description="Path to potential DRIVER (X)")
    dataset_y: str = Field(description="Path to potential RESPONSE (Y)")
    max_lag: int = Field(default=5, description="Max lag to test (e.g., months)")


# =============================================================================
# FUNCTIONS
# =============================================================================

def _get_index_interpretation(idx: str) -> str:
    """Get interpretation guide for climate indices."""
    guides = {
        'nino34': (
            "  - Positive values (+): El Nino conditions (warm central Pacific)\n"
            "  - Negative values (-): La Nina conditions (cool central Pacific)\n"
            "  - |value| > 0.5: Weak event\n"
            "  - |value| > 1.0: Moderate event\n"
            "  - |value| > 1.5: Strong event"
        ),
        'nao': (
            "  - Positive NAO: Strong Icelandic Low, strong Azores High\n"
            "    -> Warm, wet winters in N. Europe; dry Mediterranean\n"
            "  - Negative NAO: Weak pressure gradient\n"
            "    -> Cold winters in N. Europe; wet Mediterranean"
        ),
        'pdo': (
            "  - Positive PDO: Warm eastern Pacific, cool western Pacific\n"
            "    -> Often co-occurs with El Nino patterns\n"
            "  - Negative PDO: Cool eastern Pacific, warm western Pacific\n"
            "    -> Multi-decadal oscillation (20-30 year phases)"
        ),
        'amo': (
            "  - Positive AMO: Warm North Atlantic SSTs\n"
            "    -> Active Atlantic hurricane seasons, Sahel rainfall\n"
            "  - Negative AMO: Cool North Atlantic SSTs\n"
            "    -> Multi-decadal oscillation (60-80 year cycle)"
        ),
        'soi': (
            "  - Negative SOI: El Nino (low pressure over eastern Pacific)\n"
            "  - Positive SOI: La Nina (high pressure over eastern Pacific)\n"
            "  - Note: SOI is inverse of Nino indices"
        ),
    }
    return guides.get(idx, "  - Standard climate oscillation index")


def fetch_climate_index(index_name: str, start_date: str, end_date: str) -> str:
    """
    Retrieve standard climate indices from NOAA/PSL.

    Essential for ATTRIBUTION. Allows checking if a local event is driven
    by large-scale climate modes (e.g., "Is this drought caused by El Nino?").

    Available Indices:
    - nino34: El Nino 3.4 region SST anomaly (central Pacific)
    - nao: North Atlantic Oscillation
    - pdo: Pacific Decadal Oscillation
    - amo: Atlantic Multidecadal Oscillation
    - soi: Southern Oscillation Index

    Returns a time series dataset for correlation with your analysis.
    """
    xr = _import_xarray()

    # Standard NOAA PSL Data URLs
    URLS = {
        'nino34': 'https://psl.noaa.gov/data/correlation/nina34.data',
        'nao': 'https://psl.noaa.gov/data/correlation/nao.data',
        'pdo': 'https://psl.noaa.gov/data/correlation/pdo.data',
        'amo': 'https://psl.noaa.gov/data/correlation/amon.us.data',
        'soi': 'https://psl.noaa.gov/data/correlation/soi.data',
    }

    idx = index_name.lower()
    if idx not in URLS:
        return f"Error: Unknown index '{index_name}'. Supported: {list(URLS.keys())}"

    try:
        url = URLS[idx]

        # Robust parsing for NOAA's messy text files
        df = pd.read_csv(
            url,
            sep=r'\s+',
            skiprows=1,
            header=None,
            engine='python',
            on_bad_lines='skip'
        )

        # Filter for valid years
        df = df[pd.to_numeric(df[0], errors='coerce').between(1800, 2100)]

        if len(df.columns) < 13:
            return f"Error: Unexpected data format from NOAA for {index_name}"

        # Reshape
        df.columns = ['year'] + list(range(1, 13))
        df = df.melt(id_vars=['year'], var_name='month', value_name='value')

        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        df['time'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values('time').set_index('time')[['value']]

        # Filter missing values
        df = df[df['value'] > -90]

        # Filter to requested date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_subset = df.loc[mask]

        if df_subset.empty:
            return (
                f"Error: No data found for {index_name} in range {start_date} to {end_date}.\n"
                f"Available range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
            )

        # Convert to xarray Dataset
        ds = xr.Dataset(
            {idx: (['time'], df_subset['value'].values.astype(float))},
            coords={'time': df_subset.index.values}
        )
        ds[idx].attrs['long_name'] = f"{index_name.upper()} Index"
        ds[idx].attrs['units'] = 'dimensionless'
        ds.attrs['source'] = f"NOAA PSL ({url})"
        ds.attrs['index_name'] = index_name.upper()

        # Save to zarr
        from ..config import DATA_DIR
        safe_start = start_date.replace('-', '')
        safe_end = end_date.replace('-', '')
        out_path = str(Path(DATA_DIR) / f"index_{idx}_{safe_start}_{safe_end}.zarr")
        ds.to_zarr(out_path, mode='w', consolidated=True)

        # Summary stats
        mean_val = float(ds[idx].mean().values)
        std_val = float(ds[idx].std().values)
        n_months = len(df_subset)

        return (
            f"CLIMATE INDEX RETRIEVED\n"
            f"{'='*50}\n"
            f"Index: {index_name.upper()}\n"
            f"Period: {start_date} to {end_date} ({n_months} months)\n"
            f"Saved to: {out_path}\n\n"
            f"STATISTICS:\n"
            f"  - Mean: {mean_val:.3f}\n"
            f"  - Std Dev: {std_val:.3f}\n"
            f"  - Min: {float(ds[idx].min().values):.3f}\n"
            f"  - Max: {float(ds[idx].max().values):.3f}\n\n"
            f"INTERPRETATION ({index_name.upper()}):\n"
            + _get_index_interpretation(idx) +
            f"\n\nNEXT STEP: Use 'calculate_correlation' to correlate this index with your spatial data,\n"
            f"or use 'python_repl' to plot the time series."
        )

    except Exception as e:
        return f"Failed to fetch {index_name} index: {str(e)}"


def perform_composite_analysis(target_path: str, index_path: str, threshold: float = 1.0) -> str:
    """
    Creates a "Composite Map" showing the average physical state during specific events.
    
    Scientific Use:
    - "Show me the average Wind pattern when Mediterranean SST > 95th percentile."
    - If a clear High Pressure system appears, you have found the MECHANISM.
    """
    xr = _import_xarray()
    
    try:
        target_ds = xr.open_dataset(target_path, engine='zarr')
        index_ds = xr.open_dataset(index_path, engine='zarr')
        
        target_var = list(target_ds.data_vars)[0]
        index_var = list(index_ds.data_vars)[0]
        
        # Create Event Mask
        index_da = index_ds[index_var]
        if 'latitude' in index_da.dims:
            index_da = index_da.mean(dim=['latitude', 'longitude'])
            
        events = index_da > threshold
        n_events = int(events.sum().values)
        
        if n_events < 5:
            return f"Error: Only {n_events} events found. Lower the threshold."

        # Composite Mean
        composite = target_ds[target_var].where(events).mean(dim='time')
        
        # Save
        ds_out = composite.to_dataset(name=f"{target_var}_composite")
        ds_out.attrs['n_events'] = n_events
        ds_out.attrs['condition'] = f"{index_var} > {threshold}"
        
        out_path = target_path.replace(".zarr", "_COMPOSITE.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)
        
        return (
            f"COMPOSITE ANALYSIS COMPLETE\n{'='*50}\n"
            f"Found {n_events} events where {index_var} > {threshold}.\n"
            f"Average state of {target_var} saved to {out_path}.\n\n"
            f"NEXT: Plot this map. Persistent features (Highs/Lows) are your drivers."
        )
    except Exception as e:
        return f"Composite Failed: {str(e)}"


def analyze_granger_causality(dataset_x: str, dataset_y: str, max_lag: int = 5) -> str:
    """
    Perform Granger Causality Test to infer directionality.
    
    Answers: "Does X happen *before* Y reliably?"
    Differentiates between Correlation (simultaneous) and Causality (predictive).
    """
    xr = _import_xarray()
    grangercausalitytests = _import_granger()
    
    try:
        ds_x = xr.open_dataset(dataset_x, engine='zarr')
        ds_y = xr.open_dataset(dataset_y, engine='zarr')
        
        # Convert to 1D time series
        da_x = ds_x[list(ds_x.data_vars)[0]]
        da_y = ds_y[list(ds_y.data_vars)[0]]
        
        if 'latitude' in da_x.dims: 
            da_x = da_x.mean(dim=['latitude', 'longitude'])
        if 'latitude' in da_y.dims: 
            da_y = da_y.mean(dim=['latitude', 'longitude'])
        
        # Align
        da_x, da_y = xr.align(da_x, da_y, join='inner')
        df = pd.DataFrame({'y': da_y.values, 'x': da_x.values}).dropna()
        
        if len(df) < 30: 
            return "Error: Need at least 30 time steps for Granger causality."
        
        # Run Test
        gc_res = grangercausalitytests(df[['y', 'x']], maxlag=max_lag, verbose=False)
        
        summary = f"GRANGER CAUSALITY TEST (X -> Y)\n{'='*50}\n"
        summary += f"Testing if X (driver) predicts Y (response)\n\n"
        significant = False
        best_lag = None
        best_p = 1.0
        
        for lag in range(1, max_lag + 1):
            p_val = gc_res[lag][0]['ssr_ftest'][1]
            marker = " ** SIGNIFICANT" if p_val < 0.05 else ""
            if p_val < 0.05: 
                significant = True
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag
            summary += f"  Lag {lag}: p = {p_val:.4f}{marker}\n"
            
        summary += "\n"
        if significant:
            summary += f"✓ CONCLUSION: X GRANGER-CAUSES Y (strongest at lag {best_lag})\n"
            summary += f"  This suggests X happens BEFORE Y with predictive power.\n"
            summary += f"  Physical interpretation: X is likely a DRIVER of Y."
        else:
            summary += "✗ CONCLUSION: No causal link found.\n"
            summary += "  Correlation may be coincidental or driven by a third variable."
            
        return summary
    except Exception as e:
        return f"Causality Failed: {str(e)}"


# =============================================================================
# TOOL EXPORTS
# =============================================================================

index_tool = StructuredTool.from_function(
    func=fetch_climate_index,
    name="fetch_climate_index",
    description=(
        "Fetch standard climate indices (Nino3.4, NAO, PDO, AMO, SOI) from NOAA. "
        "ESSENTIAL for attribution - correlate local events with large-scale modes. "
        "Returns monthly time series for the requested period."
    ),
    args_schema=IndexArgs
)

composite_tool = StructuredTool.from_function(
    func=perform_composite_analysis,
    name="perform_composite_analysis",
    description=(
        "Create composite maps showing average atmospheric state during events. "
        "Use to discover MECHANISMS - e.g., 'What does pressure look like during heatwaves?' "
        "Essential for explaining WHY extremes occur."
    ),
    args_schema=CompositeArgs
)

granger_tool = StructuredTool.from_function(
    func=analyze_granger_causality,
    name="analyze_granger_causality",
    description=(
        "Perform Granger Causality test to prove X drives Y (not just correlated). "
        "Answers: Does NAO CAUSE SST anomalies, or is it the other way around? "
        "Critical for Nature papers - moves from correlation to causation."
    ),
    args_schema=GrangerArgs
)

ATTRIBUTION_TOOLS = [index_tool, composite_tool, granger_tool]
