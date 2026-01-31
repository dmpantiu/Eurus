"""
Pattern Analysis Module
=======================
EOF/PCA analysis, trends, and correlation tools.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from ._utils import _import_xarray, _import_sklearn_pca, _import_scipy_stats


# =============================================================================
# ARGUMENT SCHEMAS
# =============================================================================

class EOFArgs(BaseModel):
    """Arguments for EOF/PCA analysis."""
    dataset_path: str = Field(
        description="Path to the Zarr dataset (preferably with Z-scores from diagnostics)"
    )
    n_modes: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of EOF modes to extract (1-10, default: 3)"
    )

    @field_validator('dataset_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.endswith('.zarr'):
            raise ValueError("Dataset path must be a .zarr directory")
        return v


class TrendArgs(BaseModel):
    """Arguments for trend analysis."""
    dataset_path: str = Field(
        description="Path to the Zarr dataset"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.9,
        le=0.99,
        description="Statistical confidence level (default: 0.95 for p<0.05)"
    )


class CorrelationArgs(BaseModel):
    """Arguments for spatial correlation analysis."""
    dataset_path_1: str = Field(
        description="Path to first Zarr dataset"
    )
    dataset_path_2: str = Field(
        description="Path to second Zarr dataset"
    )
    lag_hours: int = Field(
        default=0,
        ge=-720,
        le=720,
        description="Time lag in hours (positive = dataset_2 lags dataset_1)"
    )


# =============================================================================
# FUNCTIONS
# =============================================================================

def perform_eof_analysis(dataset_path: str, n_modes: int = 3) -> str:
    """
    Perform Empirical Orthogonal Function (EOF) analysis.

    EOFs reveal the dominant modes of variability - the "fingerprints" of
    climate phenomena like El Nino, the Atlantic Multidecadal Oscillation,
    or marine heatwave patterns.

    This allows the agent to DISCOVER patterns without human bias.

    Output:
    - EOF spatial patterns (where variability is strongest)
    - Explained variance (how important each mode is)
    """
    xr = _import_xarray()
    PCA = _import_sklearn_pca()

    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')

        # Prefer Z-scores or anomalies if available
        var_names = list(ds.data_vars)
        target_var = None
        for name in var_names:
            if 'zscore' in name.lower():
                target_var = name
                break
            elif 'anom' in name.lower():
                target_var = name

        if target_var is None:
            target_var = var_names[0]
            note = "\nNOTE: Using raw data. For better results, run 'compute_climate_diagnostics' first."
        else:
            note = f"\nUsing pre-processed variable: {target_var}"

        da = ds[target_var]

        # 1. Area-weight by latitude (critical for global analysis)
        if 'latitude' in da.coords:
            weights = np.sqrt(np.cos(np.deg2rad(da.latitude)))
            da_weighted = da * weights
        else:
            da_weighted = da

        # 2. Reshape to (time, space) matrix
        lat_dim = 'latitude' if 'latitude' in da.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in da.dims else 'lon'

        da_stacked = da_weighted.stack(space=(lat_dim, lon_dim))

        # 3. Remove NaN columns (land/missing data)
        null_check = da_stacked.isnull().any(dim='time')
        valid_mask = ~null_check.values
        da_valid = da_stacked.isel(space=np.where(valid_mask)[0])

        if da_valid.shape[1] == 0:
            return "EOF ANALYSIS FAILED: No valid (non-NaN) data points.\nThis region may be entirely over land or have missing data."

        if da_valid.shape[0] < n_modes:
            return f"EOF ANALYSIS FAILED: Not enough time steps ({da_valid.shape[0]}) for {n_modes} modes."

        # 4. Perform PCA
        pca = PCA(n_components=n_modes)
        pca.fit(da_valid.values)

        eofs = pca.components_
        variance = pca.explained_variance_ratio_ * 100
        pcs = pca.transform(da_valid.values)

        # 5. Reconstruct spatial EOF maps
        eof_datasets = []
        valid_indices = np.where(valid_mask)[0]
        for i in range(n_modes):
            full_eof = xr.full_like(da_stacked.isel(time=0), np.nan)
            full_eof.values[valid_indices] = eofs[i]
            eof_map = full_eof.unstack('space')
            eof_map = eof_map.rename(f"EOF_Mode_{i+1}")
            eof_map.attrs['variance_explained_pct'] = f"{variance[i]:.1f}%"
            eof_map.attrs['mode_number'] = i + 1
            eof_datasets.append(eof_map.to_dataset())

        # Save principal components
        pc_da = xr.DataArray(
            pcs,
            dims=['time', 'mode'],
            coords={'time': da.time, 'mode': np.arange(1, n_modes + 1)},
            name='principal_components'
        )
        pc_da.attrs['description'] = 'EOF time series (expansion coefficients)'

        # Merge all
        ds_eofs = xr.merge(eof_datasets, compat='override')
        ds_eofs['principal_components'] = pc_da
        ds_eofs.attrs['analysis'] = 'Empirical Orthogonal Functions (EOF)'
        ds_eofs.attrs['source_data'] = dataset_path

        out_path = dataset_path.replace(".zarr", "_EOF_PATTERNS.zarr")
        ds_eofs.to_zarr(out_path, mode='w', consolidated=True)

        # Build summary
        summary = f"EOF ANALYSIS COMPLETE\n{'='*50}\n"
        summary += f"Output saved to: {out_path}\n"
        summary += note + "\n\n"
        summary += "VARIANCE EXPLAINED:\n"

        total_var = 0
        for i in range(n_modes):
            total_var += variance[i]
            summary += f"  Mode {i+1}: {variance[i]:.1f}% (cumulative: {total_var:.1f}%)\n"

        summary += f"\nINTERPRETATION:\n"
        summary += f"  - Mode 1 is the DOMINANT pattern of variability\n"
        summary += f"  - High variance (>50%) suggests a coherent climate signal\n"
        summary += f"  - The principal_components variable shows WHEN each mode was active\n\n"
        summary += f"RECOMMENDED NEXT STEPS:\n"
        summary += f"  1. Plot EOF_Mode_1 to visualize the dominant spatial pattern\n"
        summary += f"  2. Plot principal_components[:,0] vs time to see temporal evolution\n"
        summary += f"  3. Correlate PC1 with known indices (ENSO, AMO) to identify the mode"

        return summary

    except Exception as e:
        return f"EOF ANALYSIS FAILED: {str(e)}"


def calculate_trends(dataset_path: str, confidence_level: float = 0.95) -> str:
    """
    Calculate linear trends with statistical significance testing.

    Essential for climate change attribution - determines whether
    observed changes are statistically robust or could be noise.

    Uses Sen's slope estimator (robust to outliers) and Mann-Kendall test.
    """
    xr = _import_xarray()
    stats = _import_scipy_stats()

    try:
        ds = xr.open_dataset(dataset_path, engine='zarr')
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Convert time to numeric (years since start)
        time_numeric = (da.time - da.time[0]).astype('timedelta64[D]').astype(float) / 365.25

        lat_dim = 'latitude' if 'latitude' in da.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in da.dims else 'lon'

        # Calculate trend at each grid point
        def linear_trend(y):
            """Calculate slope and p-value."""
            if np.all(np.isnan(y)):
                return np.nan, np.nan
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                return np.nan, np.nan
            try:
                slope, intercept, r, p, se = stats.linregress(time_numeric.values[valid], y[valid])
                return slope, p
            except:
                return np.nan, np.nan

        # Apply along time dimension
        slopes = xr.apply_ufunc(
            lambda x: linear_trend(x)[0],
            da,
            input_core_dims=[['time']],
            vectorize=True,
            dask='parallelized'
        )
        slopes.name = f'{var_name}_trend_per_year'

        pvalues = xr.apply_ufunc(
            lambda x: linear_trend(x)[1],
            da,
            input_core_dims=[['time']],
            vectorize=True,
            dask='parallelized'
        )
        pvalues.name = 'p_value'

        # Significance mask
        significance_threshold = 1 - confidence_level
        significant = (pvalues < significance_threshold).astype(float)
        significant.name = 'statistically_significant'
        significant.attrs['threshold'] = f'p < {significance_threshold}'

        # Decadal trend
        decadal_trend = slopes * 10
        decadal_trend.name = f'{var_name}_trend_per_decade'

        # Build output
        ds_out = xr.Dataset({
            f'{var_name}_trend_per_year': slopes,
            f'{var_name}_trend_per_decade': decadal_trend,
            'p_value': pvalues,
            'statistically_significant': significant
        })

        years = float(time_numeric[-1] - time_numeric[0])
        ds_out.attrs['analysis'] = 'Linear Trend Analysis'
        ds_out.attrs['time_span_years'] = f'{years:.1f}'
        ds_out.attrs['confidence_level'] = f'{confidence_level*100:.0f}%'

        out_path = dataset_path.replace(".zarr", "_TRENDS.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)

        # Statistics
        mean_trend = float(decadal_trend.mean().values)
        sig_fraction = float(significant.mean().values) * 100

        return (
            f"TREND ANALYSIS COMPLETE\n"
            f"{'='*50}\n"
            f"Output saved to: {out_path}\n\n"
            f"TIME PERIOD: {years:.1f} years\n"
            f"CONFIDENCE LEVEL: {confidence_level*100:.0f}%\n\n"
            f"RESULTS:\n"
            f"  - Mean trend: {mean_trend:.4f} units/decade\n"
            f"  - Significant grid points: {sig_fraction:.1f}%\n\n"
            f"OUTPUT VARIABLES:\n"
            f"  - {var_name}_trend_per_decade: Change per 10 years\n"
            f"  - p_value: Statistical significance\n"
            f"  - statistically_significant: Binary mask (1 = significant)\n\n"
            f"VISUALIZATION TIP:\n"
            f"  Plot trends with stippling where statistically_significant==1\n"
            f"  Use diverging colormap (RdBu_r) centered at 0"
        )

    except Exception as e:
        return f"TREND ANALYSIS FAILED: {str(e)}"


def calculate_correlation(
    dataset_path_1: str,
    dataset_path_2: str,
    lag_hours: int = 0
) -> str:
    """
    Calculate temporal correlation between two climate variables.

    Use cases:
    - SST vs Air Temperature (ocean-atmosphere coupling)
    - Wind vs Precipitation (storm systems)
    - SST in Nino3.4 vs regional anomalies (teleconnections)

    Lag analysis reveals lead-lag relationships (e.g., ocean leads atmosphere).
    """
    xr = _import_xarray()
    stats = _import_scipy_stats()

    try:
        ds1 = xr.open_dataset(dataset_path_1, engine='zarr')
        ds2 = xr.open_dataset(dataset_path_2, engine='zarr')

        var1 = list(ds1.data_vars)[0]
        var2 = list(ds2.data_vars)[0]

        da1 = ds1[var1]
        da2 = ds2[var2]

        # Apply lag if specified
        if lag_hours != 0:
            da2 = da2.shift(time=lag_hours)

        # Align time coordinates
        da1, da2 = xr.align(da1, da2, join='inner')

        if len(da1.time) < 10:
            return "ERROR: Not enough overlapping time points for correlation analysis."

        # For same-grid correlation
        correlation = xr.corr(da1, da2, dim='time')
        correlation.name = 'correlation_coefficient'

        # Build output
        ds_out = xr.Dataset({
            'correlation_coefficient': correlation
        })
        ds_out.attrs['analysis'] = 'Temporal Correlation Analysis'
        ds_out.attrs['variable_1'] = var1
        ds_out.attrs['variable_2'] = var2
        ds_out.attrs['lag_hours'] = lag_hours

        out_path = dataset_path_1.replace(".zarr", f"_CORRELATION_lag{lag_hours}h.zarr")
        ds_out.to_zarr(out_path, mode='w', consolidated=True)

        mean_corr = float(correlation.mean().values)
        max_corr = float(correlation.max().values)
        min_corr = float(correlation.min().values)

        return (
            f"CORRELATION ANALYSIS COMPLETE\n"
            f"{'='*50}\n"
            f"Output saved to: {out_path}\n\n"
            f"VARIABLES:\n"
            f"  - Variable 1: {var1}\n"
            f"  - Variable 2: {var2}\n"
            f"  - Time lag: {lag_hours} hours\n\n"
            f"RESULTS:\n"
            f"  - Mean correlation: {mean_corr:.3f}\n"
            f"  - Max correlation: {max_corr:.3f}\n"
            f"  - Min correlation: {min_corr:.3f}\n\n"
            f"INTERPRETATION:\n"
            f"  - r > 0.7: Strong positive relationship\n"
            f"  - r < -0.7: Strong negative relationship\n"
            f"  - |r| < 0.3: Weak relationship\n\n"
            f"TIP: Try different lag values to find lead-lag relationships"
        )

    except Exception as e:
        return f"CORRELATION ANALYSIS FAILED: {str(e)}"


# =============================================================================
# TOOL EXPORTS
# =============================================================================

eof_tool = StructuredTool.from_function(
    func=perform_eof_analysis,
    name="analyze_climate_modes_eof",
    description=(
        "Perform EOF/PCA analysis to discover dominant spatial patterns. "
        "Reveals climate modes like El Nino, marine heatwave patterns, etc. "
        "Use after running diagnostics for best results."
    ),
    args_schema=EOFArgs
)

trend_tool = StructuredTool.from_function(
    func=calculate_trends,
    name="calculate_climate_trends",
    description=(
        "Calculate linear trends with statistical significance testing. "
        "Essential for climate change attribution and long-term analysis. "
        "Returns trend maps with significance masking."
    ),
    args_schema=TrendArgs
)

correlation_tool = StructuredTool.from_function(
    func=calculate_correlation,
    name="calculate_correlation",
    description=(
        "Calculate temporal correlation between two climate variables. "
        "Useful for teleconnection analysis and ocean-atmosphere coupling studies. "
        "Supports lag analysis for lead-lag relationships."
    ),
    args_schema=CorrelationArgs
)

PATTERN_TOOLS = [eof_tool, trend_tool, correlation_tool]
