"""
Climate Science Intelligence Package
=====================================
Advanced statistical methods for climate attribution, pattern discovery,
and compound extreme detection.

This package transforms Vostok from a data utility into a scientific
discovery engine capable of publication-grade analysis.

Modules:
    - diagnostics: Anomalies, Z-scores, and detrending
    - patterns: EOF/PCA, trends, and correlation analysis
    - extremes: Compound events, percentile detection, return periods
    - attribution: Climate indices, composite analysis, Granger causality
"""

# Import all tools from submodules
from .diagnostics import (
    DiagnosticsArgs,
    DetrendArgs,
    calculate_climate_diagnostics,
    detrend_climate_data,
    diagnostics_tool,
    detrend_tool,
    DIAGNOSTICS_TOOLS,
)

from .patterns import (
    EOFArgs,
    TrendArgs,
    CorrelationArgs,
    perform_eof_analysis,
    calculate_trends,
    calculate_correlation,
    eof_tool,
    trend_tool,
    correlation_tool,
    PATTERN_TOOLS,
)

from .extremes import (
    CompoundExtremeArgs,
    PercentileArgs,
    ReturnPeriodArgs,
    detect_compound_extremes,
    detect_percentile_extremes,
    calculate_return_periods,
    compound_tool,
    percentile_tool,
    return_period_tool,
    EXTREME_TOOLS,
)

from .attribution import (
    IndexArgs,
    CompositeArgs,
    GrangerArgs,
    fetch_climate_index,
    perform_composite_analysis,
    analyze_granger_causality,
    index_tool,
    composite_tool,
    granger_tool,
    ATTRIBUTION_TOOLS,
)


# =============================================================================
# AGGREGATED EXPORTS
# =============================================================================

# Complete list of all 11 science tools
SCIENCE_TOOLS = [
    # Core diagnostics (2)
    diagnostics_tool,
    detrend_tool,
    # Pattern analysis (3)
    eof_tool,
    trend_tool,
    correlation_tool,
    # Extreme detection (3)
    compound_tool,
    percentile_tool,
    return_period_tool,
    # Attribution & causality (3)
    index_tool,
    composite_tool,
    granger_tool,
]

# Convenience function
def get_science_tools():
    """Return all climate science tools."""
    return SCIENCE_TOOLS


__all__ = [
    # Tool list
    'SCIENCE_TOOLS',
    'get_science_tools',
    # Individual tools
    'diagnostics_tool',
    'detrend_tool',
    'eof_tool',
    'trend_tool',
    'correlation_tool',
    'compound_tool',
    'percentile_tool',
    'return_period_tool',
    'index_tool',
    'composite_tool',
    'granger_tool',
    # Functions (for direct use)
    'calculate_climate_diagnostics',
    'detrend_climate_data',
    'perform_eof_analysis',
    'calculate_trends',
    'calculate_correlation',
    'detect_compound_extremes',
    'detect_percentile_extremes',
    'calculate_return_periods',
    'fetch_climate_index',
    'perform_composite_analysis',
    'analyze_granger_causality',
    # Argument schemas
    'DiagnosticsArgs',
    'DetrendArgs',
    'EOFArgs',
    'TrendArgs',
    'CorrelationArgs',
    'CompoundExtremeArgs',
    'PercentileArgs',
    'ReturnPeriodArgs',
    'IndexArgs',
    'CompositeArgs',
    'GrangerArgs',
]
