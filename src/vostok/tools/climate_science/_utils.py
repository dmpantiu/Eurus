"""
Shared utilities and lazy imports for climate science tools.
"""

import numpy as np


def _import_xarray():
    import xarray as xr
    return xr


def _import_sklearn_pca():
    from sklearn.decomposition import PCA
    return PCA


def _import_scipy_stats():
    from scipy import stats
    return stats


def _import_granger():
    from statsmodels.tsa.stattools import grangercausalitytests
    return grangercausalitytests
