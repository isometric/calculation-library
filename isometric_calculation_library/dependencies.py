# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Re-exported dependencies of `isometric_calculation_library`.

This library acts as a choke point for external dependencies. As long as a Python module imports
only from `isometric_calculation_library`, this library's version alone is sufficient to reproduce
all its dependencies.
"""

import geopandas
import numpy  # noqa: ICN001
import pandas  # noqa: ICN001
import rasterio
import scipy
import shapely
import statsmodels
import statsmodels.formula.api as statsmodels_formula_api
import xarray

__all__ = [
    "geopandas",
    "numpy",
    "pandas",
    "rasterio",
    "scipy",
    "shapely",
    "statsmodels",
    "statsmodels_formula_api",
    "xarray",
]
