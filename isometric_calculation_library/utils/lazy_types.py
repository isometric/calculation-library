# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Aliases of third-party types, for the field annotations that Python evaluates when it creates a
class.

Dataclasses, NamedTuples and TypedDicts evaluate their field annotations when the class is created.
Python evaluates the value of a `type` statement only when something reads it, so with lazy imports
(PEP 810), a class that names an alias from this module does not load the library when its module
is imported.
"""

import numpy as np
import pandas as pd
from rasterio import crs as rasterio_crs
from rasterio import transform as rasterio_transform
from sklearn import preprocessing as sklearn_preprocessing

from isometric_calculation_library.utils.types import Np1DArray, Np2DArray

type DataFrame = pd.DataFrame
type Series = pd.Series

type NpArray = np.ndarray
type NpBool1DArray = Np1DArray[np.bool_]
type NpFloat641DArray = Np1DArray[np.float64]
type NpFloat642DArray = Np2DArray[np.float64]
type NpFloating1DArray = Np1DArray[np.floating]
type NpFloating2DArray = Np2DArray[np.floating]
type NpInteger1DArray = Np1DArray[np.integer]
type NpInteger2DArray = Np2DArray[np.integer]
type NpIntp1DArray = Np1DArray[np.intp]

type RasterioAffine = rasterio_transform.Affine
type RasterioCrs = rasterio_crs.CRS

type SklearnStandardScaler = sklearn_preprocessing.StandardScaler
