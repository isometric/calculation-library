# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np

type Np1DArray[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
type Np2DArray[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]


def mass_fraction_column_name(element: str) -> str:
    """Column name for an element's mass fraction (mg/kg) in soil or feedstock DataFrames."""
    return f"mass_fraction_{element.lower()}"
