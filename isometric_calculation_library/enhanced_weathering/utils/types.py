# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np

type Np1DArray[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
