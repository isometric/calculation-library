# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from collections.abc import Callable
from typing import NewType

__version__ = "0.22.0"
"""Version format: MAJOR.MINOR.PATCH.

The minor version is normally incremented for changes to the calculations themselves, the patch
version for dependency updates and non-substantive changes to the internals of the library. The
major version is not normally incremented.
"""

ModelKey = NewType("ModelKey", str)

type ModelImplementation = Callable[..., object]
