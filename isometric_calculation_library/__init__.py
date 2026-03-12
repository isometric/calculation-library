# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from collections.abc import Callable
from typing import NewType

__version__ = "0.15.0"
"""Version format: M.R.0 where M is the major version, R is a revision number incrementing on each release, and the patch number is always 0."""

ModelKey = NewType("ModelKey", str)

type ModelImplementation = Callable[..., object]
