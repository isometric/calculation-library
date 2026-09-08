# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Access to the NSVB coefficient and reference tables shipped with this package."""

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

_cache = dict[str, pd.DataFrame]()


def load_table(name: str) -> pd.DataFrame:
    """Read one of the packaged NSVB tables, caching it for the process lifetime."""
    if name not in _cache:
        _cache[name] = pd.read_csv(_DATA_DIR / f"{name}.csv")
    return _cache[name]
