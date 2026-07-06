# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Shared scientific constants for biosphere biomass quantification."""

from isometric_calculation_library.utils.elements import atomic_weight

DBH_ERROR_SLOPE = 0.0062
"""Slope of the DBH measurement error model (Chave et al. 2004, p.412)."""

DBH_ERROR_INTERCEPT = 0.0904
"""Intercept of the DBH measurement error model (Chave et al. 2004, p.412)."""

CARBON_FRACTION = 0.47
"""IPCC 2006 Guidelines, Vol. 4, Table 4.3."""

CO2_TO_CARBON_RATIO = (atomic_weight("C") + 2 * atomic_weight("O")) / atomic_weight("C")
"""Stoichiometric ratio to convert carbon mass to CO2 mass."""

M2_PER_HECTARE = 10_000
"""Square metres per hectare."""

CONSERVATIVE_PERCENTILE = 16
"""Percentile used for conservative estimates in Monte Carlo distributions."""
