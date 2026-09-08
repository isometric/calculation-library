# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""NSVB allometric equations for above-ground biomass of United States tree species.

Implements the above-ground part of the USDA Forest Inventory and Analysis national-scale
volume and biomass (NSVB) framework:
  Westfall JA, Coulston JW, Gray AN, Shaw JD, Radtke PJ, Walker DM, Weiskittel AR,
  MacFarlane DW, Affleck DLR, Zhao D, Temesgen H, Poudel KP, Domke GM, Tyrrell ML (2024)
  "A national-scale tree volume, biomass, and carbon modeling system for the United
  States." USDA Forest Service General Technical Report WO-104.

NSVB is species-specific rather than pantropical: coefficients are selected by FIA species
code, optionally refined by ecodivision and stand origin. This makes it the counterpart of
:mod:`..chave` for temperate and boreal North America.

Entry point is :func:`.coefficients.get_nsvb_model`. ``README.md`` beside this file records
the scope, the provenance of the coefficient tables, how uncertainty is derived, and where
the implementation departs from the framework's published R script.
"""
