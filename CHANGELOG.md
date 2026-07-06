# Changelog

All releases are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/).

## [0.34.5](https://github.com/isometric/calculation-library/releases/tag/v0.34.5)

### Changed

- `biosphere.constants`: `CO2_TO_CARBON_RATIO` - derived from the centralized `atomic_weight()` helper instead of hardcoded literals. Small numerical differences in results are expected, but they are not significant.

## [0.34.4](https://github.com/isometric/calculation-library/releases/tag/v0.34.4)

### Changed

- `CHANGELOG.md`: non-material formatting change

## [0.34.3](https://github.com/isometric/calculation-library/releases/tag/v0.34.3)

### Changed

- Enforced canonical `CHANGELOG.md` entry format and reformatted old entries
- Upgraded the Keep a Changelog reference from 1.1.0 to 2.0.0

## [0.34.2](https://github.com/isometric/calculation-library/releases/tag/v0.34.2)

### Changed

- Documentation: `CHANGELOG.md` is now bundled into the source distribution, so it's mirrored to the public GitHub repo.

## [0.34.1](https://github.com/isometric/calculation-library/releases/tag/v0.34.1)

### Changed

- Documentation: backfilled the remaining patch-version entries in `CHANGELOG.md` from every released GitHub tag (previously only minor versions were covered), and switched to inline release links in each heading.

## [0.34.0](https://github.com/isometric/calculation-library/releases/tag/v0.34.0)

### Changed

- Vectorised the inventory Monte Carlo correlated draw (internal performance improvement); results are unchanged.

## [0.33.0](https://github.com/isometric/calculation-library/releases/tag/v0.33.0)

### Changed

- Internal naming cleanup (`clip_sigmas`, `dbh_with_blunders`). No changes to public calculation results.

### Removed

- Deprecated control-correction compatibility shims.

## [0.32.12](https://github.com/isometric/calculation-library/releases/tag/v0.32.12)

### Changed

- Documentation: added `CHANGELOG.md` with full release history backfilled from PyPI, and a link to it in `README.public.md`.

## [0.32.11](https://github.com/isometric/calculation-library/releases/tag/v0.32.11)

Dependency version update.

## [0.32.10](https://github.com/isometric/calculation-library/releases/tag/v0.32.10)

Dependency version update.

## [0.32.9](https://github.com/isometric/calculation-library/releases/tag/v0.32.9)

Dependency version update.

## [0.32.8](https://github.com/isometric/calculation-library/releases/tag/v0.32.8)

Dependency version update.

## [0.32.7](https://github.com/isometric/calculation-library/releases/tag/v0.32.7)

Dependency version update.

## [0.32.6](https://github.com/isometric/calculation-library/releases/tag/v0.32.6)

Dependency version update.

## [0.32.5](https://github.com/isometric/calculation-library/releases/tag/v0.32.5)

### Changed

- `enhanced_weathering.utils.feedstock_weighting`: `compute_weighted_feedstock_composition` and `bootstrap_weighted_feedstock` now raise `ValueError` when `batch_weights` sums to zero/negative, or when none of the weighted batches are present/complete in the input data
- `enhanced_weathering.utils.statistical_checks.power_analysis.compute_power_analysis`: now raises `ValueError` if fewer than 2 non-null baseline/reporting-period values per element
- `enhanced_weathering.utils.statistical_checks.representativeness.check_representativeness`: now raises `ValueError` if either group has fewer than 2 samples
- `enhanced_weathering.utils.statistical_checks.weathering_signal`: `check_weathering_significance` and `check_weathering_significance_paired` now raise `ValueError` if fewer than 2 (matched) samples are provided

## [0.32.4](https://github.com/isometric/calculation-library/releases/tag/v0.32.4)

Internal improvements only, no changes to public-facing functionality.

## [0.32.3](https://github.com/isometric/calculation-library/releases/tag/v0.32.3)

Dependency version update.

## [0.32.2](https://github.com/isometric/calculation-library/releases/tag/v0.32.2)

Internal improvements only, no changes to public-facing functionality.

## [0.32.1](https://github.com/isometric/calculation-library/releases/tag/v0.32.1)

Dependency version update.

## [0.32.0](https://github.com/isometric/calculation-library/releases/tag/v0.32.0)

### Changed

- `enhanced_weathering.utils.feedstock_weighting`: `compute_weighted_feedstock_composition` and `bootstrap_weighted_feedstock` now raise `ValueError` on zero total weight or when no weighted batch is present in the input data (previously silently normalised over an empty set)

## [0.31.0](https://github.com/isometric/calculation-library/releases/tag/v0.31.0)

### Changed

- `calculations.enhanced_weathering_cdr_tracer_ti_treatment_only.main`: all parameters are now keyword-only; now raises `ValueError` if no samples are assigned to a treatment/control plot, if a plot type has samples in only one of baseline/reporting-period, or if treatment area is non-positive (previously could silently produce zero/NaN results)
- `enhanced_weathering.utils.cdr.compute_weathered_fraction`: now raises `ValueError` when `theoretical_potential_tco2` is non-positive instead of dividing to inf/NaN
- `enhanced_weathering.utils.statistical_checks.tracer_resolvability.calculate_tracer_resolvability`: now raises `ValueError` on non-positive total mass or non-positive noise term, instead of silently returning NaN/inf
- `enhanced_weathering.utils.tracer.compute_mass_ratio_from_immobile_tracer`: infinite ratios are now converted to NaN instead of returned as inf

### Fixed

- `enhanced_weathering.utils.data_cleaning.ProcessingReport.summary`: no longer raises `ZeroDivisionError` when the first processing step starts from zero rows; reports "n/a" instead

## [0.30.1](https://github.com/isometric/calculation-library/releases/tag/v0.30.1)

Dependency version update.

## [0.30.0](https://github.com/isometric/calculation-library/releases/tag/v0.30.0)

### Added

- `enhanced_weathering.utils.cdr.compute_weathered_fraction_standard_tca` — weathered fraction from measured post-application baseline, end-of-reporting-period, pre-application baseline, and control-dissolution concentrations; returns NaN (not inf) where the denominator is zero

## [0.29.19](https://github.com/isometric/calculation-library/releases/tag/v0.29.19)

Dependency version update.

## [0.29.18](https://github.com/isometric/calculation-library/releases/tag/v0.29.18)

Internal improvements only, no changes to public-facing functionality.

## [0.29.17](https://github.com/isometric/calculation-library/releases/tag/v0.29.17)

Dependency version update.

## [0.29.16](https://github.com/isometric/calculation-library/releases/tag/v0.29.16)

Internal improvements only, no changes to public-facing functionality.

## [0.29.15](https://github.com/isometric/calculation-library/releases/tag/v0.29.15)

Internal improvements only, no changes to public-facing functionality.

## [0.29.14](https://github.com/isometric/calculation-library/releases/tag/v0.29.14)

Dependency version update.

## [0.29.13](https://github.com/isometric/calculation-library/releases/tag/v0.29.13)

Dependency version update.

## [0.29.12](https://github.com/isometric/calculation-library/releases/tag/v0.29.12)

Internal improvements only, no changes to public-facing functionality.

## [0.29.11](https://github.com/isometric/calculation-library/releases/tag/v0.29.11)

Dependency version update.

## [0.29.10](https://github.com/isometric/calculation-library/releases/tag/v0.29.10)

Internal improvements only, no changes to public-facing functionality.

## [0.29.9](https://github.com/isometric/calculation-library/releases/tag/v0.29.9)

Internal improvements only, no changes to public-facing functionality.

## [0.29.8](https://github.com/isometric/calculation-library/releases/tag/v0.29.8)

Dependency version update.

## [0.29.7](https://github.com/isometric/calculation-library/releases/tag/v0.29.7)

Internal improvements only, no changes to public-facing functionality.

## [0.29.6](https://github.com/isometric/calculation-library/releases/tag/v0.29.6)

Internal improvements only, no changes to public-facing functionality.

## [0.29.5](https://github.com/isometric/calculation-library/releases/tag/v0.29.5)

Dependency version update.

## [0.29.4](https://github.com/isometric/calculation-library/releases/tag/v0.29.4)

Dependency version update.

## [0.29.3](https://github.com/isometric/calculation-library/releases/tag/v0.29.3)

Dependency version update.

## [0.29.2](https://github.com/isometric/calculation-library/releases/tag/v0.29.2)

Dependency version update.

## [0.29.1](https://github.com/isometric/calculation-library/releases/tag/v0.29.1)

Dependency version update.

## [0.29.0](https://github.com/isometric/calculation-library/releases/tag/v0.29.0)

### Added

- `enhanced_weathering.utils.data_cleaning.null_filter` — removes samples with NaN values, mirroring `zero_filter`

### Changed

- `enhanced_weathering.utils.data_cleaning.zero_filter`: return type renamed from `ZeroFilterResult` to `SampleFilterResult`

## [0.28.1](https://github.com/isometric/calculation-library/releases/tag/v0.28.1)

Dependency version update.

## [0.28.0](https://github.com/isometric/calculation-library/releases/tag/v0.28.0)

### Added

- `enhanced_weathering.utils.cdr.compute_depth_weighted_concentration_kg_ha` — per-sample depth-weighted cation concentration (kg/ha per unit bulk density), avoiding bias from averaging concentration and depth independently

## [0.27.0](https://github.com/isometric/calculation-library/releases/tag/v0.27.0)

### Added

- `enhanced_weathering.utils.data_cleaning.iterative_sigma_clip` — outlier clipping with bounds computed iteratively from the clean subset, tighter than `winsorise` when strong outliers inflate group std

## [0.26.2](https://github.com/isometric/calculation-library/releases/tag/v0.26.2)

Dependency version update.

## [0.26.1](https://github.com/isometric/calculation-library/releases/tag/v0.26.1)

Dependency version update.

## [0.26.0](https://github.com/isometric/calculation-library/releases/tag/v0.26.0)

### Changed

- `enhanced_weathering.utils.resampling.resample_mean`: input array now cast to `float32` before bootstrap resampling

## [0.25.3](https://github.com/isometric/calculation-library/releases/tag/v0.25.3)

Internal improvements only, no changes to public-facing functionality.

## [0.25.2](https://github.com/isometric/calculation-library/releases/tag/v0.25.2)

Dependency version update.

## [0.25.1](https://github.com/isometric/calculation-library/releases/tag/v0.25.1)

Dependency version update.

## [0.25.0](https://github.com/isometric/calculation-library/releases/tag/v0.25.0)

### Added

- `enhanced_weathering.utils.feedstock_weighting.bootstrap_weighted_feedstock`: new `noise_rng`/`noise_fractions` keywords for per-column proportional Gaussian measurement noise
- `enhanced_weathering.utils.resampling.compute_resampled_means_from_indices`: new `noise_rng`/`noise_fraction` keywords, same purpose

## [0.24.0](https://github.com/isometric/calculation-library/releases/tag/v0.24.0)

### Changed

- `enhanced_weathering.utils.statistical_checks.application_rate.build_application_rate_check`: added `n_std` parameter (default 2 std); output column renamed from `known_within_3std` to dynamic `known_within_{n_std}std`

## [0.23.0](https://github.com/isometric/calculation-library/releases/tag/v0.23.0)

### Added

- `enhanced_weathering.utils.feedstock_weighting`: `compute_weighted_feedstock_composition`, `compute_plot_coverage_weights`, `bootstrap_weighted_feedstock` — weighted feedstock composition across multiple batches/crushers, with bootstrap uncertainty
- `enhanced_weathering.utils.statistical_checks.control_correction_significance`: `check_background_weathering_significance_paired`, `check_background_weathering_significance_unpaired`
- `enhanced_weathering.utils.statistical_checks.power_analysis`: `compute_power_analysis` — sampling-design power analysis
- `geospatial.spatial_autocorrelation`: `compute_morans_i_permutation_test`, `compute_neff_from_morans_i` — Moran's I and related spatial autocorrelation utilities
- `enhanced_weathering.utils.control_correction`: `apply_control_correction_delta_paired`, `apply_control_correction_delta_unpaired` — gated, bootstrapped additive control-correction delta

### Removed

- `enhanced_weathering.utils.statistical_checks.control_correction` (superseded by `control_correction_significance`)

## [0.22.0](https://github.com/isometric/calculation-library/releases/tag/v0.22.0)

### Added

- `biosphere.allometric_equations`: `chave` — Chave et al. pantropical biomass allometric equations; `wood_density` — global wood density lookup
- `biosphere.constants`, `biosphere.types` - shared constants and domain types for biosphere calculations
- `biosphere.monte_carlo`: `field_plot`, `inventory` — Monte Carlo uncertainty propagation for field plots and carbon inventories
- `biosphere.utils`: `dbh`, `height`, `clipped_normal` — diameter at breast height, tree height utilities, and clipped-normal distribution

## [0.21.17](https://github.com/isometric/calculation-library/releases/tag/v0.21.17)

Dependency version update.

## [0.21.16](https://github.com/isometric/calculation-library/releases/tag/v0.21.16)

Dependency version update.

## [0.21.15](https://github.com/isometric/calculation-library/releases/tag/v0.21.15)

Dependency version update.

## [0.21.14](https://github.com/isometric/calculation-library/releases/tag/v0.21.14)

Dependency version update.

## [0.21.13](https://github.com/isometric/calculation-library/releases/tag/v0.21.13)

Internal improvements only, no changes to public-facing functionality.

## [0.21.12](https://github.com/isometric/calculation-library/releases/tag/v0.21.12)

### Fixed

- `utils.elements`: test suite now imports `ELEMENTS` from `molmass.elements` instead of `molmass` (no production code change)

## [0.21.11](https://github.com/isometric/calculation-library/releases/tag/v0.21.11)

Dependency version update.

## [0.21.10](https://github.com/isometric/calculation-library/releases/tag/v0.21.10)

Dependency version update.

## [0.21.9](https://github.com/isometric/calculation-library/releases/tag/v0.21.9)

Dependency version update.

## [0.21.8](https://github.com/isometric/calculation-library/releases/tag/v0.21.8)

### Added

- `biosphere` (new module) — inventory-based biomass estimation
- `biosphere.allometric_equations.chave`: `ChaveModel`, `CHAVE_DEFAULT`, `linearize_allometric_se`, `create_chave_model_generator` — Chave et al. 2014 pantropical AGB allometry
- `biosphere.allometric_equations.wood_density`: `WoodDensityRecord`, `get_wood_density`, `list_species`, `tree_type_to_species` — species-level wood density lookup
- `biosphere.constants`: `DBH_ERROR_SLOPE`, `DBH_ERROR_INTERCEPT`, `CARBON_FRACTION`, `CO2_TO_CARBON_RATIO`, `M2_PER_HECTARE`, `CONSERVATIVE_PERCENTILE`
- `biosphere.monte_carlo.field_plot`: `TreeMeasurements`, `FieldPlot` — per-plot tCO2e/ha from tree-level measurements
- `biosphere.monte_carlo.inventory`: `inventory_monte_carlo`, `MONTE_CARLO_VARIANTS` — Monte Carlo error propagation for DBH, height, wood density, carbon ratio
- `biosphere.utils`: `clipped_normal`, `dbh`, `height` — perturbation utilities

## [0.21.7](https://github.com/isometric/calculation-library/releases/tag/v0.21.7)

Dependency version update.

## [0.21.5](https://github.com/isometric/calculation-library/releases/tag/v0.21.5)

Internal improvements only, no changes to public-facing functionality.

## [0.21.4](https://github.com/isometric/calculation-library/releases/tag/v0.21.4)

### Added

- `dependencies`: `scipy` added to the vendored dependency exports

## [0.21.3](https://github.com/isometric/calculation-library/releases/tag/v0.21.3)

### Added

- `calculations.enhanced_weathering_cdr_tracer_ti_treatment_only`: `main` — first end-to-end published calculation model, Ti tracer-corrected total-cation-analysis CDR quantification for treatment plots only, with ratio-based control correction and bootstrap uncertainty
- `enhanced_weathering.utils.statistical_checks.weathering_signal.check_weathering_significance_paired` — paired (matched-location) one-tailed significance test, more powerful than the unpaired test when spatial variance is high relative to the weathering signal
- `enhanced_weathering.utils.resampling.summarize_distributions` — summarizes bootstrap distributions into mean/std/p5/p16/median/p84/p95

## [0.21.2](https://github.com/isometric/calculation-library/releases/tag/v0.21.2)

Dependency version update.

## [0.21.1](https://github.com/isometric/calculation-library/releases/tag/v0.21.1)

Internal improvements only, no changes to public-facing functionality.

## [0.21.0](https://github.com/isometric/calculation-library/releases/tag/v0.21.0)

### Added

- `dependencies`: `rasterio`, `statsmodels` added to the vendored dependency exports

## [0.20.9](https://github.com/isometric/calculation-library/releases/tag/v0.20.9)

Dependency version update.

## [0.20.8](https://github.com/isometric/calculation-library/releases/tag/v0.20.8)

Dependency version update.

## [0.20.7](https://github.com/isometric/calculation-library/releases/tag/v0.20.7)

Dependency version update.

## [0.20.6](https://github.com/isometric/calculation-library/releases/tag/v0.20.6)

Internal improvements only, no changes to public-facing functionality.

## [0.20.5](https://github.com/isometric/calculation-library/releases/tag/v0.20.5)

Dependency version update.

## [0.20.4](https://github.com/isometric/calculation-library/releases/tag/v0.20.4)

Dependency version update.

## [0.20.3](https://github.com/isometric/calculation-library/releases/tag/v0.20.3)

Internal improvements only, no changes to public-facing functionality.

## [0.20.2](https://github.com/isometric/calculation-library/releases/tag/v0.20.2)

Dependency version update.

## [0.20.1](https://github.com/isometric/calculation-library/releases/tag/v0.20.1)

Dependency version update.

## [0.20.0](https://github.com/isometric/calculation-library/releases/tag/v0.20.0)

### Added

- `utils.elements` — `ElementSymbol`, `to_element_symbol`, `atomic_number`, `atomic_weight`, `element_name` — full periodic table data for accurate molar masses

### Changed

- `enhanced_weathering.utils.conversions`: `convert_cation_kg_to_co2_kg` now uses precise per-element atomic weights via `utils.elements.atomic_weight` (e.g. Ca: 40.078 vs previous 40.08), giving slightly different CO2-equivalent results
- `enhanced_weathering.utils.tracer.ImmobileTracer`: changed from a `StrEnum` to a `Literal["Zr", "Ti"]` type alias
- `enhanced_weathering.utils.control_correction.bootstrap_control_correction_ratios` and `statistical_checks.weathering_signal.run_significance_tests`: `elements` parameter now typed as `ElementSymbol` instead of plain `str`

## [0.19.1](https://github.com/isometric/calculation-library/releases/tag/v0.19.1)

Internal improvements only, no changes to public-facing functionality.

## [0.19.0](https://github.com/isometric/calculation-library/releases/tag/v0.19.0)

### Added

- `enhanced_weathering.utils`: `control_correction` — correction of treatment measurements against paired control plots; `pairing` — spatial pairing of treatment and control samples
- `enhanced_weathering.utils.statistical_checks`: `application_rate`, `tracer_resolvability`

## [0.18.0](https://github.com/isometric/calculation-library/releases/tag/v0.18.0)

### Added

- `enhanced_weathering.utils`: `cdr`, `conversions`, `data_cleaning`, `resampling`, `spatial`, `tracer`, `types` — core utilities for enhanced weathering CDR quantification
- `enhanced_weathering.utils.statistical_checks`: `representativeness`, `weathering_signal`
- `dependencies`: `xarray` added to the vendored dependency exports

## [0.17.8](https://github.com/isometric/calculation-library/releases/tag/v0.17.8)

Dependency version update.

## [0.17.7](https://github.com/isometric/calculation-library/releases/tag/v0.17.7)

Dependency version update.

## [0.17.6](https://github.com/isometric/calculation-library/releases/tag/v0.17.6)

### Added

- `enhanced_weathering` (new module) — introduces the enhanced weathering CDR quantification building blocks
- `enhanced_weathering.utils.conversions`: `Cation`, `convert_mg_kg_to_kg_ha`, `convert_kg_ha_to_mg_kg`, `convert_cation_kg_to_co2_kg` — soil mass fraction/CO2 unit conversions
- `enhanced_weathering.utils.cdr`: `compute_cation_stock_kg_ha`, `compute_feedstock_cation_kg_ha`, `compute_cdr_from_stocks`, `compute_cdr_density`, `compute_control_dissolved_kg_ha`, `convert_cdr_to_co2`, `compute_weathered_fraction`, `WeatheredFractionResult` — core CDR calculation from cation stocks (total cation and tracer methods)
- `enhanced_weathering.utils.tracer`: `ImmobileTracer`, `compute_mass_ratio_from_immobile_tracer`, `compute_post_application_concentration`, `compute_control_correction_ratio`, `compute_fraction_dissolved`, `calculate_tracer_resolvability`, `compute_application_rate_from_tracer` — immobile tracer (Zr/Ti) mass-balance methodology
- `enhanced_weathering.utils.data_cleaning`: `ProcessingStep`, `ProcessingReport`, `ZeroFilterResult`, `WinsoriseResult`, `zero_filter`, `winsorise` — sample cleaning/outlier utilities with reporting
- `enhanced_weathering.utils.resampling`: `resample_mean`, `resample_dataframe_unpaired`, `resample_dataframe_paired`, `bootstrap_bulk_density_unpaired`, `bootstrap_bulk_density_paired`, `generate_bootstrap_location_indices`, `compute_resampled_means_from_indices`, `resample_by_group` — bootstrap resampling utilities
- `enhanced_weathering.utils.spatial`: `PlotType`, `assign_area_type`, `calculate_area_hectares_by_plot_type` — spatial join of samples to plot polygons and area calculation
- `enhanced_weathering.utils.statistical_checks.representativeness`: `RepresentativenessTestResult`, `check_representativeness` — two-tailed test comparing treatment/deployment distributions
- `enhanced_weathering.utils.statistical_checks.weathering_signal`: `SignificanceTestResult`, `infer_post_application_concentrations`, `check_weathering_significance` — one-tailed significance test for weathering signal

## [0.17.5](https://github.com/isometric/calculation-library/releases/tag/v0.17.5)

Dependency version update.

## [0.17.4](https://github.com/isometric/calculation-library/releases/tag/v0.17.4)

Internal improvements only, no changes to public-facing functionality.

## [0.17.3](https://github.com/isometric/calculation-library/releases/tag/v0.17.3)

Internal improvements only, no changes to public-facing functionality.

## [0.17.2](https://github.com/isometric/calculation-library/releases/tag/v0.17.2)

Dependency version update.

## [0.17.1](https://github.com/isometric/calculation-library/releases/tag/v0.17.1)

Internal improvements only, no changes to public-facing functionality.

## [0.17.0](https://github.com/isometric/calculation-library/releases/tag/v0.17.0)

### Added

- Initial internal README and packaging metadata; no public modules yet.

## [0.16.0](https://github.com/isometric/calculation-library/releases/tag/v0.16.0)

Internal improvements only, no changes to public-facing functionality.

## [0.15.0](https://github.com/isometric/calculation-library/releases/tag/v0.15.0)

### Added

- Initial public release: package scaffolding only (`pyproject.toml`, license, empty `calculations` module); no public calculation modules yet
- `dependencies` — re-exports `geopandas`, `numpy`, `pandas`, `shapely` so downstream code only needs to pin this library's version
