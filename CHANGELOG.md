# Changelog

All releases are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/).

## [0.48.7](https://github.com/isometric/calculation-library/releases/tag/v0.48.7)

Dependency version update.

## [0.48.6](https://github.com/isometric/calculation-library/releases/tag/v0.48.6)

Dependency version update.

## [0.48.5](https://github.com/isometric/calculation-library/releases/tag/v0.48.5)

Dependency version update.

## [0.48.4](https://github.com/isometric/calculation-library/releases/tag/v0.48.4)

Dependency version update.

## [0.48.3](https://github.com/isometric/calculation-library/releases/tag/v0.48.3)

Dependency version update.

## [0.48.2](https://github.com/isometric/calculation-library/releases/tag/v0.48.2)

Dependency version update.

## [0.48.1](https://github.com/isometric/calculation-library/releases/tag/v0.48.1)

Dependency version update.

## [0.48.0](https://github.com/isometric/calculation-library/releases/tag/v0.48.0)

### Added

- `enhanced_weathering.utils.conversions`: `convert_to_equivalent_soil_mass` - restates a mass fraction on the soil mass of a different sampling event, `C * (rho_measured / rho_reference)`. A soil sample is taken to a fixed depth and so covers a fixed volume, but a mass fraction is per kilogram: when bulk density differs between two events, each event's kilogram represents a different depth of profile and their mass fractions are not directly comparable. Protocol v1.2 requires bulk density every reporting period, so a model with a measured density per event can now put both on a common basis rather than pooling them. Matters more than the density change suggests, because an immobile-tracer difference `T_rp - T_bl` is a small difference of large numbers - a ~2.4% density change is a ~5.5% change in the difference the mass balance inverts. Both densities accept bootstrap replicates
- `enhanced_weathering.utils.conversions`: `compute_residual_equivalent_soil_mass_ratio` - the per-replicate factor to apply on top of values already corrected at the two events' mean densities, `(rho_m_boot / rho_r_boot) / (mean(rho_m) / mean(rho_r))`. Applying the point correction once to the per-sample values lets every consumer working from measured values (spatial autocorrelation, power analysis, significance testing) see the same corrected difference; multiplying bootstrap replicates by this residual then swaps the fixed correction for a resampled one without applying the conversion twice. Its mean ratio is 1, so it reintroduces the correction's sampling error without shifting the central estimate - which is what a density difference within sampling error warrants

### Changed

- `enhanced_weathering.utils.spatial`: `assign_area_type`, `assign_and_split_by_plot_type` and `calculate_area_hectares_by_plot_type` take plots under the site field names `plot_type` and `geometry`. Plot boundaries come from `ENHANCED_WEATHERING_FIELD` sites now, so that is the one supported spelling and a model no longer renames columns before every call. `Type`/`Geometry` still resolve but raise a `DeprecationWarning` against the calling model: they are the column names inside the GeoJSON files the pre-site models were given, so those models cannot be moved off them by editing code alone - the uploaded files have to be reissued, and breaking them meanwhile would make an already-issued quantification unreproducible. The fallback goes once no model depends on it. A frame missing the plot type or geometry entirely raises naming what it looked for. `resolve_plot_columns`, `PLOT_TYPE_COLUMN` and `GEOMETRY_COLUMN` are exported for callers needing the same resolution
- `enhanced_weathering.utils.control_correction`: `floor_at_zero` is now a required argument on `apply_control_correction_delta_paired` and `apply_control_correction_delta_unpaired`, in place of defaulting to `True`. Flooring keeps the control correction's downside while discarding its upside, which biases CDR upward, so which behaviour applies is a modelling decision that should be visible at the call site rather than inherited silently. Existing callers that relied on the default now pass `floor_at_zero=True` explicitly and are unchanged

## [0.47.0](https://github.com/isometric/calculation-library/releases/tag/v0.47.0)

### Changed

- `biosphere.allometric_equations.wood_density`: adds nine supported tree types the table did not cover - `Anacardium occidentale`, `Bauhinia ungulata`, `Brosimum guianense`, `Byrsonima chrysophylla`, `Byrsonima densa`, `Casearia grandiflora`, `Heisteria ovata`, `Myrcia eximia` and `Pradosia lactescens`. Each value is a genus mean, carrying the genus-level standard deviation the table already uses for a genus-derived record

## [0.46.2](https://github.com/isometric/calculation-library/releases/tag/v0.46.2)

Dependency version update.

## [0.46.1](https://github.com/isometric/calculation-library/releases/tag/v0.46.1)

### Changed

- `enhanced_weathering.utils.tracer`: the `ImmobileTracer` docstring is now a single line naming what the alias is for. Which tracer element is appropriate for a deployment, and the mobility caveats that bear on that choice, are documented in the protocols rather than in the library, superseding the copper caveat added in 0.46.0. `"Zr"`, `"Ti"` and `"Cu"` all remain accepted and there are no calculation changes

## [0.46.0](https://github.com/isometric/calculation-library/releases/tag/v0.46.0)

### Changed

- `enhanced_weathering.utils.tracer`: `ImmobileTracer` accepts `"Cu"` alongside `"Zr"` and `"Ti"`, so a deployment that characterises copper can quantify against it without the caller suppressing a type error. No calculation changes: the tracer element only selects which mass fraction column is read and labels the derivation method, and the resolvability and mass-balance arithmetic is element-agnostic. Copper is weakly mobile - it sorbs to organic matter, is taken up by plants, and is applied in some fungicides - so its docstring notes that a model using it should read the tracer resolvability output, and treat a soil-derived application rate well above the operational one as a sign the tracer is not behaving conservatively

## [0.45.0](https://github.com/isometric/calculation-library/releases/tag/v0.45.0)

### Added

- `geospatial.spatial_autocorrelation`: `build_spatial_autocorrelation_report` - tabulates a `NeffResult` as a per-variable report DataFrame with an `overall` summary row (the minimum-across-variables `n_eff` the bootstrap resamples with). Takes a `variable_labels` mapping so each tested difference column is reported under a caller-chosen element label, falling back to the raw column name. Extracts the row-building previously duplicated inline across EW tracer models

## [0.44.0](https://github.com/isometric/calculation-library/releases/tag/v0.44.0)

### Added

- `enhanced_weathering.utils.statistical_checks.power_analysis`: `compute_power_analysis_paired` - power analysis for a matched-location design. Eq. 23's noise term is the variance of the within-pair difference (reporting-period minus baseline at the same location), `sigma_bl^2 + sigma_rp^2 - 2*rho*sigma_bl*sigma_rp`, in place of the unpaired `sigma_bl^2 + sigma_rp^2`. Takes the same arguments as the unpaired variant and expects a DataFrame matched by location, e.g. from `pairing.pair_locations`. Eq. 22 is identical between the two variants

### Changed

- `enhanced_weathering.utils.statistical_checks.power_analysis`: `compute_power_analysis` renamed to `compute_power_analysis_unpaired`, matching the paired/unpaired naming used elsewhere in this package. It now takes the two sampling events as separate `baseline_samples` / `reporting_period_samples` frames with unprefixed `mass_fraction_<element>` columns, in place of one location-matched frame. The two may hold different numbers of samples, which the protocol allows and an unpaired variance never required. `compute_power_analysis_paired` continues to take the matched `paired` frame. Each variant now validates its own input — `pairing.require_complete_pairs` for the matched frame, `data_cleaning.check_measured_values` per event otherwise — in place of the previous per-column `dropna`
- `enhanced_weathering.utils.statistical_checks.power_analysis`: `compute_power_analysis_unpaired`'s Eq. 23 noise term is now `sigma_bl^2 + sigma_rp^2 / k` with `k = n_reporting_period / n_baseline`, the protocol's unequal-allocation form, in place of `sigma_bl^2 + sigma_rp^2`. `n_required` is correspondingly a count of baseline samples, the reporting-period event assumed to scale at the same ratio. Equal-sized events give `k = 1` and results are unchanged, so this only bites once the two events differ in size — which the separate-frames signature above is what first allows
- `enhanced_weathering.utils.statistical_checks.power_analysis`: both variants' `elements` narrows from `Sequence[str]` to `Sequence[ElementSymbol]`, and the column name comes from `enhanced_weathering.utils.types.mass_fraction_column_name` rather than a local duplicate of it
- `enhanced_weathering.utils.statistical_checks.power_analysis`: `PowerAnalysisResult` gains `sigma_diff`, populated by the paired variant and `None` for the unpaired one, which has no matching to difference across. `n_actual` is replaced by `n_baseline` and `n_reporting_period`, since the two events can now differ in size; both equal the matched-location count for a paired design

## [0.43.0](https://github.com/isometric/calculation-library/releases/tag/v0.43.0)

### Added

- `hidden.biosphere.lidar.stats`: `canopy_height_p99` - 99th-percentile canopy height, added to every statistics path so it can be used as an upper-canopy regressor
- `hidden.biosphere.lidar.stats`: `segmented_percentile` - per-segment percentile over a concatenated pixel array, for the `reduceat`-based zonal statistics paths

## [0.42.0](https://github.com/isometric/calculation-library/releases/tag/v0.42.0)

### Added

- `biosphere.monte_carlo.plot_geometry`: `PlotShape`, `plot_area_m2`, `PlotDeformer`, `build_plot_deformer` - decomposes field plot georeferencing error into a GPS centre offset and angularly-autocorrelated shape deformation, replacing independent per-vertex jitter

### Changed

- `biosphere.allometric_equations.wood_density`: `tree_type_to_species` now also accepts the colon-delimited qualifier spelling of a tree type (`taxonomic_rank:species:cecropia_obtusa`) alongside the enum spelling (`SPECIES_CECROPIA_OBTUSA`). Measurements read through the activity model carry the qualifier form, so a model consuming them previously raised `KeyError` on every identified species
- `biosphere.monte_carlo.field_plot`: `FieldPlot` takes `area_m2` in place of `plot_size_m`, so per-hectare biomass comes from a plot's surveyed polygon rather than a nominal side length. Plots are rarely laid out to their nominal dimension, that dimension is sometimes absent altogether, and where it disagrees with the polygon it can be badly wrong - a plot labelled "40 m square" enclosing 3000 m² would have had its per-hectare biomass overstated by nearly a factor of two. `plot_area_m2` remains for deriving an area from a nominal dimension when no polygon is available

## [0.41.0](https://github.com/isometric/calculation-library/releases/tag/v0.41.0)

### Changed

- `biosphere.allometric_equations.wood_density`: refreshed the wood density table from Mombak's July 2026 dataset. Adds `Adenanthera pavonina`, `Aegiphila sellowiana`, `Persea americana`, `Sapium argutum` and `Talisia esculenta`, and restates the species already present at the source dataset's full precision rather than the previously truncated values. `Alexa grandiflora` (0.36717 -> 0.66) and `Hymenaea intermedia` (0.85 -> 0.82) are corrections of values that matched no entry in the Global Wood Density Database; both new values reproduce its species mean exactly. Above-ground biomass quantified for plots containing either species changes accordingly. `Symmeria paniculata` is absent from the new dataset and keeps its existing values, as it remains a supported tree type

## [0.40.0](https://github.com/isometric/calculation-library/releases/tag/v0.40.0)

### Added

- `enhanced_weathering.utils.application_rate_derivation`: `derive_application_rate_from_post_application_samples`, `derive_application_rate_from_tracer`, `pool_cations_as_charge_equivalents`, `DerivedApplicationRate` - back-calculates the application rate a soil received, from either post-application (BLP) cation enrichment or an immobile tracer mass balance. Both return `DerivedApplicationRate`. The BLP route pools calcium, magnesium, sodium and potassium onto a charge-equivalent basis and inverts the mass balance once rather than per cation. Both routes take `paired` (default `True`): paired designs inner-join the two sampling events by location and resample them through shared bootstrap indices, unpaired designs resample each event independently and may have different sample counts per event. `DerivedApplicationRate` reports `n_baseline`, `n_post_application` and `paired`. Raises on fewer than `min_samples` paired locations or samples per event, a null measurement in either event or in the feedstock, or no physically valid replicate
- `enhanced_weathering.utils.statistical_checks.application_rate`: `resolve_application_rate`, `ApplicationRateDecision` - selects the application rate to quantify with and builds the report row for that decision. The operational rate is accepted when it falls within `n_std` standard deviations of the soil-derived distribution, otherwise the lower of the two is used. The distribution is summarised by its median, reported alongside `p16` / `p84`; the median both decides which rate is lower and is what `rate_kg_ha` returns, as a scalar. Raises on a non-positive operational rate, a non-positive `n_std`, or a soil-derived distribution with no finite replicate
- `enhanced_weathering.utils.pairing`: `require_complete_pairs`, `paired_column_names` - raises if any location in a `pair_locations` result is missing a measurement in the given columns, reporting the null count per column; and builds the `baseline_` / `reporting_period_` column names `pair_locations` produces
- `enhanced_weathering.utils.resampling`: `resample_columns_together`, `resample_events_together`, `SamplingEventMeans` - bootstraps one or two sampling events across one or more value columns at once, sharing one set of indices across the columns of an event so a replicate is a coherent draw of samples. `paired` decides whether the two events also share that set. A single column is the one-entry case; `SamplingEventMeans.column` unpacks it back to a `(baseline, reporting_period)` pair
- `enhanced_weathering.utils.data_cleaning`: `check_measured_values` - returns one column of measurements, raising if the column is absent, has no rows, or carries a null. The raise-loudly counterpart to `null_filter`, and the unpaired counterpart to `pairing.require_complete_pairs`
- `enhanced_weathering.utils.conversions`: `compute_soil_mass_kg_ha`, `compute_feedstock_soil_mass_ratio` - soil mass per hectare to a given depth, and the rock-to-soil mass ratio `r = R / (BD * D * 10000)` built on it

### Changed

- `enhanced_weathering.utils.statistical_checks.power_analysis`: `compute_power_analysis`'s `effective_application_rate_kg_ha` and `bulk_density_kg_m3` each accept a bootstrap distribution as well as a scalar. Eq. 22 is evaluated per replicate, and `PowerAnalysisResult` gains `delta_mg_kg_p16` / `delta_mg_kg_p84`. `delta_mg_kg` is now the median of that distribution; for scalar inputs all three are equal and results are unchanged. Two distributions are paired replicate-for-replicate rather than crossed and must be the same length; mismatched lengths raise, as does a rate and bulk density forming no finite mass ratio
- `enhanced_weathering.utils.statistical_checks.weathering_signal`: `infer_post_application_concentrations`'s `application_rate_kg_ha` and `bulk_density_kg_m3` each accept a bootstrap distribution as well as a scalar. The mixing formula is applied at each replicate and the inferred concentrations pooled. Scalar inputs return one concentration per baseline sample, unchanged

### Removed

- `enhanced_weathering.utils.resampling`: `resample_dataframe_paired` and `resample_dataframe_unpaired` - superseded by `resample_events_together`, which covers one column or many through the same entry point. Callers pass the values as a one-entry mapping and unpack with `SamplingEventMeans.column`. Note the replacement is not numerically identical: it draws indices rather than values, keeps float64 where `resample_dataframe_unpaired` cast to float32 via `resample_mean`, and takes the mean rather than a `nan`-skipping mean, so a null in the input now propagates instead of being silently dropped

## [0.39.4](https://github.com/isometric/calculation-library/releases/tag/v0.39.4)

Dependency version update.

## [0.39.3](https://github.com/isometric/calculation-library/releases/tag/v0.39.3)

Dependency version update.

## [0.39.2](https://github.com/isometric/calculation-library/releases/tag/v0.39.2)

Dependency version update.

## [0.39.1](https://github.com/isometric/calculation-library/releases/tag/v0.39.1)

Dependency version update.

## [0.39.0](https://github.com/isometric/calculation-library/releases/tag/v0.39.0)

### Added

- `enhanced_weathering.utils.statistical_checks._significance`: `run_paired_significance_test` / `run_unpaired_significance_test` and `PairedSignificanceTest` / `UnpairedSignificanceTest` - shared primitives that run a paired or unpaired significance test, picking a parametric or rank-based implementation (paired t-test / Wilcoxon signed-rank, Welch's t-test / Mann-Whitney U) from a Shapiro-Wilk normality check. Now used by both the control-correction and weathering-signal tests

### Changed

- `enhanced_weathering.utils.statistical_checks.control_correction_significance`: `check_background_weathering_significance_paired` / `check_background_weathering_significance_unpaired` are now normality-selected (paired t-test / Wilcoxon signed-rank, Welch's t-test / Mann-Whitney U) rather than always using a t-test, and return `ControlPlotChangeSignificanceTest` / `UnpairedControlPlotChangeSignificanceTest`. The always-t-test variants and `ControlPlotAlkalinityChangeSignificanceTest` have been removed; the paired result reports `n_pairs` in place of `n_baseline_samples` / `n_reporting_period_samples`, and neither result carries the `paired` flag
- `enhanced_weathering.utils.statistical_checks.control_correction_significance`: `check_background_weathering_significance_unpaired` is now two-sided, matching the paired variant. Background cation change is real in either direction, and the previous one-sided depletion hypothesis reported a significant control *enrichment* as "not significant" rather than as what it was. Whether an enrichment may increase CDR remains the `floor_at_zero` decision on `apply_control_correction_delta_*`
- `enhanced_weathering.utils.tracer`: `compute_fraction_dissolved` gains `control_correction_delta_mg_kg` (default 0.0), implementing `f_d = ((1+m)/m) * (C_post * cc - C_rp - delta) / C_feed`. `control_correction_ratio` is unchanged and still defaults to neutral, so existing results are bit-identical
- `enhanced_weathering.utils.conversions`: `Cation` now includes `"Na"` and `"K"` alongside `"Ca"` and `"Mg"`, and `_cation_to_charge` returns 1 for them since sodium and potassium are monovalent — they capture half the CO2 per mole of a divalent cation. Existing calcium and magnesium arithmetic is unchanged, so callers passing only those are unaffected
- `calculations.enhanced_weathering_cdr_tracer_ti_treatment_only`: quantifies sodium and potassium in addition to calcium and magnesium. Requires `mass_fraction_na` and `mass_fraction_k` in the soil and feedstock inputs, and changes credited CDR for this calculation

### Deprecated

- Ratio-based control correction is sunset in favour of the additive delta. `compute_control_correction_ratio`, `bootstrap_control_correction_ratios` and `compute_fraction_dissolved`'s `control_correction_ratio` argument still work but are deprecated

### Fixed

- `enhanced_weathering.utils.control_correction`: `apply_control_correction_delta_paired` computed its delta as `reporting_period - baseline`, the opposite sign to the unpaired variant and to the documented application. With the default `floor_at_zero=True` this clamped the correction to zero in exactly the case it exists for - a depleted control - while applying a spurious CDR reduction when the control was *enriched*. The delta is now `baseline - reporting_period` in both variants, so a positive delta always means background loss and always reduces CDR

## [0.38.2](https://github.com/isometric/calculation-library/releases/tag/v0.38.2)

Dependency version update.

## [0.38.1](https://github.com/isometric/calculation-library/releases/tag/v0.38.1)

Dependency version update.

## [0.38.0](https://github.com/isometric/calculation-library/releases/tag/v0.38.0)

### Added

- `biosphere.reforestation_dynamic_baselining.spectral_matcher`: `SpectralMatcher` - donor-pixel k-NN spectral matching for dynamic baselining
- `biosphere.reforestation_dynamic_baselining.performance_benchmark_calculator`: `perform_paired_ttest`, `calculate_intra_plot_difference` - paired t-test and intra-plot difference benchmarking against matched donor pixels
- `biosphere.utils.raster`: `sample_raster_vectorized` - vectorized raster sampling at WGS84 coordinates

## [0.37.7](https://github.com/isometric/calculation-library/releases/tag/v0.37.7)

### Added

- Added `scikit-learn` as a dependency

## [0.37.6](https://github.com/isometric/calculation-library/releases/tag/v0.37.6)

Dependency version update.

## [0.37.5](https://github.com/isometric/calculation-library/releases/tag/v0.37.5)

Dependency version update.

## [0.37.4](https://github.com/isometric/calculation-library/releases/tag/v0.37.4)

Dependency version update.

## [0.37.3](https://github.com/isometric/calculation-library/releases/tag/v0.37.3)

Dependency version update.

## [0.37.2](https://github.com/isometric/calculation-library/releases/tag/v0.37.2)

Dependency version update.

## [0.37.1](https://github.com/isometric/calculation-library/releases/tag/v0.37.1)

### Changed

- Consolidated the `Np1DArray` / `Np2DArray` numpy array type aliases into `utils.types`, removing the duplicate definitions previously in `biosphere.types` and `enhanced_weathering.utils.types`.

## [0.37.0](https://github.com/isometric/calculation-library/releases/tag/v0.37.0)

### Added

- `enhanced_weathering.utils.conversions`: `convert_cation_kg_to_charge_equivalents` - cation mass to charge-equivalent conversion

## [0.36.0](https://github.com/isometric/calculation-library/releases/tag/v0.36.0)

### Added

- `enhanced_weathering.utils.statistical_checks.multiple_testing`: `benjamini_hochberg`, `permutation_test_median_difference` - promoted shared multiple-testing correction and permutation-test utilities previously duplicated across models and geospatial `spatial_autocorrelation`

## [0.35.0](https://github.com/isometric/calculation-library/releases/tag/v0.35.0)

### Changed

- Renamed baseline / reporting-period column prefixes from `bl_`/`rp_` to `baseline_`/`reporting_period_` across pairing and all consumers. No aliases retained.

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
