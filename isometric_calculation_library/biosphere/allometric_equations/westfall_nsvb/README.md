# NSVB above-ground biomass allometry

Above-ground biomass for United States tree species, from the USDA Forest Service national-scale volume and biomass (NSVB) framework.

Source: Westfall JA, Coulston JW, Gray AN, Shaw JD, Radtke PJ, Walker DM, Weiskittel AR, MacFarlane DW, Affleck DLR, Zhao D, Temesgen H, Poudel KP, Domke GM, Tyrrell ML (2024), "A national-scale tree volume, biomass, and carbon modeling system for the United States", USDA Forest Service General Technical Report WO-104. <https://research.fs.usda.gov/treesearch/66998>

## Scope

Two of NSVB's components are implemented, which together make total above-ground biomass:

| Component | NSVB name | Coefficients | Fit statistics |
| --- | --- | --- | --- |
| Wood and bark, excluding foliage | `TT_WDBK_DW_ADJ` | Table S8 | Tables S12, S13 |
| Foliage | `FOL_DW` | Table S9 | Tables S12, S13 |
| Their sum, total AGB | `TT_DW_ADJ` | — | Tables S12, S13 |

Out of scope: volume, bark and branch components taken separately, coarse roots, and the woodland junipers and pinyons (Jenkins group 10), which NSVB handles with a separate diameter-at-root-collar formulation. Asking for a woodland species raises `KeyError`.

## Layout

| File | Contents |
| --- | --- |
| `models.py` | The five published model forms (Eqs 1–5) and the `NsvbComponentModel` / `NsvbModel` dataclasses that carry fitted coefficients |
| `coefficients.py` | Turns a species code into a fitted model, choosing between the coefficient sets NSVB publishes for it — `get_nsvb_model`, `published_divisions`, `create_nsvb_model_generator` |
| `species.py` | FIA species reference lookup by code or binomial name — `get_species`, `find_species_code` |
| `tables.py` | Cached reads of the packaged CSVs |
| `data/` | The coefficient, fit-statistic and species tables, transcribed from the paper's supplementary material |

## Units

The published coefficients are only valid in imperial units: diameter in inches, height in feet, dry weight in pounds. `compute_biomass_pounds` exposes that native form for cross-checking against FIA output; everything else takes centimetres and metres and returns tonnes, converting internally.

## Coefficient selection

NSVB publishes several coefficient sets per component, at different levels of specificity, and `get_nsvb_model` takes the most specific one available for the species. It records which level that was in `coefficient_level`:

1. the species-and-ecodivision fit, where one is published;
2. otherwise the species' ecodivision-pooled entry — "if a species occurs in an ecodivision not explicitly listed, the entry having no ecodivision noted is used";
3. otherwise the fit for the species' Jenkins species group, for species with too few sampled trees to model individually.

The two components resolve independently, because NSVB models more species for foliage than for wood and bark: one component can come from a species-level fit while the other falls back to a Jenkins group.

Stand origin only enters for slash pine (SPCD 111) and loblolly pine (SPCD 131), the two species NSVB fits separately for natural and planted stands. `DEFAULT_STAND_ORIGIN` is `PLANTED`; pass `StandOrigin.NATURAL` for naturally regenerated stands.

## Uncertainty

**Residual error** is the published `Sigma`. NSVB fits by weighted nonlinear least squares with `1/D²` weights, so residual variance is proportional to `D²` and `Sigma` is a residual standard deviation *per inch of diameter* — a tree of diameter `D` has residual SD `Sigma × D` pounds, not a constant `Sigma`. AGB uses the `TT_DW_ADJ` statistic for the summed component rather than combining the two parts, which would need an assumption about their correlation. A handful of species were fitted on a single felled tree and so have no residual degrees of freedom; NSVB publishes `Sigma` as infinite for those, and they fall back to the national `Sigma` rather than making every downstream standard error infinite.

**Parameter error** is carried by `create_nsvb_model_generator`, which resamples the alternative ecodivision coefficient sets NSVB publishes for the species. NSVB reports no coefficient standard errors or covariance, so there is no replicate table to draw from as there is for Chave et al. 2014. Validated against a case-resampling bootstrap refit of the Radtke et al. 2023 open felled-tree data, this proxy overstates the true parameter sampling spread by roughly 2.6× for wood and bark and 3.0× for foliage — regional allometric differences are real signal, not sampling noise, so the proxy errs wide. It also has nothing to resample for the many species with only a pooled fit, where the generator returns the same model every time. Residual error dominates either way: on a 30 cm / 20 m shortleaf pine, 12.1% CV from residuals against 1.4% from the generator.

## Why the equations follow the paper and not `nsvb.R`

NSVB ships a reference R implementation, `nsvb.R`, alongside the report. The two disagree on the parenthesisation of two model forms, and this package follows the paper in both cases. Both choices are pinned by tests in `tests/test_models.py`.

**Eq 2, the segmented form.** Below the segmentation point `k` the paper's lower branch uses the exponent `b`; the script uses `b1`, the exponent of the upper branch. The upper branch carries a `k^(b − b1)` factor whose entire purpose is to make the two branches meet at `k` — with `b1` on both sides that factor introduces a step discontinuity at the segmentation point instead of removing one. For slash pine in natural stands the script's version jumps by a factor of 1.6 at the 9-inch segmentation point.

**Eq 3, the continuously variable form.** The paper's diameter exponent is `a1 · (1 − e^(−bD))^c1`, which rises with diameter towards an asymptote of `a1`. The script computes `(a1 · (1 − e^(−bD)))^c1`, which instead asymptotes to `a1^c1`. Since `c1` is well below 1 for every fitted species, that flattens the large-tree exponent badly — for slash pine plantations, `a1 = 1.85` and `c1 = 0.33` give an asymptotic exponent of 1.23 rather than 1.85, and a 60 cm tree comes out at 14% of the paper's biomass.

The check that settles it: refit nothing, and simply evaluate each form on the felled trees its own fit was built from (Radtke et al. 2023, the open subset of NSVB's training data), then compare the weighted residual spread to the `Sigma` the paper publishes for that species and component. The paper's forms reproduce the published `Sigma` to within 6%. The script's forms are off by up to 4× for the species that use Eqs 2 and 3 — a fit cannot have been estimated with the form that reproduces its own residuals that badly.

Eqs 1, 4 and 5 — which between them cover most species — are unambiguous, and the two implementations agree there.

## Naming a species

`species` means the full binomial here, spelled as `wood_density` spells it — `"Pinus echinata"`, capitalised genus, lowercase epithet, one space. `find_species_code` takes it in that form, the same shape `wood_density.get_wood_density` takes, so a tree type read off activity data reaches either lookup through the one existing adapter:

```python
species_code = find_species_code(
    tree_type_to_species("taxonomic_rank:species:quercus_bicolor")
)  # 804
```

`NsvbSpecies.specific_epithet` is the epithet alone, as FIA stores it. FIA also codes genera, writing them `"Abies spp."`, so a `taxonomic_rank:genus:` qualifier resolves too — `find_species_code("Abies")` and `find_species_code("Abies spp.")` both give SPCD 10.

## Known gaps

- The two sides of that join answer to different taxonomic authorities, so the misses are systematic rather than random. Species names in the backend are validated through GBIF, which returns the *accepted* binomial; FIA's reference carries its own, sometimes superseded. Chestnut oak is the live case — GBIF gives *Quercus montana*, FIA says *Quercus prinus*, and `find_species_code("Quercus montana")` raises rather than returning SPCD 832. Wiring NSVB to backend activity data needs a synonym table.
- Ten binomials in the reference carry more than one species code, usually a species and a variety FIA still codes separately, so a name alone does not always identify a species. `find_species_code` raises on those rather than picking one, because the codes are not interchangeable: *Pinus elliottii* is both slash pine (111) and Honduras pine (144), 26% apart in wood specific gravity, and *Abies lasiocarpa* is both corkbark (18) and subalpine fir (19), 16% apart. Callers that hit one have to choose a code and pass it to `get_species` directly.
- Only a handful of the tree types the backend currently defines are US species, because that vocabulary was built for Amazonian and southern-African projects: 9 of its 512 species-level qualifiers and 3 of its 160 genus-level ones resolve to an SPCD today. Nothing is wrong with the join; a US project would simply need its species added first.
- The reference is FIA's, not a taxonomic authority: SPCD 813 is cherrybark oak (*Quercus pagoda*) and SPCD 830 is pin oak (*Q. palustris*), which are easy to transpose when mapping inventory codes.
