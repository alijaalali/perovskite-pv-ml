# Validation, noise ceilings and design effects in machine learning of perovskite solar cell stability (Perovskite Database Project)

Code and result files for the manuscript "Validation, Noise Ceilings and Design Effects in Machine Learning of Perovskite Solar Cell Stability". The study benchmarks prediction of stability (censored lifetimes under ISOS protocols) with author-aware validation, noise ceilings, censoring-weighted and landmark metrics, matched within-paper design effects and a selection analysis. A shorter analysis of power conversion efficiency from composition (39,780 cells) serves as a comparator and is summarised in the Supporting Information; scripts of the efficiency analysis that are not reported in the manuscript are kept for completeness.

## Scripts

Every script locates `data/` (and writes `figures/`) relative to its own file, so run the scripts from a clone of this repository with the scripts and `data/` in the same directory (Python 3.11 and the packages under Requirements). Several stability scripts load the feature-building head of `stability_enriched_features.py`, `within_paper_contrasts.py`, `stability_author_heldout.py`, `stability_benchmark_more_models.py` or `stability_ipcw_horizons.py` from the same directory, so keep all files together. The heavy stability scripts were run as batch jobs; the arguments used were:

- `stability_tuned_headline.py [paper|temporal]` - tuned headline run for one protocol (without an argument both protocols run)
- `stability_tuned_robustness.py author last_author|purged_any_author`, `... seed K` (K = 1-4), `... comp paper|temporal`, `... size K` (K = 1-3) - author-aware, repeated-fold, composition-ablation and training-size runs
- `MV_CHUNK=k python sensitivity_multiverse_tuned.py` for k = 0-5 (one sixth of the 360 paths each; `quick` as the first argument gives a reduced test run)
- `stability_ipcw_horizons.py`, `stability_ipcw_decomposition.py`, `stability_ipcw_strata.py` run without arguments

Run order:

Efficiency analysis (composition features)

1. `phase1_pipeline.py` - parse the A-, B- and C-site fields into elemental formulas, Magpie features, chemistry subgroups and families; baseline split check
2. `phase2_pipeline.py` - split diagnosis, bootstrap intervals, subgroup diagnosis, multi-model panel
3. `phase2b_multipartition.py` - 40 independent family partitions against matched random splits
4. `phase4_multimodel_multipartition.py` - the same partitions for XGBoost, random forest, Ridge and ElasticNet
5. `phase6_mlp_multipartition.py` - the same partitions for a neural network
6. `phase8_ridge_standardization_check.py`, `phase8_elasticnet_standardization_check.py`, `phase8_tree_scale_invariance_check.py` - standardization and scale-invariance checks
7. `phase7_ridge_eigenstructure.py`, `phase4_extrapolation_distance.py` - condition number and extrapolation-distance tests
8. `phase4_dedup_benchmark.py` - deduplicated benchmark
9. `phase7_duplicate_rootcause.py` - within-composition variance and its relation to device fields
10. `phase4_device_ablation.py` - composition-plus-device feature ablation
11. `phase7_temporal_split.py` - temporal generalization
12. `phase7_shap_physics.py` - SHAP importances of the deduplicated model

Stability analysis

13. `build_stability_table.py` - stability table from the raw export: targets, censoring status, composition and device features
14. `stability_benchmark.py` - retention regression, retained-at-least-80% classification and censoring-aware survival models under six validation protocols; feature-group ablation
15. `stability_benchmark_extra_models.py` - thirteen further regression models, twelve classifiers, six survival models
16. `stability_benchmark_more_models.py` - ten further models on base and enriched features
17. `stability_noise_ceiling.py` - noise ceilings for stability and efficiency
18. `stability_feature_audit.py` - coverage of every database column among the stability cells
19. `stability_enriched_features.py` - enriched feature groups and paired gains
20. `sensitivity_multiverse.py` - 720-path sensitivity analysis of the enrichment gain
21. `within_paper_contrasts.py` - matched within-paper design contrasts
22. `prospective_contrast_check.py` - contrasts estimated up to 2018 and tested in 2019-2020 papers
23. `pairwise_within_paper_learning.py`, `pairwise_trivial_baselines.py` - within-paper pairwise ranking and trivial baselines
24. `design_effects_selection_and_gap_map.py` - selection model for design effects and map of untested contrasts
25. `fetch_author_metadata.py` - author lists for each paper from OpenAlex
26. `stability_author_heldout.py` - paper-, last-author-, first-author- and purged any-author-held-out validation
27. `stability_all_models_enriched.py`, `stability_nested_tuning.py` - all regression models with base and enriched features; nested hyperparameter search
28. `stability_ipcw_horizons.py` - censoring-weighted and landmark discrimination at fixed horizons
29. `stability_alt_targets.py`, `stability_t80_and_encoding.py`, `stability_reporting_value.py` - alternative targets, T80 and composition-encoding checks, value of individual reporting fields
30. `noise_ceiling_validation.py`, `simulation_validation.py` - nested-subset ceilings and simulation checks of the ceiling, concordance and selection estimators
31. `contrast_meta_analysis.py` - random-effects meta-analysis and small-study checks of the matched contrasts
32. `efficiency_partition_blocks.py`, `efficiency_device_features.py` - dependence of the family partitions; efficiency with device and process features and ceilings
33. `make_figures.py`, `make_figures_extra.py` - figures, tables and number macros
34. `stability_tuned_headline.py`, `stability_ipcw_decomposition.py`, `high_r2_mechanisms.py`, `lead_free_contrast.py`, `within_paper_contrasts_strict.py` - tuned headline gains with paired intervals; cohort decomposition and simulation of horizon AUC; targets and features that give very high R2; lead-free contrast; strict matching of contrasts
35. `stability_tuned_robustness.py`, `stability_ipcw_strata.py`, `pairwise_prior_baselines.py`, `sensitivity_multiverse_tuned.py` - tuned author-aware gains, repeated fold assignments and composition ablation; horizon AUC within strata; trivial ranking baselines; reduced tuned multiverse
36. `efficiency_canonical_split.py` - refit of the primary single split with fixed settings and seed (reproducibility check); the training-size control is `stability_tuned_robustness.py size K`
37. `stability_conformal.py` - split-conformal 90% prediction intervals (paper-grouped and cell-level calibration)
38. `make_toc_graphic.py` - table-of-contents graphic with the headline results

## Data

Input tables

- `perovskite_raw.csv` - unchanged Perovskite Database Project export (version 1.0.1, public, CC BY 4.0)
- `pdp_featurized.parquet`, `pdp_deduplicated.parquet` - cleaned efficiency tables with Magpie features, and the version with duplicate compositions removed
- `stability_table.parquet`, `stability_table_flow.json` - stability table and its row-count flow
- `paper_authors_openalex.json` - author lists from OpenAlex

Efficiency results

- `phase1_sanity_check.json`, `hyperparam_sweep.json` - split check and hyperparameter sweep of `phase1_pipeline.py`
- `leakage_diagnosis.json`, `split_bootstrap_ci.json`, `subgroup_diagnosis.json`, `multimodel_panel.json` - outputs of `phase2_pipeline.py`
- `multipartition_robustness.json`, `multimodel_multipartition_robustness.json`, `mlp_multipartition_robustness.json` - results over the 40 family partitions
- `ridge_standardization_check.json`, `elasticnet_standardization_check.json`, `tree_scale_invariance_sanity_check.json` - standardization and scale-invariance checks
- `ridge_eigenstructure_mechanism.json`, `extrapolation_distance_mechanism.json` - eigenstructure and extrapolation-distance tests
- `dedup_benchmark_comparison.json`, `duplicate_disagreement_rootcause.json`, `temporal_generalization.json`, `composition_plus_device_ablation.json`, `shap_feature_physics.json` - deduplicated benchmark, within-composition variance, temporal split, device ablation, SHAP importances
- `efficiency_canonical_split.json`, `efficiency_partition_blocks.json`, `efficiency_partitions_flat.csv`, `efficiency_device_features.json`, `efficiency_ceilings.json` - canonical split refit, partition dependence, device and process features, efficiency noise ceilings

Stability results

- `stability_benchmark.json`, `stability_benchmark_extra_models.json`, `stability_benchmark_more_models.json`, `stability_benchmark_all_models.csv`, `stability_benchmark_matrix.csv` - benchmark across models and protocols
- `stability_noise_ceiling.json`, `noise_ceiling_validation.json`, `simulation_validation.json` - noise ceilings and their validation
- `stability_column_audit.csv`, `stability_enriched_features.json`, `stability_all_models_enriched.json`, `stability_nested_tuning.json` - feature audit, enriched features, nested tuning
- `stability_author_heldout.json`, `stability_tuned_headline_paper.json`, `stability_tuned_headline_temporal.json`, `stability_tuned_robustness_*.json` - author-held-out validation, tuned gains, seed, size, author and composition variants
- `sensitivity_multiverse.csv`, `sensitivity_multiverse_summary.json`, `sensitivity_multiverse_tuned_c*.csv`, `sensitivity_multiverse_tuned_summary_c*.json` - multiverse analyses
- `stability_ipcw_horizons.json`, `stability_ipcw_decomposition.json`, `stability_ipcw_strata.json`, `stability_conformal.json` - censoring-weighted discrimination and conformal intervals
- `stability_alt_targets.json`, `stability_t80_and_encoding.json`, `stability_reporting_value.json`, `high_r2_mechanisms.json` - alternative targets, encodings, reporting fields, high-R2 cases
- `within_paper_contrasts.json`, `within_paper_contrasts_window*.csv`, `within_paper_contrasts_strict_window01.csv`, `within_paper_fixed_effects.csv`, `prospective_contrasts.csv`, `lead_free_contrast.json` - matched within-paper contrasts
- `contrast_meta_analysis.csv`, `contrast_meta_regression.csv`, `contrast_meta_diagnostics.json` - meta-analysis of contrasts
- `design_effects_selection_model.csv`, `design_effects_selection_pooled.json`, `untested_contrast_map.csv`, `untested_contrast_map_check.csv` - selection model and map of untested contrasts
- `pairwise_learning.csv`, `pairwise_learning_nofresh.csv`, `pairwise_trivial_baselines.json`, `pairwise_prior_baselines.json` - within-paper pairwise ranking

## License

Code: MIT License (see `LICENSE`). The Perovskite Database Project export in `data/perovskite_raw.csv` is distributed under its original CC BY 4.0 license; please cite Jacobsson et al., Nat. Energy 2022, 7, 107.

## Requirements

Python 3.11; pymatgen; matminer; scikit-learn; XGBoost; LightGBM; CatBoost; PyTorch (CPU); NumPy; pandas; SciPy; Matplotlib (SHAP is needed only for `phase7_shap_physics.py`). Exact versions are listed in the Supporting Information ("Software versions").
