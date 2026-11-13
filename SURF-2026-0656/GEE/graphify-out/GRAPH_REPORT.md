# Graph Report - GEE  (2026-10-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 402 nodes · 784 edges · 18 communities (14 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17

## God Nodes (most connected - your core abstractions)
1. `compute_credibility()` - 12 edges
2. `EfficientUNet` - 12 edges
3. `export_training_patch()` - 11 edges
4. `export_single_patch()` - 10 edges
5. `export_single_patch()` - 10 edges
6. `main()` - 10 edges
7. `build_activity_feature_stack()` - 9 edges
8. `build_activity_feature_stack()` - 9 edges
9. `main()` - 9 edges
10. `build_activity_feature_stack()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `try_unet_inference()` --calls--> `EfficientUNet`  [EXTRACTED]
  对抗/对抗.py → train_unet_effb0.py
- `evaluate_redteam()` --calls--> `EfficientUNet`  [EXTRACTED]
  对抗/eval_redteam_attack.py → train_unet_effb0.py
- `make_obs()` --calls--> `ProxyObservation`  [EXTRACTED]
  demo_run.py → greenwash_scorer.py
- `build_observations_from_zonal_stats()` --calls--> `ProxyObservation`  [EXTRACTED]
  demo_run.py → greenwash_scorer.py

## Import Cycles
- None detected.

## Communities (18 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (16): build_observations_from_zonal_stats(), make_obs(), BaselineStats, batch_score_to_dataframe(), calibrate_thresholds(), check_contradiction(), Claim, ClaimCategory (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (18): build_activity_feature_stack(), compute_ndvi(), export_activity_features(), export_to_drive(), export_training_patch(), get_aoi_from_bbox(), get_aoi_from_place(), get_combined_industrial_mask() (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (15): colorize_mask(), compute_class_weights(), compute_miou(), create_data_from_directory(), create_data_loaders(), _double_conv(), EfficientUNet, main() (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (16): compute_activity_indices(), compute_spectral_stats(), load_model_class(), load_unet(), main(), predict_batch(), save_visualization(), write_excel_report() (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (16): build_activity_feature_stack(), compute_ndvi(), export_activity_features(), export_to_drive(), export_training_patch(), get_aoi_from_bbox(), get_aoi_from_place(), get_combined_industrial_mask() (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (11): export_single_patch(), get_corine_industrial_mask(), get_s2_projection_from_image(), init_gee(), main(), export_single_patch(), get_corine_industrial_mask(), get_s2_projection_from_image() (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (10): compute_anomaly_scores(), compute_spectral_indices(), discover_files(), extract_activity_metrics(), extract_power_stats(), load_tile(), main(), try_unet_inference() (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.19
Nodes (8): build_activity_feature_stack(), compute_ndvi(), get_nighttime_lights(), get_s2_projection(), get_sentinel2_for_ndvi(), get_thermal(), init_gee(), main()

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (6): main(), scan_mask_classes(), main(), slice_one_image(), count_pixels(), main()

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (8): colorize_mask(), compute_class_weights(), create_data_from_directory(), create_data_from_multi_dirs(), create_data_loaders(), main(), predict_full_image(), visualize_results()

### Community 10 - "Community 10"
Cohesion: 0.20
Nodes (8): colorize_mask(), compute_class_weights(), create_data_from_directory(), create_data_from_multi_dirs(), create_data_loaders(), main(), predict_full_image(), visualize_results()

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (8): compute_per_class_metrics(), load_masks(), load_model_class(), main(), plot_confusion_matrix(), run_inference(), visualize_predictions(), write_eval_report()

### Community 14 - "Community 14"
Cohesion: 0.20
Nodes (4): build_criterion(), compute_miou(), DiceLoss, train_unet()

### Community 15 - "Community 15"
Cohesion: 0.20
Nodes (4): build_criterion(), compute_miou(), DiceLoss, train_unet()

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EfficientUNet` connect `Community 2` to `Community 6`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07568027210884354 - nodes in this community are weakly interconnected._
- **Why does `EfficientUNet` connect `Community 13` to `Community 10`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09878048780487805 - nodes in this community are weakly interconnected._
- **Why does `EfficientUNet` connect `Community 12` to `Community 9`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.07435897435897436 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.08392603129445235 - nodes in this community are weakly interconnected._