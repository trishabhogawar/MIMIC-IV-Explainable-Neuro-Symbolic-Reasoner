**Neuro-Symbolic ICU Mortality Prediction (24h)**

Purpose:
Early mortality risk prediction from the first 24 hours of ICU data, combining tabular ML (LightGBM, XGBoost) with a neuro-symbolic causal layer (rules + DAG + counterfactuals). Backend only.

Key features:
- Uses only first 24h to avoid leakage and keep predictions actionable.
- Models: LightGBM and XGBoost, blended for best AUROC/PR-AUC and stable calibration.
- Explanations: global SHAP and per-patient reasoning.
- Causal layer: medical rules + causal graph with do() counterfactuals.

Run from scratch (end to end):

**Build data and features:**
- python build_day1_dataset.py
- python build_day1_postprocess.py
- python build_clinical_scores.py
- python extract_meds_24h.py
- python make_med_flags.py
- python build_pathway_features.py
  
**Train and blend:**
- python train_lgbm_tabular_cv.py
- python train_xgb_tabular_cv.py
- python ensemble_blend.py --lgbm_dir models_lgbm_cv --xgb_dir models_xgb_cv --out_dir models_blend_cv
  
**Evaluate and explain:**
- python evaluation_metrics.py --probs models_blend_cv/oof_probs.npy --labels y_labels.npy --out_dir reports
- python shap_demo.py --model_dir models_lgbm_cv --X X_tab.npy --feature_names feature_names_tab.json --out_dir reports
- python explain_one.py --idx 123 (any patient id here)
  
**Per-patient neuro-symbolic reasoning:**
- python reason_case.py --idx 123 (any aptient id here)
