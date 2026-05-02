@echo off

python scripts/aggregate_single_runs.py --run-dirs data/baselines/drl_latent_regime_no_dir_42 data/baselines/drl_latent_regime_no_dir_52 data/baselines/drl_latent_regime_no_dir_63 data/baselines/drl_latent_regime_no_dir_69 data/baselines/drl_latent_regime_no_dir_81 --output-dir data/baselines/no_dir_aggregated_tdqn --variants gating,no_gating,no_sentiment

echo Aggregation Done
pause