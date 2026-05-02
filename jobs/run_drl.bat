@echo off

for %%s in (52 63 69 81) do (
    echo ============================================
    echo Running seed %%s ...
    echo ============================================

    python -m scripts.drl_latent_regime_training ^
    --output-dir "data/baselines/drl_latent_regime_no_dir_%%s" ^
    --transformer-root "data/transformer_npy_no_dir" ^
    --train-timesteps 500000 ^
    --seed %%s
)

echo All runs finished.
pause