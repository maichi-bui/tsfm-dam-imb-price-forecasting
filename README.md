# Chronos-2 Inference and Hyperparameter Tuning

This directory contains code for performing inference and hyperparameter tuning for electricity price forecasting using the Chronos-2 model, targeting the Belgian Day-Ahead Market (DAM) and Imbalance Market. It can be adapted to adjacent markets and bidding zones.

---

## Zero-shot Inference

```bash
cd scripts/
pip install -r requirements.txt
python3 run_zeroshot.py --config "./config_dam.json" --mode AR --context-length 1024 --add-temporal-features --forecast-end "2023-03-31"
python3 run_zeroshot.py --config "./config_imb.json" --mode ARX --context-length 2048 --add-temporal-features
```

---

## LoRA Fine-Tuning with Covariates

Fine-tuning uses the native **`pipeline.fit()` API** from `chronos-forecasting`, which supports past and future covariates natively. No manual training loop is required.

### Covariate design

| Group | Columns | Role |
|---|---|---|
| Physical | `Solar`, `Wind`, `Load`, `Temp`, `Hum` | Past history + day-ahead forecasts (future-known) |
| Temporal | `Week_cos`, `Week_sin`, `Day_cos`, `Day_sin`, `Holidays` | Deterministic calendar features (future-known) |

These are configured in `config_dam.json` under `past_covariates`, `future_covariates`, and `temporal_covariates`. During training the model sees all historical covariate values; at inference time the 24-hour-ahead covariate values are fed as `future_df`.

### Single training run

```bash
cd scripts/
python run_dam.py --mode train --config config_dam.json
```

Saves the LoRA adapter to `outputs/chronos2_lora/best_checkpoint/`.

### Hyperparameter tuning with Optuna

```bash
python run_dam.py --mode tune --config config_dam.json --n-trials 30 --study-name dam_lora
```

Resume a halted study (add more trials):

```bash
python run_dam.py --mode tune --config config_dam.json --n-trials 50 --study-name dam_lora
```

The search space is defined in `hyper_opt.json`. Each trial saves its adapter under `outputs/chronos2_lora/trial_N/`. After all trials finish, the best hyperparameters are re-trained and saved to `outputs/chronos2_lora/best_retrain/`.

### Rolling inference

```bash
python run_dam.py --mode infer --config config_dam.json \
    --checkpoint outputs/chronos2_lora/best_retrain/best_checkpoint
```

Produces a CSV with per-hour quantile forecasts (`0.1`, `0.5`, `0.9`) written to the path set by `forecast_csv` in the config. If `--checkpoint` is omitted the script falls back to `outputs/chronos2_lora/best_checkpoint`.

### Key config fields (`config_dam.json`)

| Field | Description |
|---|---|
| `past_covariates` | Columns used as historical (past-only) covariates during training |
| `future_covariates` | Columns available as known-future values at inference time |
| `temporal_covariates` | Deterministic calendar features added automatically if listed |
| `context_lengths` | Context window sizes sampled during training |
| `infer_context_length` | Context length used during rolling inference |
| `prediction_length` | Forecast horizon in hours (24 = one day ahead) |
| `lora_r`, `lora_alpha`, `lora_dropout` | LoRA adapter hyperparameters |
| `max_steps`, `eval_every` | Training budget and logging interval |

---

## TimesFM Inference

```bash
cd scripts/
# Clone repo first — cannot install from PyPI
git clone https://github.com/google-research/timesfm.git
pip install -e ./timesfm[torch]

# DAM (default)
python run_zeroshot_timesfm.py --config config_dam.json --allow-negative

# Imbalance prices (can go negative → use --allow-negative)
python run_zeroshot_timesfm.py --config config_imb.json --allow-negative

# Smaller context, larger batch
python run_zeroshot_timesfm.py --context-length 512 --batch-size 64

# Skip torch.compile for faster startup during debugging
python run_zeroshot_timesfm.py --no-compile
```

---

## References

- [Chronos-2 technical report](https://arxiv.org/abs/2510.15821)
- [Chronos-2 quickstart notebook](notebooks/chronos-2-quickstart.ipynb)
- [PEFT / LoRA](https://github.com/huggingface/peft)
- [Optuna docs](https://optuna.readthedocs.io/)
- [Diebold-Mariano test](https://en.wikipedia.org/wiki/Diebold%E2%80%93Mariano_statistic)
