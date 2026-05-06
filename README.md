# Chronos-2 Inference and Hyperparameter Tuning
This directory contains code for performing inference and hyperparameter tuning for electricity forecasting using the Chronos-2 model, specifically to Belgian Day Ahead Market and Imbalance Market, and can be adapted to adjacent markets and bidding zones. 
## Zeroshot Inference

```bash
cd scripts/
pip install -r requirements.txt
python3 run_zeroshot.py --config "./config_dam.json" --mode AR --context-length 1024 --forecast-end "2023-03-31"
python3 run_zeroshot.py --config "./config_imb.json" --mode ARX --context-length 2048 --add-temporal-features
```

## Hyperparameter Tuning with Optuna

```bash
cd scripts/
pip install -r requirements.txt
python run_dam.py --mode tune --config config.json --n-trials 30 --study-name dam_lora
```

Resume a halted study:
```bash
python run.py --mode tune --config config.json --n-trials 50 --study-name dam_lora
```
Rolling inference with best hyperparameters:

```bash
python run.py --mode infer --config config.json \
              --checkpoint outputs/chronos2_lora/best_retrain/best_checkpoint
```


## References

- [Chronos-2 paper](https://arxiv.org/abs/2310.10996)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Optuna docs](https://optuna.readthedocs.io/)
- [Diebold-Mariano test](https://en.wikipedia.org/wiki/Diebold%E2%80%93Mariano_statistic)
