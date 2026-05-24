import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler

from datasets import load_data, prepare_fit_inputs
from model import finetune_pipeline, load_finetuned_pipeline, run_inference
from utils import (
    mean_absolute_error,
    root_mean_squared_error,
)

warnings.filterwarnings("ignore")

# ── Default configuration ─────────────────────────────────────────────────────

DEFAULT_CFG: dict = {
    "model_id": "autogluon/chronos-2",
    "train_csv": "dataset/dam/data_train.csv",
    "test_csv": "dataset/dam/data_test.csv",
    "output_dir": "outputs/chronos2_lora",
    "forecast_csv": "raw_forecasts_dam/chronos2_lora_tuned.csv",
    "train_end": "2022-09-30 23:00:00",
    "val_start": "2022-10-01 00:00:00",
    "val_end": "2022-12-31 23:00:00",
    "forecast_start": "2023-01-01",
    "forecast_end": "2025-01-01",
    "horizon": "1D",
    "step": "1D",
    "context_lengths": [8192],
    "infer_context_length": 2048,
    "prediction_length": 24,
    "quantile_levels": [0.1, 0.5, 0.9],
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q", "k", "v"],
    "lr": 5e-5,
    "batch_size": 64,
    "max_steps": 2000,
    "eval_every": 100,
    "seed": 42,
    "infer_batch_size": 64,
    "timestamp_column": "Date",
    "target_column": "Price",
    "past_covariates": [],
    "future_covariates": [],
    "temporal_covariates": [],
    "add_temporal_features": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _past_cov_cols(cfg: dict) -> list[str]:
    """Deduplicated list of all past covariate columns (physical + temporal)."""
    return list(dict.fromkeys(
        cfg.get("past_covariates", []) + cfg.get("temporal_covariates", [])
    ))


def _future_cov_cols(cfg: dict) -> list[str]:
    """Deduplicated list of all future-known covariate columns (physical + temporal)."""
    return list(dict.fromkeys(
        cfg.get("future_covariates", []) + cfg.get("temporal_covariates", [])
    ))


def _use_temporal(cfg: dict) -> bool:
    return bool(cfg.get("temporal_covariates")) or cfg.get("add_temporal_features", False)


def _eval_val_mae(
    pipeline,
    df_train: pd.DataFrame,
    cfg: dict,
    past_cov_cols: list[str],
    future_cov_cols: list[str],
) -> float:
    """
    """
    ts_col = cfg["timestamp_column"]
    tgt_col = cfg["target_column"]
    pred_len = cfg["prediction_length"]

    df = df_train.copy()
    df["id"] = "val"
    df[ts_col] = pd.to_datetime(
        df[ts_col]).dt.tz_localize(None)

    val_cutoffs = pd.date_range(
        start=cfg["forecast_start"],
        end=pd.Timestamp(cfg["forecast_end"]) - pd.Timedelta(cfg["horizon"]),
        freq=pd.Timedelta(cfg["step"]),
    )
    pred_df = run_inference(
        pipeline,
        df,
        val_cutoffs,
        ctx_len=cfg["infer_context_length"],
        prediction_length=pred_len,
        quantile_levels=cfg['quantile_levels'],
        past_cov_cols=past_cov_cols,
        future_cov_cols=future_cov_cols,
        timestamp_col=ts_col,
        target_col=tgt_col,
        batch_size=cfg["infer_batch_size"],
    )

    pred_col = "0.5" if "0.5" in pred_df.columns else "predictions"
    pred_df = pred_df.sort_values(ts_col).reset_index(drop=True)
    true_vals = (
        df_train.set_index(ts_col)
        .loc[pd.DatetimeIndex(pred_df[ts_col].values), tgt_col]
        .values
    )
    return mean_absolute_error(true_vals, pred_df[pred_col].values)


# ── Core training function ────────────────────────────────────────────────────

def train(cfg: dict, trial: optuna.Trial | None = None) -> float:
    """
    Fine-tune Chronos-2 with LoRA and covariates via the native pipeline.fit() API.

    Covariates come from config keys:
      past_covariates    — physical features seen historically (Solar, Wind, …)
      future_covariates  — same physical features available as day-ahead forecasts
      temporal_covariates — deterministic calendar features (Week_cos, Holidays, …)

    Returns val MAE of the first 24-h forecast window at val_start.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    df_train, _ = load_data(
        cfg["train_csv"],
        cfg["test_csv"],
        cfg["train_end"],
        cfg["val_start"],
        cfg["val_end"],
        add_temporal_feats=_use_temporal(cfg),
        timestamp_col=cfg["timestamp_column"],
    )

    train_inputs, val_inputs = prepare_fit_inputs(
        df_train,
        cfg["train_end"],
        cfg["val_start"],
        cfg["val_end"],
        past_cov_cols=cfg.get("past_covariates", []),
        future_cov_cols=cfg.get("future_covariates", []),
        temporal_cov_cols=cfg.get("temporal_covariates", []),
        target_col=cfg["target_column"],
        timestamp_col=cfg["timestamp_column"],
        max_context=max(cfg["context_lengths"]),
    )

    finetuned = finetune_pipeline(
        cfg["model_id"],
        train_inputs,
        val_inputs,
        cfg["prediction_length"],
        cfg,
        device,
    )

    ckpt = os.path.join(cfg["output_dir"], "best_checkpoint")
    finetuned.model.save_pretrained(ckpt)
    with open(os.path.join(cfg["output_dir"], "run_config.json"), "w") as f:
        json.dump(
            {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, list, bool))},
            f,
            indent=2,
        )
    print(f"\nTraining complete. Checkpoint saved to {ckpt}")

    val_mae = _eval_val_mae(finetuned, df_train, cfg, _past_cov_cols(cfg), _future_cov_cols(cfg))
    print(f"Val MAE (first 24-h window): {val_mae:.4f}")

    if trial is not None:
        trial.report(val_mae, cfg["max_steps"])

    return val_mae


# ── Optuna objective ──────────────────────────────────────────────────────────

_SEARCH_SPACE_PATH = os.path.join(os.path.dirname(__file__), "hyper_opt.json")


def load_search_space(path: str = _SEARCH_SPACE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _sample(trial: optuna.Trial, name: str, spec: dict):
    """Dispatch a single Optuna suggest call from a hyper_opt.json spec entry."""
    t = spec["type"]
    if t == "categorical":
        choices = [tuple(c) if isinstance(c, list) else c for c in spec["choices"]]
        value = trial.suggest_categorical(name, choices)
        return list(value) if isinstance(value, tuple) else value
    if t == "float":
        kwargs = {k: spec[k] for k in ("low", "high") if k in spec}
        if spec.get("log"):
            kwargs["log"] = True
        if "step" in spec and not spec.get("log"):
            kwargs["step"] = spec["step"]
        return trial.suggest_float(name, **kwargs)
    if t == "int":
        kwargs = {k: spec[k] for k in ("low", "high") if k in spec}
        if spec.get("log"):
            kwargs["log"] = True
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_int(name, **kwargs)
    raise ValueError(f"Unknown search space type '{t}' for parameter '{name}'")


def _make_objective(base_cfg: dict, search_space: dict):
    """Return an Optuna objective driven by the given search_space dict."""

    def objective(trial: optuna.Trial) -> float:
        cfg = base_cfg.copy()

        sampled = {name: _sample(trial, name, spec)
                   for name, spec in search_space.items()}
        cfg.update(sampled)

        cfg["output_dir"] = os.path.join(
            base_cfg["output_dir"], f"trial_{trial.number}")

        print(f"\n── Trial {trial.number} params ──")
        for k, v in sampled.items():
            print(f"  {k}: {v}")

        return train(cfg, trial=trial)

    return objective


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(cfg: dict, checkpoint_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts_col = cfg["timestamp_column"]
    tgt_col = cfg["target_column"]

    df_train, df_test = load_data(
        cfg["train_csv"],
        cfg["test_csv"],
        cfg["train_end"],
        cfg["val_start"],
        cfg["val_end"],
        add_temporal_feats=_use_temporal(cfg),
        timestamp_col=ts_col,
    )

    df_all = (
        pd.concat([df_train, df_test])
        .sort_values(ts_col)
        .reset_index(drop=True)
    )
    df_all["id"] = cfg.get("dataset_name", "series")

    pipeline = load_finetuned_pipeline(cfg["model_id"], checkpoint_path, device)

    past_cols = _past_cov_cols(cfg)
    fut_cols = _future_cov_cols(cfg)

    cutoff_dates = pd.date_range(
        start=cfg["forecast_start"],
        end=pd.Timestamp(cfg["forecast_end"]) - pd.Timedelta(cfg["horizon"]),
        freq=pd.Timedelta(cfg["step"]),
    )
    print(
        f"Generating {len(cutoff_dates)} forecasts "
        f"({cutoff_dates[0]} → {cutoff_dates[-1]})..."
    )
    if past_cols:
        print(f"Past covariates  : {[c for c in past_cols if c in df_all.columns]}")
    if fut_cols:
        print(f"Future covariates: {[c for c in fut_cols if c in df_all.columns]}")

    pred_df = run_inference(
        pipeline,
        df_all,
        cutoff_dates,
        ctx_len=cfg["infer_context_length"],
        prediction_length=cfg["prediction_length"],
        quantile_levels=cfg["quantile_levels"],
        past_cov_cols=past_cols,
        future_cov_cols=fut_cols,
        timestamp_col=ts_col,
        target_col=tgt_col,
        batch_size=cfg["infer_batch_size"],
    )

    os.makedirs(os.path.dirname(cfg["forecast_csv"]), exist_ok=True)
    pred_df.to_csv(cfg["forecast_csv"], index=False)
    print(f"Saved {len(pred_df):,} rows → {cfg['forecast_csv']}")
    print(f"Range: {pred_df[ts_col].min()} → {pred_df[ts_col].max()}")

    if "0.5" in pred_df.columns:
        test_mask = pred_df[ts_col] >= cfg["forecast_start"]
        df_test_eval = df_test[df_test[ts_col] >= cfg["forecast_start"]].sort_values(ts_col)
        if test_mask.any() and len(df_test_eval):
            preds = pred_df.loc[test_mask, "0.5"].values[:len(df_test_eval)]
            mae = mean_absolute_error(df_test_eval[tgt_col].values[:len(preds)], preds)
            rmse = root_mean_squared_error(df_test_eval[tgt_col].values[:len(preds)], preds)
            print(f"\nForecast period — MAE: {mae:.3f}  RMSE: {rmse:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos-2 LoRA finetuning")
    p.add_argument(
        "--mode",
        choices=["train", "tune", "infer"],
        required=True,
        help="train: single run | tune: Optuna HPO | infer: rolling forecast",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (falls back to built-in defaults)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to LoRA checkpoint directory (required for --mode infer)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (tune mode only)",
    )
    p.add_argument(
        "--study-name",
        default="chronos2_lora_hpo",
        help="Optuna study name (tune mode only)",
    )
    p.add_argument(
        "--study-db",
        default=None,
        help=(
            "SQLite URL for persistent Optuna storage, e.g. "
            "sqlite:///outputs/optuna.db  (tune mode only; "
            "defaults to outputs/<output_dir>/optuna.db)"
        ),
    )
    p.add_argument(
        "--search-space",
        default=None,
        help=(
            "Path to hyper_opt.json search space file "
            "(defaults to hyper_opt.json next to run.py)"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from config",
    )
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path is not None:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    os.makedirs(cfg["output_dir"], exist_ok=True)

    if args.mode == "train":
        train(cfg)

    elif args.mode == "tune":
        db_url = args.study_db or f"sqlite:///{cfg['output_dir']}/optuna.db"
        print(f"Optuna storage: {db_url}")
        print(f"Study name    : {args.study_name}")
        print(f"Trials        : {args.n_trials}")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=cfg["seed"]),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=5),
            study_name=args.study_name,
            storage=db_url,
            load_if_exists=True,
        )
        search_space_path = args.search_space or _SEARCH_SPACE_PATH
        search_space = load_search_space(search_space_path)
        print(
            f"Search space  : {search_space_path}  ({len(search_space)} params)")

        study.optimize(_make_objective(cfg, search_space),
                       n_trials=args.n_trials)

        best = study.best_params
        print("\n── Best hyperparameters ──")
        for k, v in best.items():
            print(f"  {k}: {v}")

        best_params_path = os.path.join(cfg["output_dir"], "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nBest params saved to {best_params_path}")

        print("\nRe-training with best hyperparameters...")
        cfg_best = cfg.copy()
        cfg_best.update(best)
        cfg_best["output_dir"] = os.path.join(
            cfg["output_dir"], "best_retrain")
        train(cfg_best)

    elif args.mode == "infer":
        if args.checkpoint is None:
            ckpt = os.path.join(cfg["output_dir"], "best_checkpoint")
            print(f"No --checkpoint provided; using {ckpt}")
        else:
            ckpt = args.checkpoint
        infer(cfg, ckpt)


if __name__ == "__main__":
    main()
