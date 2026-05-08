import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from tqdm import tqdm

from datasets import load_data, make_dataloaders, add_temporal_features
from model import build_lora_model, forward_step, load_finetuned_pipeline, run_inference
from utils import (
    mean_absolute_error,
    plot_training_curve,
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
    "context_lengths": [8192],
    "infer_context_length": 2048,
    "prediction_length": 24,
    "quantile_levels": [0.1, 0.5, 0.9],
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q", "k", "v"],
    "lr": 5e-5,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_steps": 2000,
    "eval_every": 100,
    "patience": 5,
    "grad_clip": 1.0,
    "seed": 42,
    "num_workers": 2,
    "add_temporal_features": False,
    "infer_batch_size": 64,
}


# ── Core training function ────────────────────────────────────────────────────

def train(cfg: dict, trial: optuna.Trial | None = None) -> float:
    """
    Train the LoRA model with the given config.

    If `trial` is provided (Optuna mode), intermediate val losses are reported
    for pruning and the function raises TrialPruned when appropriate.

    Returns the best validation loss achieved.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Data
    train_prices, val_prices, _, _ = load_data(
        cfg["train_csv"],
        cfg["test_csv"],
        cfg["train_end"],
        cfg["val_start"],
        cfg["val_end"],
        add_temporal_feats=cfg.get("add_temporal_features", False),
    )
    train_loader, val_loader = make_dataloaders(
        train_prices,
        val_prices,
        cfg["context_lengths"],
        cfg["prediction_length"],
        cfg["batch_size"],
        cfg.get("num_workers", 2),
    )

    # Model
    _, lora_model = build_lora_model(
        cfg["model_id"],
        cfg["lora_r"],
        cfg["lora_alpha"],
        cfg["lora_dropout"],
        cfg["lora_target_modules"],
        device,
    )

    # Optimiser + scheduler
    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_steps"], eta_min=cfg["lr"] * 0.1
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_log: list[tuple[int, float]] = []

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, cfg["max_steps"] + 1), desc="Finetuning")

    for step in pbar:
        lora_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        loss = forward_step(lora_model, batch, device)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        pbar.set_postfix(
            {"train": f"{loss.item():.4f}",
             "lr": f"{scheduler.get_last_lr()[0]:.1e}"}
        )

        if step % cfg["eval_every"] == 0:
            lora_model.eval()
            val_accum, n_val = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    val_accum += forward_step(lora_model, vb, device).item()
                    n_val += 1

            avg_val = val_accum / n_val
            avg_tr = float(np.mean(train_losses[-cfg["eval_every"]:]))
            val_log.append((step, avg_val))

            tqdm.write(
                f"Step {step:4d} | train: {avg_tr:.4f} | val: {avg_val:.4f}"
                f" | lr: {scheduler.get_last_lr()[0]:.1e}"
            )

            # Optuna pruning
            if trial is not None:
                trial.report(avg_val, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                ckpt = os.path.join(cfg["output_dir"], "best_checkpoint")
                lora_model.save_pretrained(ckpt)
                tqdm.write(
                    f"  >> New best val loss {best_val_loss:.4f} — saved to {ckpt}")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    tqdm.write(
                        f"Early stopping at step {step} "
                        f"(no improvement for {cfg['patience']} eval intervals)"
                    )
                    break

    # Save final checkpoint, config, and training curve
    lora_model.save_pretrained(os.path.join(
        cfg["output_dir"], "final_checkpoint"))
    with open(os.path.join(cfg["output_dir"], "run_config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if isinstance(
            v, (str, int, float, list, bool))}, f, indent=2)
    plot_training_curve(train_losses, val_log, cfg["output_dir"])

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


# ── Optuna objective ──────────────────────────────────────────────────────────

_SEARCH_SPACE_PATH = os.path.join(os.path.dirname(__file__), "hyper_opt.json")


def load_search_space(path: str = _SEARCH_SPACE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _sample(trial: optuna.Trial, name: str, spec: dict):
    """Dispatch a single Optuna suggest call from a hyper_opt.json spec entry."""
    t = spec["type"]
    if t == "categorical":
        # JSON lists-of-lists come in as lists; Optuna needs them hashable → tuple
        choices = [tuple(c) if isinstance(c, list)
                   else c for c in spec["choices"]]
        value = trial.suggest_categorical(name, choices)
        # Convert back to plain list for the config
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
    _, _, df_train, df_test = load_data(
        cfg["train_csv"],
        cfg["test_csv"],
        cfg["train_end"],
        cfg["val_start"],
        cfg["val_end"],
        add_temporal_feats=cfg.get("add_temporal_features", False),
    )

    df_all = (
        pd.concat([df_train, df_test])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df_all["id"] = "BE_DAM"

    pipeline = load_finetuned_pipeline(
        cfg["model_id"], checkpoint_path, device)

    cutoff_dates = pd.date_range(
        start=cfg["forecast_start"],
        end=pd.Timestamp(cfg["forecast_end"]) - pd.Timedelta("1D"),
        freq="1D",
    )
    print(
        f"Generating {len(cutoff_dates)} daily forecasts "
        f"({cutoff_dates[0].date()} → {cutoff_dates[-1].date()})..."
    )

    pred_df = run_inference(
        pipeline,
        df_all,
        cutoff_dates,
        ctx_len=cfg["infer_context_length"],
        prediction_length=cfg["prediction_length"],
        quantile_levels=cfg["quantile_levels"],
        batch_size=cfg["infer_batch_size"],
    )

    os.makedirs(os.path.dirname(cfg["forecast_csv"]), exist_ok=True)
    pred_df.to_csv(cfg["forecast_csv"], index=False)
    print(f"Saved {len(pred_df):,} rows → {cfg['forecast_csv']}")
    print(f"Date range: {pred_df['Date'].min()} → {pred_df['Date'].max()}")

    # Quick point-forecast MAE on the test period (2024)
    test_mask = pred_df["Date"] >= "2024-01-01"
    if test_mask.any() and "0.5" in pred_df.columns:
        df_test_2024 = df_test[df_test["Date"]
                               >= "2024-01-01"].sort_values("Date")
        preds_2024 = pred_df.loc[test_mask, "0.5"].values
        mae = mean_absolute_error(df_test_2024["Price"].values, preds_2024)
        rmse = root_mean_squared_error(
            df_test_2024["Price"].values, preds_2024)
        print(f"\n2024 test set — MAE: {mae:.3f}  RMSE: {rmse:.3f}")


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

        # Re-train with best params on the full budget
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
