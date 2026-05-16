import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from timesfm import TimesFM_2p5_200M_torch
    from timesfm.configs import ForecastConfig
except ImportError:
    # Fallback for editable installs cloned from the repo
    from timesfm.src.timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # type: ignore
    from timesfm.src.timesfm.configs import ForecastConfig  # type: ignore

from utils import generate_cutoff_dates, load_config, load_dataset

MAX_CONTEXT = 8192
# quantile_forecast[:, :, 0] = mean; [:, :, k] = k/10 quantile for k in 1..9
TIMESFM_QUANTILE_IDX: dict[float, int] = {
    round(k / 10, 1): k for k in range(1, 10)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot TimesFM-2.5 forecasts with a rolling window."
    )
    parser.add_argument(
        "--config",
        default="./config_dam.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=MAX_CONTEXT,
        help=f"Context length (max {MAX_CONTEXT} for TimesFM 2.5).",
    )
    parser.add_argument(
        "--forecast-start",
        default="2023-01-01",
        help="First cutoff date for rolling forecast.",
    )
    parser.add_argument(
        "--forecast-end",
        default="2025-01-01",
        help="End date for rolling forecast (exclusive horizon end).",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_forecast",
        help="Directory where forecast CSV files will be written.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Quantile levels to save (must be multiples of 0.1 in [0.1, 0.9]).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of cutoff windows to forecast in a single model.forecast() call.",
    )
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="Disable the non-negative output constraint (needed for prices that go negative).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (faster startup, slower inference).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context_length = min(args.context_length, MAX_CONTEXT)
    if args.context_length > MAX_CONTEXT:
        print(
            f"Warning: requested context {args.context_length} exceeds TimesFM 2.5 "
            f"limit; clamped to {MAX_CONTEXT}."
        )

    for q in args.quantiles:
        if round(q, 1) not in TIMESFM_QUANTILE_IDX:
            raise ValueError(
                f"Quantile {q} not supported by TimesFM 2.5. "
                f"Supported values: {sorted(TIMESFM_QUANTILE_IDX)}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    print(f"Zero-shot TimesFM-2.5 on dataset: {cfg['dataset_name']}")

    train_df = load_dataset(cfg["train_csv"], cfg["timestamp_column"])
    test_df = load_dataset(cfg["test_csv"], cfg["timestamp_column"])
    df_all = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(cfg["timestamp_column"])
        .reset_index(drop=True)
    )

    cutoff_dates = generate_cutoff_dates(
        pd.Timestamp(args.forecast_start),
        pd.Timestamp(args.forecast_end),
        pd.Timedelta(cfg["horizon"]),
        pd.Timedelta(cfg["step"]),
    )
    print(
        f"{len(cutoff_dates)} cutoff dates: "
        f"{cutoff_dates[0]} → {cutoff_dates[-1]}"
    )

    print("Loading TimesFM-2.5-200M (PyTorch)…")
    model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=not args.no_compile,
    )

    print("Compiling with ForecastConfig…")
    model.compile(
        ForecastConfig(
            max_context=context_length,
            max_horizon=cfg["prediction_length"],
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            # disable non-negative constraint when prices can go negative
            infer_is_positive=not args.allow_negative,
            fix_quantile_crossing=True,
        )
    )

    ts_col = cfg["timestamp_column"]
    target_col = cfg["target_column"]
    pred_len: int = cfg["prediction_length"]
    horizon_td = pd.Timedelta(cfg["horizon"])
    batch_size: int = args.batch_size

    result_frames: list[pd.DataFrame] = []

    for batch_start in tqdm(
        range(0, len(cutoff_dates), batch_size), desc="Batches"
    ):
        batch_cutoffs = cutoff_dates[batch_start: batch_start + batch_size]
        batch_inputs: list[np.ndarray] = []
        batch_timestamps: list[np.ndarray] = []

        for cutoff in batch_cutoffs:
            context_vals = (
                df_all[df_all[ts_col] < cutoff][target_col]
                .values[-context_length:]
                .astype(np.float32)
            )
            batch_inputs.append(context_vals)

            future_ts = df_all[
                (df_all[ts_col] >= cutoff)
                & (df_all[ts_col] < cutoff + horizon_td)
            ][ts_col].values[:pred_len]
            batch_timestamps.append(future_ts)

        # TimesFM batches via a list of arrays — no special batch dimension needed
        point_forecast, quantile_forecast = model.forecast(
            horizon=pred_len,
            inputs=batch_inputs,
        )
        # point_forecast  : (batch, pred_len)
        # quantile_forecast: (batch, pred_len, 10)

        for i, (cutoff, future_ts) in enumerate(
            zip(batch_cutoffs, batch_timestamps)
        ):
            if len(future_ts) == 0:
                continue
            n = len(future_ts)
            frame = pd.DataFrame({ts_col: future_ts})
            frame["predictions"] = point_forecast[i, :n]
            for q in args.quantiles:
                q_idx = TIMESFM_QUANTILE_IDX[round(q, 1)]
                frame[str(q)] = quantile_forecast[i, :n, q_idx]
            result_frames.append(frame)

    all_preds = pd.concat(result_frames, ignore_index=True)

    forecast_folder = os.path.join(args.output_dir, cfg["dataset_name"])
    os.makedirs(forecast_folder, exist_ok=True)
    output_name = f"timesfm25_{context_length}_AR"
    if args.allow_negative:
        output_name += "_neg"
    output_path = os.path.join(forecast_folder, f"{output_name}.csv")
    print(f"Saving {len(all_preds)} rows → {output_path}")
    all_preds.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
