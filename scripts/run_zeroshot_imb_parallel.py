"""
run_zeroshot.py — Zero-shot Chronos-2 inference runner.

This script runs rolling zero-shot forecasts using a pretrained Chronos-2 pipeline.
It supports two modes:
  - AR: autoregressive univariate forecasting using only the target series.
  - ARX: autoregressive forecasting with covariates.

Temporal features can be added to the input data before inference, and the model's
context length is adjustable via `pipeline.model.chronos_config.context_length.`
"""

import argparse
import os
from typing import Iterable
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from chronos import BaseChronosPipeline
from utils import generate_cutoff_dates, load_config, load_dataset
from datasets import add_temporal_features

CONFIG = load_config('./config_imb.json')
NUM_WORKERS = 4          # 4 workers × 4 threads = 16 cores fully used
THREADS_PER_WORKER = 16 // NUM_WORKERS
BATCH_SIZE = 32
# BATCH_CONTEXT = 64
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot Chronos-2 forecasts with varying context length and covariates."
    )
    parser.add_argument(
        "--model-id",
        default=CONFIG['model_id'],
        help="Pretrained Chronos-2 model identifier or path.",
    )
    parser.add_argument(
        "--train-csv",
        default=CONFIG['train_csv'],
        help="Training CSV file for context construction.",
    )
    parser.add_argument(
        "--test-csv",
        default=CONFIG['test_csv'],
        help="Test CSV file containing the forecast period.",
    )
    parser.add_argument(
        "--mode",
        choices=["AR", "ARX"],
        default="ARX",
        help="Forecast mode: AR uses only the target, ARX includes covariates.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONFIG["infer_context_length"],
        help="Model context length for zero-shot prediction.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=CONFIG["prediction_length"],
        help="Prediction horizon in time steps.",
    )
    parser.add_argument(
        "--add-temporal-features",
        action="store_true",
        help="Add cyclical temporal features and Belgian holidays to the data.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["da_price", "si"],
        help="Covariate columns to include in ARX mode.",
    )
    parser.add_argument(
        "--forecast-start",
        default=CONFIG["forecast_start"],
        help="First cutoff date for rolling forecast.",
    )
    parser.add_argument(
        "--forecast-end",
        default=CONFIG["forecast_end"],
        help="End date for rolling forecast (exclusive horizon end).",
    )
    parser.add_argument(
        "--step",
        default="15 minutes",
        help="Step size between rolling forecast cutoffs (e.g. 1D, 15min).",
    )
    parser.add_argument(
        "--horizon",
        default="2 hour",
        help="Forecast horizon for cutoff generation (e.g. 1D, 2H).",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_forecast_imb",
        help="Directory where forecast CSV files will be written.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=CONFIG['quantile_levels'],
        help="Quantile levels for probabilistic predictions. Defaults to pipeline.quantiles.",
    )
    parser.add_argument(
        "--target-column",
        default="pos_ip",
        help="Target column name in the input CSV files.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Timestamp column name in the input CSV files.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Series id column name in the input CSV files.",
    )
    return parser.parse_args()


def build_context_frame(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    context_length: int,
    timestamp_column: str,
    prediction_length: int,
    past_cov_columns: Iterable[str],
    future_cov_columns: Iterable[str]
):
    context = df_all[df_all[timestamp_column] < cutoff].iloc[-context_length:]
    if len(future_cov_columns) > 2:
        # hard-code by prediction length unit by hour
        future_df = df_all[(df_all[timestamp_column] >= cutoff) &
                           (df_all[timestamp_column] < cutoff + pd.Timedelta(hours=prediction_length/4))]
        return context.loc[:, list(past_cov_columns)].copy(), future_df.loc[:, list(future_cov_columns)].copy()
    else:
        return context.loc[:, list(past_cov_columns)].copy(), pd.DataFrame()

def run_worker(worker_idx, cutoff_chunk, df_all, model_id, context_length,
               prediction_length, quantiles, context_columns, future_cols,
               id_col, ts_col, target_col):
    torch.set_num_threads(THREADS_PER_WORKER)
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
    pipeline.model.chronos_config.context_length = context_length

    # ts_array = df_all[ts_col].values  # for fast searchsorted slicing
    batch_indices = range(0, len(cutoff_chunk), BATCH_SIZE)
    result_frames = []
    for batch_start in tqdm(
            batch_indices,
            position=worker_idx,          # each worker occupies its own line
            desc=f"Worker {worker_idx:>2}",
            leave=True,
            total=len(batch_indices),
        ):
        batch_cutoffs = cutoff_chunk[batch_start : batch_start + BATCH_SIZE]
        batch_contexts, batch_futures = [], []

        for i, cutoff in enumerate(batch_cutoffs):
            virtual_id = f"w{id(cutoff_chunk)}_{batch_start + i}"
            ctx, fut = build_context_frame(
                df_all, cutoff, context_length, ts_col,
                prediction_length, context_columns, future_cols,
            )
            ctx[id_col] = virtual_id
            batch_contexts.append(ctx)
            if len(fut) > 0:
                fut[id_col] = virtual_id
                batch_futures.append(fut)

        batch_df = pd.concat(batch_contexts, ignore_index=True)
        predict_kwargs = dict(
            prediction_length=prediction_length, quantile_levels=quantiles,
            id_column=id_col, timestamp_column=ts_col, target=target_col,
        )
        if batch_futures:
            predict_kwargs["future_df"] = pd.concat(batch_futures, ignore_index=True)

        pred_batch = pipeline.predict_df(batch_df, **predict_kwargs)

        for i, cutoff in enumerate(batch_cutoffs):
            virtual_id = f"w{id(cutoff_chunk)}_{batch_start + i}"
            pred = pred_batch[pred_batch[id_col] == virtual_id].copy()
            pred["cutoff_dates"] = cutoff
            result_frames.append(pred)

    return result_frames

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = load_dataset(args.train_csv, args.timestamp_column)
    test_df = load_dataset(args.test_csv, args.timestamp_column)

    # if args.add_temporal_features:
    print("Adding temporal features...")
    train_df = train_df.set_index(args.timestamp_column)
    test_df = test_df.set_index(args.timestamp_column)
    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all = df_all.sort_values(args.timestamp_column).reset_index(drop=True)
    df_all['id'] = 'BE'
    df_all[args.timestamp_column] = pd.to_datetime(
        df_all[args.timestamp_column]).dt.tz_localize(None)
    cutoff_dates = generate_cutoff_dates(
        pd.Timestamp(args.forecast_start),
        pd.Timestamp(args.forecast_end),
        pd.Timedelta(args.horizon),
        pd.Timedelta(args.step),
    )

    print(
        f"Using {len(cutoff_dates)} cutoff dates from {cutoff_dates[0]} to {cutoff_dates[-1]}.")

    required_columns = {args.target_column,
                        args.id_column, args.timestamp_column}
    if args.mode == "ARX":
        required_columns.update(args.features)
        if args.add_temporal_features:
            required_columns.update(
                ["Week_cos", "Week_sin", "Day_cos", "Day_sin", "Holidays"])

    missing = required_columns - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")

    print("Loading pretrained Chronos-2 pipeline...")
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device)
    pipeline.model.chronos_config.context_length = args.context_length
    quantiles = args.quantiles if args.quantiles is not None else pipeline.quantiles

    print(f"Mode: {args.mode}")
    print(f"Context length: {args.context_length}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Quantiles: {quantiles}")

    result_frames = []
    context_columns = [args.timestamp_column,
                    args.id_column, args.target_column]
    future_cols = [args.timestamp_column,
                    args.id_column, 'da_price']
    if args.mode == "ARX":
        context_columns.extend(args.features)
        if args.add_temporal_features:
            temporal_cols = ["Week_cos", "Week_sin", "Day_cos", "Day_sin", "Holidays"]
            context_columns.extend(temporal_cols)
            future_cols.extend(temporal_cols)
    
    # Split cutoffs across workers
    chunks = [cutoff_dates[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    worker_args = dict(
        df_all=df_all, model_id=args.model_id, context_length=args.context_length,
        prediction_length=args.prediction_length, quantiles=quantiles,
        context_columns=context_columns, future_cols=future_cols,
        id_col=args.id_column, ts_col=args.timestamp_column, target_col=args.target_column,
    )

    all_frames = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as ex:
        futures = {ex.submit(run_worker, i, chunk, **worker_args): i
                for i, chunk in enumerate(chunks)}
        for fut in tqdm(
            as_completed(futures),
            total=NUM_WORKERS,
            position=NUM_WORKERS,    # master bar sits below all worker bars
            desc="Workers done",
            leave=True,
        ):
            all_frames.extend(fut.result())

    all_preds = pd.concat(all_frames, ignore_index=True)

    output_name = f"chronos2_{args.context_length}_{args.mode}"
    if args.add_temporal_features:
        output_name += "_temporal"
    output_path = os.path.join(args.output_dir, f"{output_name}.csv")
    print(f"Saving forecasts to {output_path}")
    all_preds.drop(columns=[args.timestamp_column]
                   ).to_csv(output_path, index=False)

    print("Zero-shot forecast complete.")


if __name__ == "__main__":
    main()
