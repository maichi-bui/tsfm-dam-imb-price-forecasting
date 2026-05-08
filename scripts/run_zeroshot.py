import argparse
import os
from typing import Iterable
from tqdm import tqdm
import pandas as pd
from chronos import BaseChronosPipeline
from utils import generate_cutoff_dates, load_config, load_dataset
from datasets import add_temporal_features


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run zero-shot Chronos-2 forecasts with varying context length and covariates."
    )
    parser.add_argument(
        "--config",
        default='./config_dam.json',
        help="Path to JSON config file (falls back to built-in defaults)",
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
        default=2048,
        help="Model context length for zero-shot prediction.",
    )
    parser.add_argument(
        "--add-temporal-features",
        action="store_true",
        help="Add cyclical temporal features and Belgian holidays to the data.",
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
        default=None,
        help="Quantile levels for probabilistic predictions. Chronos-2 supports 21 quantiles",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Series id column name for model inference",
    )
    return parser.parse_args()


def build_context_frame(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    context_length: int,
    timestamp_column: str,
    prediction_length: int,
    past_cov_columns: Iterable[str],
    future_cov_columns: Iterable[str],
    is_imb=False
):
    context = df_all[df_all[timestamp_column] < cutoff].iloc[-context_length:]
    if len(future_cov_columns) > 2:
        # hard-code by prediction length unit by hour
        future_df = df_all[(df_all[timestamp_column] >= cutoff) &
                           (df_all[timestamp_column] < cutoff + pd.Timedelta(hours=prediction_length))]
        if is_imb:
            future_df = df_all[(df_all[timestamp_column] >= cutoff) &
                               (df_all[timestamp_column] < cutoff + pd.Timedelta(hours=prediction_length/4))]
        return context.loc[:, list(past_cov_columns)].copy(), future_df.loc[:, list(future_cov_columns)].copy()
    else:
        return context.loc[:, list(past_cov_columns)].copy(), pd.DataFrame()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    print(
        f"Start zeroshot inference with {cfg['dataset_name']} and {cfg['model_id']}")
    is_imb = False
    if cfg['dataset_name'] == 'imb':
        is_imb = True

    train_df = load_dataset(cfg['train_csv'], cfg['timestamp_column'])
    test_df = load_dataset(cfg['test_csv'], cfg['timestamp_column'])

    if args.add_temporal_features:
        print("Using temporal features...")
        print("Adding temporal features...")
        train_df = train_df.set_index(cfg['timestamp_column'])
        test_df = test_df.set_index(cfg['timestamp_column'])
        train_df = add_temporal_features(train_df)
        test_df = add_temporal_features(test_df)
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()

    df_all = pd.concat([train_df, test_df], ignore_index=True).sort_values(
        cfg['timestamp_column']).reset_index(drop=True)

    cutoff_dates = generate_cutoff_dates(
        pd.Timestamp(args.forecast_start),
        pd.Timestamp(args.forecast_end),
        pd.Timedelta(cfg['horizon']),
        pd.Timedelta(cfg['step']),
    )

    print(
        f"Using {len(cutoff_dates)} cutoff dates from {cutoff_dates[0]} to {cutoff_dates[-1]}.")

    context_columns = [cfg['timestamp_column'],
                       args.id_column, cfg['target_column']]
    future_cols = [cfg['timestamp_column'],
                   args.id_column]
    has_future = False

    if args.mode == "ARX":
        context_columns.extend(cfg['past_covariates'])
        if len(cfg['future_covariates']) > 0:
            future_cols.extend(cfg['future_covariates'])
            has_future = True
        if args.add_temporal_features:
            has_future = True
            context_columns.extend(cfg['temporal_covariates'])
            future_cols.extend(cfg['temporal_covariates'])

    missing = set(context_columns).union(
        set(future_cols)) - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")

    print("Loading pretrained Chronos-2 pipeline...")
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        cfg['model_id'], device_map=device, cache_dir='model_dir/pretrain_chronos2/')
    pipeline.model.chronos_config.context_length = args.context_length

    quantiles = cfg['quantile_levels'] if cfg['quantile_levels'] is not None else pipeline.quantiles

    print(f"Mode: {args.mode}")
    print(f"Context length: {args.context_length}")
    print(f"Prediction length: {cfg['prediction_length']}")
    print(f"Quantiles: {quantiles}")
    predict_kwargs = dict(
        prediction_length=cfg['prediction_length'],
        quantile_levels=quantiles,
        id_column='virtual_id',
        timestamp_column=cfg['timestamp_column'],
        target=cfg['target_column']
    )
    result_frames = []
    BATCH_SIZE = cfg['infer_batch_size']
    for batch_start in tqdm(range(0, len(cutoff_dates), BATCH_SIZE), desc="Batches"):
        batch_cutoffs = cutoff_dates[batch_start: batch_start + BATCH_SIZE]
        batch_contexts = []
        batch_futures = []
        for i, cutoff in enumerate(batch_cutoffs):
            context_frame, future_frame = build_context_frame(
                df_all,
                cutoff,
                args.context_length,
                cfg['timestamp_column'],
                cfg['prediction_length'],
                context_columns,
                future_cols,
                is_imb
            )
            # id of the batch
            context_frame['virtual_id'] = f"BE_{i}_{str(cutoff)}"
            batch_contexts.append(context_frame)

            if len(future_frame) == 0:
                has_future = False
                continue
            future_frame['virtual_id'] = f"BE_{i}_{str(cutoff)}"
            batch_futures.append(future_frame)

        batch_df = pd.concat(
            batch_contexts, ignore_index=True).drop('id', axis=1)

        if has_future:
            predict_kwargs["future_df"] = pd.concat(
                batch_futures, ignore_index=True).drop('id', axis=1)

        pred_batch = pipeline.predict_df(batch_df, **predict_kwargs)
        result_frames.append(pred_batch)
    all_preds = pd.concat(result_frames, ignore_index=True)
    output_name = f"chronos2_{args.context_length}_{args.mode}"
    if args.add_temporal_features:
        output_name += "_temporal"

    forecast_folder = os.path.join(args.output_dir, cfg['dataset_name'])
    os.makedirs(forecast_folder, exist_ok=True)
    output_path = os.path.join(forecast_folder, f"{output_name}.csv")
    print(f"Saving forecasts to {output_path}")
    all_preds.drop(columns=['predictions']).to_csv(output_path, index=False)

    print("Zero-shot forecast complete.")


if __name__ == "__main__":
    main()
