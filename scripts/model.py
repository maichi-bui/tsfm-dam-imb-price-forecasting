"""
model.py — Chronos-2 fine-tuning (native pipeline.fit API) and rolling inference.
"""

import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel
from chronos import BaseChronosPipeline
from tqdm import tqdm

from run_zeroshot import build_context_frame


def finetune_pipeline(
    model_id: str,
    train_inputs: list[dict],
    val_inputs: list[dict],
    prediction_length: int,
    cfg: dict,
    device: str,
) -> BaseChronosPipeline:
    """
    Fine-tune Chronos-2 with LoRA using the native pipeline.fit() API.

    train_inputs / val_inputs are lists of dicts with keys:
      target, past_covariates, future_covariates
    as produced by datasets.prepare_fit_inputs().
    """
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
    )

    return pipeline.fit(
        inputs=train_inputs,
        prediction_length=prediction_length,
        num_steps=cfg["max_steps"],
        learning_rate=cfg["lr"],
        batch_size=cfg["batch_size"],
        logging_steps=cfg.get("eval_every", 100),
        context_length=max(cfg["context_lengths"]),
        finetune_mode="lora",
        lora_config=lora_config,
        validation_inputs=val_inputs,
    )


def load_finetuned_pipeline(
    model_id: str,
    checkpoint_path: str,
    device: str,
) -> BaseChronosPipeline:
    """Reload the base pipeline and attach a saved LoRA adapter for inference."""
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
    pipeline.model = PeftModel.from_pretrained(pipeline.model, checkpoint_path)
    pipeline.model.eval()
    print(f"Loaded finetuned {model_id} from {checkpoint_path}")
    return pipeline


def run_inference(
    pipeline: BaseChronosPipeline,
    df_all: pd.DataFrame,
    cutoff_dates: pd.DatetimeIndex,
    ctx_len: int,
    prediction_length: int,
    quantile_levels: list[float],
    past_cov_cols: list[str],
    future_cov_cols: list[str],
    timestamp_col: str = "Date",
    target_col: str = "Price",
    id_col: str = "id",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Rolling 1-day-ahead quantile forecast with future covariates.

    Uses build_context_frame() from run_zeroshot.py to slice context and
    future DataFrames consistently with the zero-shot inference pipeline.
    Multiple cutoffs are packed into a single predict_df call via virtual
    series IDs for faster GPU throughput.

    context_cols  = [timestamp, id, target] + past_cov_cols (what the model sees historically)
    future_cols   = [timestamp, id] + future_cov_cols       (known-future values for next 24 h)

    future_df is only passed to predict_df when at least one future covariate
    is available in df_all, mirroring the has_future logic in run_zeroshot.py.
    """
    # Build column lists as run_zeroshot.py does for ARX mode
    context_cols = [timestamp_col, id_col, target_col] + [
        c for c in past_cov_cols if c in df_all.columns
    ]
    future_cols_list = [timestamp_col, id_col] + [
        c for c in future_cov_cols if c in df_all.columns
    ]
    # future_cols_list has > 2 entries only when covariates are present
    has_future = len(future_cols_list) > 2

    predict_kwargs = dict(
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        id_column="virtual_id",
        timestamp_column=timestamp_col,
        target=target_col,
    )

    result_frames = []
    for batch_start in tqdm(range(0, len(cutoff_dates), batch_size), desc="Rolling forecast"):
        batch_cutoffs = cutoff_dates[batch_start: batch_start + batch_size]
        batch_contexts, batch_futures = [], []
        batch_has_future = has_future

        for i, cutoff in enumerate(batch_cutoffs):
            vid = f"BE_{i}_{str(cutoff)}"

            ctx_frame, fut_frame = build_context_frame(
                df_all,
                cutoff,
                ctx_len,
                timestamp_col,
                prediction_length,
                context_cols,
                future_cols_list,
            )
            ctx_frame["virtual_id"] = vid
            batch_contexts.append(ctx_frame)

            if fut_frame.empty:
                batch_has_future = False
                continue
            fut_frame["virtual_id"] = vid
            batch_futures.append(fut_frame)

        batch_df = pd.concat(batch_contexts, ignore_index=True)
        if id_col in batch_df.columns:
            batch_df = batch_df.drop(columns=[id_col])

        if batch_has_future and batch_futures:
            fut_df = pd.concat(batch_futures, ignore_index=True)
            if id_col in fut_df.columns:
                fut_df = fut_df.drop(columns=[id_col])
            predict_kwargs["future_df"] = fut_df
        else:
            predict_kwargs.pop("future_df", None)

        pred_batch = pipeline.predict_df(batch_df, **predict_kwargs)
        result_frames.append(pred_batch)

    pred_all = pd.concat(result_frames, ignore_index=True)

    drop_cols = [c for c in ["virtual_id", "target_name", "predictions"] if c in pred_all.columns]
    if "timestamp" in pred_all.columns and timestamp_col not in pred_all.columns:
        pred_all = pred_all.rename(columns={"timestamp": timestamp_col})

    return pred_all.drop(columns=drop_cols)
