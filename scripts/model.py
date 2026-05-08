"""
model.py — Chronos-2 LoRA model construction, forward step, and inference.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from chronos import BaseChronosPipeline
from tqdm import tqdm


def build_lora_model(
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
    device: str,
) -> tuple:
    """
    Load the base Chronos-2 pipeline, wrap its backbone with LoRA, and move to device.

    Returns (pipeline, lora_model) where lora_model is the LoRA-wrapped backbone.
    The pipeline's .model attribute is replaced with lora_model so predict_df still works.
    """
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
    base_model = pipeline.model

    total = sum(p.numel() for p in base_model.parameters())
    print(f"Base model : {type(base_model).__name__}  ({total / 1e6:.1f}M params)")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model = lora_model.to(device)

    pipeline.model = lora_model
    return pipeline, lora_model


def forward_step(
    lora_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Single differentiable step using Chronos-2's internal _compute_loss.

    NaN-padded positions in context are zeroed out before the forward pass;
    the model's instance_norm derives the scale from the non-padded region.
    """
    context = batch["context"].to(device)
    target = batch["target"].to(device)

    inner = lora_model.base_model.model
    output_patch_size = inner.chronos_config.output_patch_size
    num_output_patches = math.ceil(target.shape[1] / output_patch_size)

    context_clean = torch.nan_to_num(context, nan=0.0)

    output = lora_model(
        context=context_clean,
        future_target=target,
        num_output_patches=num_output_patches,
    )
    return output.loss


def load_finetuned_pipeline(
    model_id: str,
    checkpoint_path: str,
    device: str,
) -> BaseChronosPipeline:
    """
    Reload the base pipeline and attach a saved LoRA adapter for inference.
    """
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
    timestamp_col: str = "Date",
    target_col: str = "Price",
    id_col: str = "id",
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Rolling 1-day-ahead quantile forecast over the given cutoff dates.

    At each cutoff the function slices the last ctx_len observations as context
    and calls pipeline.predict_df for the next prediction_length hours.
    With batch_size > 1, multiple cutoffs are packed into a single predict_df
    call via virtual series IDs for faster GPU inference.

    Returns a DataFrame with columns [Date, q0.1, q0.5, ...].
    """
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
        batch_contexts = []
        for i, cutoff in enumerate(batch_cutoffs):
            ctx_df = df_all[df_all[timestamp_col] < cutoff].iloc[-ctx_len:].copy()
            ctx_df["virtual_id"] = f"series_{i}_{str(cutoff)}"
            batch_contexts.append(ctx_df)

        batch_df = pd.concat(batch_contexts, ignore_index=True)
        if id_col in batch_df.columns:
            batch_df = batch_df.drop(columns=[id_col])

        pred_batch = pipeline.predict_df(batch_df, **predict_kwargs)
        result_frames.append(pred_batch)

    pred_all = pd.concat(result_frames, ignore_index=True)

    drop_cols = [c for c in ["virtual_id", "target_name", "predictions"] if c in pred_all.columns]
    if "timestamp" in pred_all.columns and timestamp_col not in pred_all.columns:
        pred_all = pred_all.rename(columns={"timestamp": timestamp_col})

    return pred_all.drop(columns=drop_cols)
