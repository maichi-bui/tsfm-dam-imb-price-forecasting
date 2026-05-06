"""
datasets.py — Data loading and sliding-window dataset for Chronos-2 LoRA finetuning.
"""
import holidays
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    """
    Sliding window over a 1-D price series.

    Context length is sampled randomly from context_lengths on each __getitem__,
    giving effective diversity without multiplying the dataset size on disk.
    Windows are anchored at the right edge so the target immediately follows.
    """

    def __init__(self, series: np.ndarray, context_lengths: list[int], prediction_length: int):
        self.series = series
        self.context_lengths = context_lengths
        self.pred_len = prediction_length
        self.max_ctx = max(context_lengths)
        self.n = max(0, len(series) - self.max_ctx - prediction_length)
        if self.n == 0:
            raise ValueError(
                f"Series too short ({len(series)}) for "
                f"max_ctx={self.max_ctx} + pred_len={prediction_length}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx_len = int(np.random.choice(self.context_lengths))
        ctx_end = idx + self.max_ctx
        ctx_start = ctx_end - ctx_len
        tgt_end = ctx_end + self.pred_len

        context = torch.tensor(
            self.series[ctx_start:ctx_end], dtype=torch.float32)
        target = torch.tensor(
            self.series[ctx_end:tgt_end], dtype=torch.float32)
        return {"context": context, "target": target}


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Left-pad variable-length contexts with NaN; Chronos ignores padded positions."""
    max_ctx = max(b["context"].shape[0] for b in batch)
    padded = []
    for b in batch:
        ctx = b["context"]
        if ctx.shape[0] < max_ctx:
            pad = torch.full((max_ctx - ctx.shape[0],), float("nan"))
            ctx = torch.cat([pad, ctx])
        padded.append(ctx)
    return {
        "context": torch.stack(padded),
        "target": torch.stack([b["target"] for b in batch]),
    }


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical temporal features and holiday indicator to a DataFrame.

    Adds columns:
      - Week_cos, Week_sin : day-of-week cyclical encoding
      - Day_cos, Day_sin   : hour-of-day cyclical encoding
      - Holidays           : Belgian holiday indicator (0/1)

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    years_range : (min_year, max_year) tuple for holiday lookup; defaults to index range

    Returns
    -------
    DataFrame with added temporal feature columns.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # Day-of-week cyclical encoding
    day_of_week = df.index.dayofweek / 7.0
    df["Week_cos"] = np.cos(2 * np.pi * day_of_week)
    df["Week_sin"] = np.sin(2 * np.pi * day_of_week)

    # Hour-of-day cyclical encoding
    hour_of_day = df.index.hour / 24.0
    df["Day_cos"] = np.cos(2 * np.pi * hour_of_day)
    df["Day_sin"] = np.sin(2 * np.pi * hour_of_day)

    # Belgian holidays
    belgian_holidays = holidays.Belgium(years=range(
        df.index.year.min(), df.index.year.max() + 1))
    df["Holidays"] = [int(ts.date() in belgian_holidays)
                      for ts in df.index]
    return df


# def add_temporal_features(df: pd.DataFrame, years_range: tuple[int, int] | None = None) -> pd.DataFrame:
#     """
#     Add cyclical temporal features and holiday indicator to a DataFrame.

#     Adds columns:
#       - Week_cos, Week_sin : day-of-week cyclical encoding
#       - Day_cos, Day_sin   : hour-of-day cyclical encoding
#       - Holidays           : Belgian holiday indicator (0/1)

#     Parameters
#     ----------
#     df : DataFrame with DatetimeIndex
#     years_range : (min_year, max_year) tuple for holiday lookup; defaults to index range

#     Returns
#     -------
#     DataFrame with added temporal feature columns.
#     """
#     df = df.copy()
#     idx = df.index if isinstance(
#         df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)

#     # Day-of-week cyclical encoding
#     day_of_week = idx.dayofweek / 7.0
#     df["Week_cos"] = np.cos(2 * np.pi * day_of_week)
#     df["Week_sin"] = np.sin(2 * np.pi * day_of_week)

#     # Hour-of-day cyclical encoding
#     hour_of_day = idx.hour / 24.0
#     df["Day_cos"] = np.cos(2 * np.pi * hour_of_day)
#     df["Day_sin"] = np.sin(2 * np.pi * hour_of_day)

#     # Belgian holidays
#     if years_range is None:
#         years_range = (idx.year.min(), idx.year.max())
#     belgian_holidays = holidays.Belgium(years=range(*years_range))
#     df["Holidays"] = [int(date in belgian_holidays) for date in idx.date]

#     return df


def load_data(
    train_csv: str,
    test_csv: str,
    train_end: str,
    val_start: str,
    val_end: str,
    add_temporal_feats: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs and split into finetune-train / finetune-val price arrays.

    Parameters
    ----------
    train_csv, test_csv : paths
    train_end, val_start, val_end : date strings for splitting
    add_temporal_feats : if True, adds cyclical encodings and holidays to the DataFrames

    Returns (train_prices, val_prices, df_train_full, df_test_full).

    Note: prices are extracted as 1-D numpy arrays. Temporal features are added
    to the full DataFrames if requested but are not used by default in the
    SlidingWindowDataset. To use temporal features, you can extend SlidingWindowDataset
    or add them as covariates in a custom forward_step.
    """
    df_train = (
        pd.read_csv(train_csv, parse_dates=["Date"])
        .sort_values("Date")
        .set_index("Date")
        .drop(columns=["Price_DE_LU"], errors="ignore")
        .reset_index()
    )
    df_test = (
        pd.read_csv(test_csv, parse_dates=["Date"])
        .sort_values("Date")
        .set_index("Date")
        .drop(columns=["Price_DE_LU"], errors="ignore")
        .reset_index()
    )

    if add_temporal_feats:
        df_train = df_train.set_index("Date")
        df_test = df_test.set_index("Date")
        df_train = add_temporal_features(df_train)
        df_test = add_temporal_features(df_test)
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()
        print("Temporal features added: Week_cos, Week_sin, Day_cos, Day_sin, Holidays")

    mask_tr = df_train["Date"] <= train_end
    mask_val = (df_train["Date"] >= val_start) & (df_train["Date"] <= val_end)

    train_prices = df_train.loc[mask_tr, "Price"].values.astype(np.float32)
    val_prices = df_train.loc[mask_val, "Price"].values.astype(np.float32)

    print(
        f"Train file : {df_train['Date'].min().date()} → {df_train['Date'].max().date()} "
        f"({len(df_train):,} rows)"
    )
    print(
        f"Test file  : {df_test['Date'].min().date()} → {df_test['Date'].max().date()} "
        f"({len(df_test):,} rows)"
    )
    print(
        f"Finetune train : {mask_tr.sum():,} hours "
        f"({df_train.loc[mask_tr, 'Date'].min().date()} → "
        f"{df_train.loc[mask_tr, 'Date'].max().date()})"
    )
    print(
        f"Finetune val   : {mask_val.sum():,} hours "
        f"({df_train.loc[mask_val, 'Date'].min().date()} → "
        f"{df_train.loc[mask_val, 'Date'].max().date()})"
    )

    return train_prices, val_prices, df_train, df_test


def make_dataloaders(
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    context_lengths: list[int],
    prediction_length: int,
    batch_size: int,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Val series is prepended with the last max(context_lengths) train observations
    so early val windows have a full context.
    """
    val_series = np.concatenate(
        [train_prices[-max(context_lengths):], val_prices]
    )

    train_ds = SlidingWindowDataset(
        train_prices, context_lengths, prediction_length)
    val_ds = SlidingWindowDataset(
        val_series, context_lengths, prediction_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train windows : {len(train_ds):,}")
    print(f"Val windows   : {len(val_ds):,}")
    return train_loader, val_loader
