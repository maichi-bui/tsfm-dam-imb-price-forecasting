"""
datasets.py — Data loading and input preparation for Chronos-2 LoRA finetuning.
"""
import holidays
import numpy as np
import pandas as pd


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


def load_data(
    train_csv: str,
    test_csv: str,
    train_end: str,
    val_start: str,
    val_end: str,
    add_temporal_feats: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs and return the full train and test DataFrames.

    Parameters
    ----------
    train_csv, test_csv : paths
    train_end, val_start, val_end : date strings (used only for split-size logging)
    add_temporal_feats : if True, adds cyclical encodings and holidays to the DataFrames
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

    return df_train, df_test


def prepare_fit_inputs(
    df_train: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    past_cov_cols: list[str],
    future_cov_cols: list[str],
    temporal_cov_cols: list[str],
    target_col: str = "Price",
    max_context: int = 8192,
) -> tuple[list[dict], list[dict]]:
    """
    Build input dicts for pipeline.fit() with past and future covariates.

    Training: full training price series with all covariate histories.
    Validation: prepend the last max_context training rows as context so
    early val windows are not starved of history.

    Physical covariates (Solar, Wind, Load, …) and temporal features
    (Week_cos, …) are all future-known: their historical values go into
    past_covariates, and future_covariates registers them with None to
    signal they will be provided at prediction time.
    """
    mask_tr = df_train["Date"] <= train_end
    mask_val = (df_train["Date"] >= val_start) & (df_train["Date"] <= val_end)

    df_tr = df_train[mask_tr].reset_index(drop=True)
    df_val = df_train[mask_val].reset_index(drop=True)

    # All covariates for which we have historical values
    all_past = list(dict.fromkeys(past_cov_cols + temporal_cov_cols))
    # Covariates that will be available as known-future at prediction time
    all_future_known = list(dict.fromkeys(future_cov_cols + temporal_cov_cols))

    def _build_input(df: pd.DataFrame) -> dict:
        past_avail = [c for c in all_past if c in df.columns]
        fut_avail = [c for c in all_future_known if c in df.columns]
        return {
            "target": df[target_col].values.astype(np.float32),
            "past_covariates": {c: df[c].values.astype(np.float32) for c in past_avail},
            "future_covariates": {c: None for c in fut_avail},
        }

    train_input = _build_input(df_tr)

    # Prepend training tail so early val windows have a full context buffer
    ctx_rows = df_tr.iloc[-max_context:] if len(df_tr) > max_context else df_tr
    df_val_ctx = pd.concat([ctx_rows, df_val], ignore_index=True)
    val_input = _build_input(df_val_ctx)

    print(f"Train input : {len(df_tr):,} steps | covariates: {list(train_input['past_covariates'].keys())}")
    print(f"Val input   : {len(df_val_ctx):,} steps ({len(df_val):,} val + {len(ctx_rows):,} context rows)")
    print(f"Future-known: {list(train_input['future_covariates'].keys())}")

    return [train_input], [val_input]
