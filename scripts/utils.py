"""
utils.py — Metrics, plotting, and Diebold-Mariano test utilities.
"""

import os
from itertools import combinations
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from scipy import stats


def generate_cutoff_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
) -> list[pd.Timestamp]:
    if horizon <= pd.Timedelta(0):
        raise ValueError("Horizon must be positive.")
    if step <= pd.Timedelta(0):
        raise ValueError("Step must be positive.")
    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    cutoff = start_date
    cutoff_dates = []
    while cutoff <= end_date - horizon:
        cutoff_dates.append(cutoff)
        cutoff += step

    if not cutoff_dates:
        raise ValueError(
            f"No cutoff dates between {start_date} and {end_date} "
            f"with horizon={horizon} and step={step}."
        )
    return cutoff_dates

# ── Forecast metrics ──────────────────────────────────────────────────────────


def load_dataset(path: str, timestamp_column: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column]).dt.tz_localize(None)
    return df


def load_config(path: str | None) -> dict:

    with open(path) as f:
        cfg = json.load(f)
    return cfg


def _validate(*arrays: ndarray) -> None:
    if any(a.size == 0 for a in arrays):
        raise ValueError("Found empty array in inputs.")
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Expected 1-D arrays.")
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")
    for a in arrays:
        if np.isnan(a).any():
            raise ValueError("Found NaN in inputs.")


def mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias_error(y_true: ndarray, y_pred: ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: ndarray, y_pred: ndarray) -> float:
    _validate(y_true, y_pred)
    if np.any(y_true == 0):
        raise ValueError("Zero in y_true — MAPE undefined.")
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def pinball_loss(y_true: ndarray, y_pred: ndarray, quantile: float) -> float:
    """Average pinball loss at a single quantile level."""
    _validate(y_true, y_pred)
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, quantile * e, (quantile - 1) * e)))


# ── Training plots ────────────────────────────────────────────────────────────

def plot_training_curve(
    train_losses: list[float],
    val_log: list[tuple[int, float]],
    output_dir: str,
    smooth_window: int = 50,
) -> None:
    steps = list(range(1, len(train_losses) + 1))
    smoothed = pd.Series(train_losses).rolling(
        smooth_window, min_periods=1).mean().values
    val_steps, val_vals = zip(*val_log) if val_log else ([], [])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(steps, train_losses, alpha=0.2,
            color="steelblue", label="train (per step)")
    ax.plot(steps, smoothed, color="steelblue", linewidth=2,
            label=f"train (smoothed {smooth_window})")
    if val_steps:
        ax.plot(val_steps, val_vals, "o-", color="tomato",
                linewidth=2, label="val loss")
        best_idx = int(np.argmin(val_vals))
        ax.axvline(val_steps[best_idx], color="tomato",
                   linestyle="--", alpha=0.5, label="best val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Pinball loss (normalised)")
    ax.set_title("Chronos-2 LoRA — Belgian DAM")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "training_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to {path}")


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def _newey_west_var(d: np.ndarray, lags: int) -> float:
    T = len(d)
    d_dm = d - d.mean()
    gamma0 = np.dot(d_dm, d_dm) / T
    nw = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gamma_k = np.dot(d_dm[k:], d_dm[:-k]) / T
        nw += 2 * w * gamma_k
    return max(nw / T, 1e-14)


def dm_test(
    loss1: pd.Series,
    loss2: pd.Series,
    h: int = 1,
    nw_lags: int | None = None,
) -> tuple[float, float]:
    """
    Modified Diebold-Mariano test (Harvey, Leybourne & Newbold 1997).

    Positive statistic → model 1 has higher (worse) loss than model 2.
    Returns (stat, two-sided p-value).
    """
    common = loss1.index.intersection(loss2.index)
    d = (loss1.loc[common] - loss2.loc[common]).values.astype(float)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3))) if nw_lags is None else nw_lags

    var_dbar = _newey_west_var(d, lags)
    dm_raw = d.mean() / np.sqrt(var_dbar)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm_raw * correction
    pval = 2.0 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(pval)


def daily_avg_pinball(df: pd.DataFrame, quantile_levels: list[float]) -> pd.Series:
    y = df["Price"].values
    losses = []
    for q in quantile_levels:
        e = y - df[str(q)].values
        losses.append(np.where(e >= 0, q * e, (q - 1) * e))
    per_hour = np.mean(losses, axis=0)
    return pd.Series(per_hour, index=df.index).resample("1D").mean().dropna()


def daily_winkler(df: pd.DataFrame, alpha: float = 0.2) -> pd.Series:
    lo_col = str(round(alpha / 2, 2))
    hi_col = str(round(1 - alpha / 2, 2))
    y = df["Price"].values
    lower = df[lo_col].values
    upper = df[hi_col].values
    width = upper - lower
    penalty = np.where(
        y < lower, (2 / alpha) * (lower - y),
        np.where(y > upper, (2 / alpha) * (y - upper), 0.0),
    )
    return pd.Series(width + penalty, index=df.index).resample("1D").mean().dropna()


def run_pairwise_dm(
    loss_dict: dict[str, pd.Series],
    h: int = 1,
    nw_lags: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run DM test for every ordered pair; return (stat_df, pval_df)."""
    names = sorted(loss_dict)
    stat_mat = pd.DataFrame(np.nan, index=names, columns=names)
    pval_mat = pd.DataFrame(np.nan, index=names, columns=names)

    for m1, m2 in combinations(names, 2):
        s, p = dm_test(loss_dict[m1], loss_dict[m2], h=h, nw_lags=nw_lags)
        stat_mat.loc[m1, m2] = s
        stat_mat.loc[m2, m1] = -s
        pval_mat.loc[m1, m2] = p
        pval_mat.loc[m2, m1] = p

    np.fill_diagonal(stat_mat.values, 0.0)
    np.fill_diagonal(pval_mat.values, 1.0)
    return stat_mat, pval_mat


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "** "
    if p < 0.10:
        return "*  "
    return "   "


def plot_dm_heatmap(
    stat_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    output_path: str,
    title: str = "Diebold-Mariano Test",
    vmax: float = 4.0,
) -> None:
    labels = [n.replace("chronos2small", "c2s").replace(
        "chronos2", "c2") for n in stat_df.index]
    ann = pd.DataFrame("", index=stat_df.index, columns=stat_df.columns)
    for r in stat_df.index:
        for c in stat_df.columns:
            if r == c:
                ann.loc[r, c] = "—"
            else:
                s = stat_df.loc[r, c]
                p = pval_df.loc[r, c]
                ann.loc[r, c] = f"{s:.2f}{_sig_stars(p)}"

    mask = np.eye(len(stat_df), dtype=bool)
    fig, ax = plt.subplots(
        figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        stat_df.values.astype(float),
        ax=ax,
        mask=mask,
        annot=ann.values,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "DM statistic  (+ = row worse)", "shrink": 0.75},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"DM heatmap saved to {output_path}")
