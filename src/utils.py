"""Utility functions: metrics and feature builders."""
import numpy as np
import pandas as pd
from typing import Dict


def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) <= seasonal_period:
        return np.nan
    naive_forecast = y_true[:-seasonal_period]
    denom = np.mean(np.abs(y_true[seasonal_period:] - naive_forecast))
    return np.mean(np.abs(y_true - y_pred)) / denom if denom != 0 else np.nan


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) <= seasonal_period:
        return np.nan
    denom = np.mean((y_true[seasonal_period:] - y_true[:-seasonal_period]) ** 2)
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / denom) if denom != 0 else np.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = np.nan
    return np.nanmean(2.0 * np.abs(y_pred - y_true) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred - y_true))


def create_fourier_features(df: pd.DataFrame, period: int, order: int, prefix: str) -> pd.DataFrame:
    df = df.copy()
    t = np.arange(len(df))
    for k in range(1, order + 1):
        df[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["ds"].dt.month
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["dow"] = df["ds"].dt.dayofweek
    return df


def create_holiday_flags(df: pd.DataFrame, holiday_calendar: Dict[pd.Timestamp, str]) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = df["ds"].isin(list(holiday_calendar.keys())).astype(int)
    df["pre_holiday"] = df["ds"].isin([d - pd.Timedelta(days=7) for d in holiday_calendar]).astype(int)
    df["post_holiday"] = df["ds"].isin([d + pd.Timedelta(days=7) for d in holiday_calendar]).astype(int)
    return df
