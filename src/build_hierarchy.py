"""Prepare weekly aggregated hierarchy and design matrices."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def _validate_monday(date: pd.Timestamp):
    if date.weekday() != 0:
        raise ValueError(f"forecast_start_date must be a Monday. Received {date.date()}")


def _make_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unique_id"] = (
        df["product_id"].astype(str)
        + "|"
        + df["location_node_id"].astype(str)
        + "|"
        + df["customer_node_id"].astype(str)
    )
    df["sku_id"] = df["product_id"].astype(str)
    df["sku_location_id"] = df["product_id"].astype(str) + "|" + df["location_node_id"].astype(str)
    return df


def _aggregate_level(df: pd.DataFrame, group_cols: List[str], label_col: str) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols + ["ds"], as_index=False)
        .agg({"y": "sum", "unit_price": "mean"})
        .assign(unique_id=lambda x: x[label_col])
    )
    return agg[["unique_id", "ds", "y", "unit_price"]]


def _build_S_matrix(bottom_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """Construct aggregation matrix S and tags used by hierarchicalforecast.

    The ordering of nodes is: Total → SKU → SKU-Location → Bottom.
    """
    bottom_ids = bottom_df["unique_id"].unique().tolist()
    sku_ids = bottom_df[["sku_id", "product_id"]].drop_duplicates()["sku_id"].tolist()
    sku_loc_ids = bottom_df[["sku_location_id", "product_id", "location_node_id"]].drop_duplicates()[
        "sku_location_id"
    ].tolist()

    tags = {
        "total": ["Total"],
        "sku": sku_ids,
        "sku_location": sku_loc_ids,
        "bottom": bottom_ids,
    }

    rows = []
    node_order = tags["total"] + tags["sku"] + tags["sku_location"] + tags["bottom"]
    bottom_index = {b: i for i, b in enumerate(bottom_ids)}

    # Total row
    rows.append(np.ones(len(bottom_ids), dtype=float))

    # SKU rows
    for sku in tags["sku"]:
        mask = [1.0 if b.startswith(sku + "|") else 0.0 for b in bottom_ids]
        rows.append(np.array(mask, dtype=float))

    # SKU-Location rows
    for sku_loc in tags["sku_location"]:
        mask = [1.0 if b.startswith(sku_loc + "|") else 0.0 for b in bottom_ids]
        rows.append(np.array(mask, dtype=float))

    # Bottom identity
    rows.extend(np.eye(len(bottom_ids), dtype=float))

    S = np.vstack(rows)
    assert S.shape[0] == len(node_order)
    assert S.shape[1] == len(bottom_ids)
    return S, tags, node_order, bottom_index


def build_weekly_hierarchy(df: pd.DataFrame, forecast_start_date: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]], List[str], Dict[str, int]]:
    """Aggregate to Monday-aligned weeks and build hierarchy components."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    cutoff = pd.to_datetime(forecast_start_date)
    _validate_monday(cutoff)

    df = df[df["ds"] < cutoff]
    df = _make_unique_ids(df)
    df = df.rename(columns={"quantity_clean": "y"})

    # Monday aligned weekly sum, mean for price
    df_weekly = (
        df.set_index("ds")
        .groupby(["unique_id", "sku_id", "sku_location_id", "product_id", "location_node_id", "customer_node_id"])
        .resample("W-MON")
        .agg({"y": "sum", "unit_price": "mean"})
        .reset_index()
    )

    bottom_df = df_weekly[["unique_id", "ds", "y", "unit_price", "sku_id", "sku_location_id"]].copy()

    sku_level = _aggregate_level(bottom_df, ["sku_id"], "sku_id")
    sku_location_level = _aggregate_level(bottom_df, ["sku_location_id"], "sku_location_id")
    total_level = (
        bottom_df.groupby(["ds"], as_index=False)
        .agg({"y": "sum", "unit_price": "mean"})
        .assign(unique_id="Total")
    )

    all_series = pd.concat([total_level, sku_level, sku_location_level, bottom_df[["unique_id", "ds", "y", "unit_price"]]], ignore_index=True)

    S, tags, node_order, bottom_index = _build_S_matrix(bottom_df)
    return all_series, S, tags, node_order, bottom_index


if __name__ == "__main__":
    print("Module is intended to be imported from generate_forecast.py")
