"""Rolling-origin backtesting and benchmark comparison."""
from typing import Dict
import pandas as pd
import numpy as np

from train_forecast import _build_forecaster, extend_future_features
from utils import mase, rmsse, smape, bias


def _naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    last = series.iloc[-1]
    return np.repeat(last, horizon)


def _build_bottom_up_baseline(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Generate bottom-up naive forecasts aggregated to upper levels."""
    bottom_mask = train_df["unique_id"].str.count("\\|") == 2
    bottom_hist = train_df[bottom_mask]
    if bottom_hist.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "yhat_bottomup"])
    last_ds = bottom_hist["ds"].max()
    future_dates = pd.date_range(last_ds + pd.Timedelta(days=7), periods=horizon, freq="W-MON")

    bottom_preds = []
    for uid, g in bottom_hist.groupby("unique_id"):
        y = g.sort_values("ds")["y"]
        fcst = _naive_forecast(y, horizon)
        bottom_preds.append(pd.DataFrame({"unique_id": uid, "ds": future_dates, "yhat_bottomup": fcst}))
    bottom_df = pd.concat(bottom_preds, ignore_index=True)

    # Parse IDs for aggregation
    id_parts = bottom_df["unique_id"].str.split("|", expand=True)
    bottom_df["product_id"] = id_parts[0]
    bottom_df["location_node_id"] = id_parts[1]

    sku_loc_preds = (
        bottom_df.groupby(["product_id", "location_node_id", "ds"], as_index=False)["yhat_bottomup"].sum()
        .assign(unique_id=lambda x: x["product_id"] + "|" + x["location_node_id"])
    )
    sku_preds = (
        bottom_df.groupby(["product_id", "ds"], as_index=False)["yhat_bottomup"].sum()
        .assign(unique_id=lambda x: x["product_id"])
    )
    total_preds = (
        bottom_df.groupby(["ds"], as_index=False)["yhat_bottomup"].sum()
        .assign(unique_id="Total")
    )

    all_preds = pd.concat(
        [bottom_df[["unique_id", "ds", "yhat_bottomup"]], sku_loc_preds[["unique_id", "ds", "yhat_bottomup"]], sku_preds[["unique_id", "ds", "yhat_bottomup"]], total_preds[["unique_id", "ds", "yhat_bottomup"]]],
        ignore_index=True,
    )
    return all_preds


def evaluate_predictions(df_feat: pd.DataFrame, forecast_horizon: int, config: Dict, holiday_calendar: Dict, gpu: bool) -> pd.DataFrame:
    results = []
    cfg_eval = config.get("evaluation", {})
    step = cfg_eval.get("rolling_step", 4)
    min_hist = cfg_eval.get("min_history_weeks", 26)
    test_horizon = cfg_eval.get("test_horizon", forecast_horizon)

    max_ds = df_feat["ds"].max()
    rolling_starts = pd.date_range(df_feat["ds"].min() + pd.Timedelta(weeks=min_hist), max_ds - pd.Timedelta(weeks=test_horizon), freq=f"{step}W")

    feature_cols = [c for c in df_feat.columns if c not in {"unique_id", "ds", "y"}]

    for origin in rolling_starts:
        sub_train = df_feat[df_feat["ds"] <= origin].copy()
        if sub_train.empty:
            continue

        # Refit model for each origin to honor rolling-origin evaluation
        fcst = _build_forecaster(config, gpu)
        fcst.fit(df=sub_train, id_col="unique_id", time_col="ds", target_col="y", X_df=sub_train[feature_cols])

        future = fcst.make_future_dataframe(sub_train, id_col="unique_id", time_col="ds", periods=test_horizon)
        future_feat = extend_future_features(future, sub_train, config, holiday_calendar)
        preds = fcst.predict(future_feat, X_df=future_feat[feature_cols])
        preds = preds.rename(columns={"LGBMRegressor": "yhat"})

        actuals = df_feat[df_feat["ds"].between(origin + pd.Timedelta(days=7), origin + pd.Timedelta(weeks=test_horizon))]
        merged = preds.merge(actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")

        naive = _build_bottom_up_baseline(sub_train[["unique_id", "ds", "y"]], test_horizon)
        merged = merged.merge(naive, on=["unique_id", "ds"], how="left")

        for uid, g in merged.groupby("unique_id"):
            y_true = g["y"].values
            if len(y_true) == 0:
                continue
            results.append(
                {
                    "origin": origin,
                    "unique_id": uid,
                    "mase": mase(y_true, g["yhat"].values, seasonal_period=1),
                    "rmsse": rmsse(y_true, g["yhat"].values, seasonal_period=1),
                    "smape": smape(y_true, g["yhat"].values),
                    "bias": bias(y_true, g["yhat"].values),
                    "mase_bottom_up": mase(y_true, g["yhat_bottomup"].values, seasonal_period=1),
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Evaluation module is orchestrated by generate_forecast.py")
