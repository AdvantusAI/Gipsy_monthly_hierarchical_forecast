"""Train LightGBM via MLForecast with feature importance exports."""
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler
from pandas.tseries.holiday import USFederalHolidayCalendar

from utils import create_fourier_features, add_calendar_features, create_holiday_flags

LAGS = [1, 2, 3, 4, 8, 12, 26, 52]


def _school_holiday_dates(years: List[int]) -> List[pd.Timestamp]:
    """Generate simple, offline-friendly school holiday windows."""
    dates: List[pd.Timestamp] = []
    for year in years:
        # Spring break: third Monday of March for 5 weekdays
        third_monday_march = pd.date_range(f"{year}-03-01", f"{year}-03-31", freq="W-MON")[2]
        dates.extend(pd.date_range(third_monday_march, periods=5, freq="D"))

        # Thanksgiving break: week containing 4th Thursday of November
        fourth_thursday = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="W-THU")[3]
        week_start = fourth_thursday - pd.Timedelta(days=3)
        dates.extend(pd.date_range(week_start, periods=7, freq="D"))

        # Winter break: Dec 24 through Jan 1
        dates.extend(pd.date_range(f"{year}-12-24", f"{year+1}-01-01", freq="D"))
    return [pd.Timestamp(d).normalize() for d in dates]


def _federal_holidays(years: List[int], country_code: str) -> List[pd.Timestamp]:
    """Return federal holidays for supported countries (US only by default)."""
    if country_code.upper() == "US":
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=f"{min(years)}-01-01", end=f"{max(years)}-12-31")
        return [pd.Timestamp(d).normalize() for d in holidays]
    # Fallback: no built-in holidays for other countries; rely on extras
    return []


def build_holiday_calendar(years: List[int], country_code: str, extra: List[str], add_school: bool) -> Dict[pd.Timestamp, str]:
    """Compose holiday calendar with federal, school, and extra holidays."""
    calendar: Dict[pd.Timestamp, str] = {}

    for d in _federal_holidays(years, country_code):
        calendar[d] = "federal"

    if add_school:
        for d in _school_holiday_dates(years):
            calendar[d] = "school"

    for d in extra:
        calendar[pd.to_datetime(d).normalize()] = "extra"

    return calendar


def prepare_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]:
    """Add calendar, seasonal, holiday, and optional exogenous features."""
    df_feat = df.copy().sort_values(["unique_id", "ds"])
    df_feat = add_calendar_features(df_feat)

    df_feat["t"] = df_feat.groupby("unique_id").cumcount()

    # Fourier seasonality
    for order, prefix, period in [
        (config["seasonality"].get("fourier_yearly_order", 6), "yearly", 52),
        (config["seasonality"].get("fourier_halfyear_order", 3), "halfyear", 26),
    ]:
        if order > 0:
            df_feat = df_feat.groupby("unique_id", group_keys=False).apply(
                lambda g: create_fourier_features(g, period=period, order=order, prefix=prefix)
            )

    if config["seasonality"].get("add_month_dummies", True):
        month_dummies = pd.get_dummies(df_feat["ds"].dt.month, prefix="month", drop_first=False)
        df_feat = pd.concat([df_feat, month_dummies], axis=1)

    holidays_cfg = config.get("holidays", {})
    years = list({d.year for d in df_feat["ds"]})
    holiday_calendar = build_holiday_calendar(
        years=years,
        country_code=holidays_cfg.get("country_code", "US"),
        extra=holidays_cfg.get("extra_holidays", []),
        add_school=holidays_cfg.get("add_school_holidays", True),
    )
    df_feat = create_holiday_flags(df_feat, holiday_calendar)

    # Exogenous lag features
    exog_cfg = config.get("exogenous", {})
    if exog_cfg.get("enabled", False):
        for v in exog_cfg.get("vars", []):
            lag_list = exog_cfg.get("lags", [])
            transforms = exog_cfg.get("transforms", {}).get(v, [])

            for l in lag_list:
                df_feat[f"{v}_lag{l}"] = df_feat.groupby("unique_id")[v].shift(l)
            for window in transforms:
                df_feat[f"{v}_rollmean_{window}"] = df_feat.groupby("unique_id")[v].transform(lambda s: s.rolling(window).mean())

    return df_feat, holiday_calendar


def _fill_exogenous_future(future: pd.DataFrame, history: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Forward-fill exogenous variables for future rows before lagging."""
    exog_cfg = config.get("exogenous", {})
    if not exog_cfg.get("enabled", False):
        return future

    vars_to_use = exog_cfg.get("vars", [])
    future_filled = future.copy()
    for var in vars_to_use:
        last_values = history.groupby("unique_id")[var].last()
        future_filled[var] = future_filled["unique_id"].map(last_values)
    return future_filled


def extend_future_features(future_df: pd.DataFrame, train_feat: pd.DataFrame, config: Dict, holiday_calendar: Dict) -> pd.DataFrame:
    """Add the same feature set to future periods used for forecasting."""
    future_feat = future_df.copy()

    # Continue time index t
    last_t = train_feat.groupby("unique_id")["t"].max()
    future_feat = future_feat.sort_values(["unique_id", "ds"])
    future_feat["t"] = future_feat.groupby("unique_id").cumcount()
    future_feat["t"] = future_feat.apply(lambda r: r["t"] + last_t.get(r["unique_id"], -1) + 1, axis=1)

    future_feat = add_calendar_features(future_feat)

    for order, prefix, period in [
        (config["seasonality"].get("fourier_yearly_order", 6), "yearly", 52),
        (config["seasonality"].get("fourier_halfyear_order", 3), "halfyear", 26),
    ]:
        if order > 0:
            future_feat = future_feat.groupby("unique_id", group_keys=False).apply(
                lambda g: create_fourier_features(g, period=period, order=order, prefix=prefix)
            )

    if config["seasonality"].get("add_month_dummies", True):
        month_dummies = pd.get_dummies(future_feat["ds"].dt.month, prefix="month", drop_first=False)
        future_feat = pd.concat([future_feat, month_dummies], axis=1)

    # Extend holiday calendar to future years if needed
    holidays_cfg = config.get("holidays", {})
    years_needed = list({d.year for d in pd.concat([train_feat["ds"], future_feat["ds"]])})
    holiday_calendar = build_holiday_calendar(
        years=years_needed,
        country_code=holidays_cfg.get("country_code", "US"),
        extra=holidays_cfg.get("extra_holidays", []),
        add_school=holidays_cfg.get("add_school_holidays", True),
    )
    future_feat = create_holiday_flags(future_feat, holiday_calendar)

    # Exogenous features
    future_feat = _fill_exogenous_future(future_feat, train_feat, config)
    if config.get("exogenous", {}).get("enabled", False):
        exog_cfg = config["exogenous"]
        for v in exog_cfg.get("vars", []):
            lag_list = exog_cfg.get("lags", [])
            transforms = exog_cfg.get("transforms", {}).get(v, [])

            hist_and_future = pd.concat([train_feat[["unique_id", "ds", v]], future_feat[["unique_id", "ds", v]]])
            hist_and_future = hist_and_future.sort_values(["unique_id", "ds"])

            for l in lag_list:
                hist_and_future[f"{v}_lag{l}"] = hist_and_future.groupby("unique_id")[v].shift(l)
            for window in transforms:
                hist_and_future[f"{v}_rollmean_{window}"] = hist_and_future.groupby("unique_id")[v].transform(
                    lambda s: s.rolling(window).mean()
                )

            future_feat = future_feat.drop(columns=[c for c in future_feat.columns if c.startswith(v + "_lag")], errors="ignore")
            future_feat = future_feat.drop(columns=[c for c in future_feat.columns if c.startswith(v + "_rollmean")], errors="ignore")
            future_feat = future_feat.merge(
                hist_and_future.loc[hist_and_future["ds"].isin(future_feat["ds"]),
                                     ["unique_id", "ds"] +
                                     [c for c in hist_and_future.columns if c.startswith(v + "_lag") or c.startswith(v + "_rollmean")]],
                on=["unique_id", "ds"],
                how="left",
            )

    return future_feat


def _build_forecaster(config: Dict, gpu: bool) -> MLForecast:
    lgbm_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        device_type="gpu" if gpu else "cpu",
    )
    model = LGBMRegressor(**lgbm_params)

    fcst = MLForecast(
        models=[model],
        freq="W-MON",
        lags=LAGS,
        lag_transforms={
            1: [
                ("rolling_mean_4", lambda s: s.rolling(4).mean()),
                ("rolling_std_4", lambda s: s.rolling(4).std()),
            ],
            4: [("rolling_mean_4", lambda s: s.rolling(4).mean())],
            52: [("yoy_diff", lambda s: s - s.shift(52))],
        },
        date_features=[],
        target_transforms=[LocalStandardScaler()],
    )
    return fcst


def train_model(df_feat: pd.DataFrame, holiday_calendar: Dict[pd.Timestamp, str], forecast_horizon: int, config: Dict, gpu: bool) -> Tuple[MLForecast, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Fit the MLForecast model and produce base forecasts plus feature importance."""
    feature_cols = [c for c in df_feat.columns if c not in {"unique_id", "ds", "y"}]
    fcst = _build_forecaster(config, gpu)

    print(f"Training with {len(feature_cols)} engineered features and lags {LAGS}")
    fcst.fit(df=df_feat, id_col="unique_id", time_col="ds", target_col="y", X_df=df_feat[feature_cols])

    future = fcst.make_future_dataframe(df_feat, id_col="unique_id", time_col="ds", periods=forecast_horizon)
    future_feat = extend_future_features(future, df_feat, config, holiday_calendar)
    preds = fcst.predict(future_feat, X_df=future_feat[feature_cols])
    base_fcst = preds.rename(columns={"unique_id": "unique_id", "ds": "ds", "LGBMRegressor": "yhat"})

    # Feature importance
    importance_outputs: Dict[str, pd.DataFrame] = {}
    booster = fcst.models_[0].booster_
    for imp_type in config.get("importance", {}).get("types", ["gain", "split"]):
        df_imp = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance": booster.feature_importance(importance_type=imp_type),
                "type": imp_type,
            }
        ).sort_values("importance", ascending=False)
        importance_outputs[imp_type] = df_imp

        top_n = config.get("importance", {}).get("top_n", 20)
        top_df = df_imp.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_df["feature"], top_df["importance"], color="steelblue")
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_n} {imp_type} importance")
        plt.tight_layout()
        fig_path = os.path.join(os.path.dirname(__file__), "..", "output", f"feature_importance_{imp_type}.png")
        plt.savefig(fig_path)
        print(f"Saved feature importance plot {fig_path}")

    # Combined CSV across types
    combined = pd.concat(importance_outputs.values(), ignore_index=True)
    importance_outputs["combined"] = combined

    return fcst, base_fcst, importance_outputs


if __name__ == "__main__":
    print("Module is designed to be orchestrated by generate_forecast.py")
