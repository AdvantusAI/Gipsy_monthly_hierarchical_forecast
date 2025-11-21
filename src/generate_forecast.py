"""Main orchestration script.

Run end-to-end offline forecasting:
    python src/generate_forecast.py
"""
import os
import sys
import yaml
import pandas as pd

# Allow imports from src when executed from project root
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)

from extract_supabase import load_supabase_data
from build_hierarchy import build_weekly_hierarchy
from train_forecast import prepare_features, train_model
from reconcile import reconcile_forecasts
from evaluate import evaluate_predictions


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)
    forecast_start = config.get("forecast_start_date")
    horizon = int(config.get("forecast_horizon", 26))
    gpu = bool(config.get("gpu", config.get("run_gpu", False)))

    print("1) Extracting data from Supabase ...")
    raw_df = load_supabase_data()

    print("2) Building weekly hierarchy ...")
    weekly_df, S, tags, node_order, bottom_index = build_weekly_hierarchy(raw_df, forecast_start)

    print("2a) Creating features and holiday flags ...")
    weekly_feat, holiday_calendar = prepare_features(weekly_df, config)

    print("3) Training LightGBM forecaster ...")
    fcst_model, base_fcst, importance_outputs = train_model(weekly_feat, holiday_calendar, horizon, config, gpu)

    # Save feature importances
    for imp_type, df_imp in importance_outputs.items():
        filename = "feature_importance.csv" if imp_type == "combined" else f"feature_importance_{imp_type}.csv"
        path = os.path.join(OUTPUT_DIR, filename)
        df_imp.to_csv(path, index=False)
        print(f"Saved feature importance CSV {path}")

    print("4) Reconciling hierarchy ...")
    recon_method_cfg = config.get("reconciliation_method", "MinTrace")
    recon_method = "mint_shrink" if recon_method_cfg == "MinTrace" else recon_method_cfg or "mint_shrink"
    reconciled = reconcile_forecasts(
        base_fcst,
        S,
        tags,
        node_order,
        method=recon_method,
        shrink=bool(config.get("shrinkage", True)),
    )
    forecasts_final = reconciled.copy()
    forecasts_final.to_parquet(os.path.join(OUTPUT_DIR, "forecasts_final.parquet"), index=False)
    print("Saved reconciled forecasts to output/forecasts_final.parquet")

    print("5) Rolling backtesting evaluation ...")
    eval_df = evaluate_predictions(weekly_feat, horizon, config, holiday_calendar, gpu)
    eval_path = os.path.join(OUTPUT_DIR, "evaluation_results.parquet")
    eval_df.to_parquet(eval_path, index=False)
    print(f"Saved evaluation results to {eval_path}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
