"""Hierarchical reconciliation using MinTrace."""
from typing import Dict, List
import pandas as pd
import numpy as np
from hierarchicalforecast import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace


def reconcile_forecasts(
    base_fcst: pd.DataFrame,
    S: np.ndarray,
    tags: Dict[str, List[str]],
    node_order: List[str],
    method: str = "mint_shrink",
    shrink: bool = True,
) -> pd.DataFrame:
    print("Running MinTrace reconciliation ...")
    mintrace = MinTrace(method=method, shrink=shrink)
    recon = HierarchicalReconciliation(reconcilers=[mintrace])
    reconciled = recon.reconcile(base_fcst, S=S, tags=tags)
    # The reconcilers append column names like MinTrace-mint_shrink
    recon_col = [c for c in reconciled.columns if c.startswith("MinTrace")][0]
    result = reconciled[["unique_id", "ds", recon_col]].rename(columns={recon_col: "yhat_reconciled"})
    return result


if __name__ == "__main__":
    print("Reconciliation module should be orchestrated by generate_forecast.py")
