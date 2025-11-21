"""Data extraction from Supabase (PostgreSQL).
All downstream steps must run completely offline after this pull.
"""
import os
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine


DEFAULT_TABLE = "daily_demand"


def load_supabase_data(table: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Load daily demand data from Supabase into a DataFrame.

    Parameters
    ----------
    table: str
        Table name containing the daily data. Defaults to DEFAULT_TABLE.
    limit: int
        Optional row limit for debugging.
    """
    conn_str = os.getenv("SUPABASE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("SUPABASE_CONNECTION_STRING environment variable is required for extraction")

    table_name = table or DEFAULT_TABLE
    engine = create_engine(conn_str)

    query = f"SELECT ds, quantity_clean, product_id, location_node_id, customer_node_id, unit_price FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"

    print(f"Pulling data from Supabase table {table_name} ...")
    df = pd.read_sql(query, con=engine)
    print(f"Extracted {len(df):,} rows")
    return df


def main():
    df = load_supabase_data()
    output_path = os.path.join(os.path.dirname(__file__), "..", "output", "raw_supabase.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved raw pull to {output_path}")


if __name__ == "__main__":
    main()
