"""
CAP182 SS2 Part B
Feature Engineering

Input:
    data/raw_dataset.csv
Output:
    data/engineered_data.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("data/raw_dataset.csv")
OUTPUT_PATH = Path("data/engineered_data.csv")
TARGET_COLUMN = "returned"


def find_column(df, candidates):
    """Return the first matching column from a list of candidates."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def add_order_value(df):
    quantity = find_column(df, ["quantity"])
    unit_price = find_column(df, ["unit_price"])

    if quantity and unit_price:
        df["order_value"] = (
            pd.to_numeric(df[quantity], errors="coerce")
            * pd.to_numeric(df[unit_price], errors="coerce"))
        return True
    return False


def add_discount_percent(df):
    original_price = find_column(df, ["original_price"])
    selling_price = find_column(df, ["selling_price"])

    if original_price and selling_price:
        original = pd.to_numeric(df[original_price], errors="coerce")
        selling = pd.to_numeric(df[selling_price], errors="coerce")

        df["discount_percent"] = np.where(
            original.ne(0),
            ((original - selling) / original) * 100,
            0.0,)
        return True
    return False


def add_customer_history_features(df):
    customer_id = find_column(df, ["customer_id", "customerid", "customer id"])
    transaction_date = find_column(
        df,
        ["transaction_date", "order_date", "purchase_date", "date"])

    if not customer_id or not transaction_date:
        return False

    dates = pd.to_datetime(df[transaction_date], errors="coerce")
    working = df.copy()
    working["_transaction_date"] = dates

    # Stable sort so historical calculations use only earlier records.
    working["_original_row"] = np.arange(len(working))
    working = working.sort_values(
        [customer_id, "_transaction_date", "_original_row"],
        kind="stable")

    # Number of previous transactions for the customer.
    working["customer_purchase_frequency"] = (
        working.groupby(customer_id, dropna=False).cumcount())

    if TARGET_COLUMN in working.columns:
        target_numeric = pd.to_numeric(
            working[TARGET_COLUMN], errors="coerce")

        # Previous-return rate uses only transactions before the current row.
        previous_returns = (
            target_numeric.groupby(
                working[customer_id], dropna=False
            ).cumsum()
            - target_numeric.fillna(0))

        previous_count = (
            working.groupby(customer_id, dropna=False).cumcount())

        working["previous_return_rate"] = np.where(
            previous_count > 0,
            previous_returns / previous_count,
            0.0,)

    working = working.sort_values("_original_row", kind="stable")
    working = working.drop(
        columns=["_transaction_date", "_original_row"],
        errors="ignore")

    df.drop(columns=list(df.columns), inplace=True)
    for column in working.columns:
        df[column] = working[column]

    return True


def add_delivery_time(df):
    order_date = find_column(
        df,
        ["order_date", "transaction_date", "purchase_date"])
    delivery_date = find_column(
        df,
        ["delivery_date", "delivered_date"])

    if order_date and delivery_date:
        order = pd.to_datetime(df[order_date], errors="coerce")
        delivery = pd.to_datetime(df[delivery_date], errors="coerce")
        df["delivery_time"] = (delivery - order).dt.days
        return True

    return False


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Place the public dataset at data/raw_dataset.csv.")
    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    created = []
    if add_order_value(df):
        created.append("order_value")
    if add_discount_percent(df):
        created.append("discount_percent")
    if add_customer_history_features(df):
        if "customer_purchase_frequency" in df.columns:
            created.append("customer_purchase_frequency")
        if "previous_return_rate" in df.columns:
            created.append("previous_return_rate")
    if add_delivery_time(df):
        created.append("delivery_time")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeatures created:")
    if created:
        for feature in created:
            print(f"  - {feature}")
    else:
        print("  None. The required source columns were not found.")

    print(f"\nSaved engineered dataset to: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
