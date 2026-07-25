import pandas as pd
import numpy as np

def load_and_merge():
    sales = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
    calendar = pd.read_csv("data/calendar_data.csv", parse_dates=["date"])
    items = pd.read_csv("data/item_info.csv")

    df = sales.merge(calendar, on="date", how="left")
    df = df.merge(items, on="item_id", how="left")

    return df


def handle_missing_and_outliers(df):
    # Missing values

    df["units_sold"] = df.groupby("item_id")["units_sold"].transform(
        lambda x: x.fillna(x.median())
    )
    # Drop any row still missing critical identifiers
    df = df.dropna(subset=["date", "store_id", "item_id"])

    # Outliers
    
    caps = df.groupby("item_id")["units_sold"].transform(lambda x: x.quantile(0.99))
    df["units_sold"] = np.where(df["units_sold"] > caps, caps, df["units_sold"])

    return df


def engineer_features(df):
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_promo"] = df["is_promo"].astype(int)

    df = df.sort_values(["store_id", "item_id", "date"])
    df["lag_1"] = df.groupby(["store_id", "item_id"])["units_sold"].shift(1)
    df["lag_7"] = df.groupby(["store_id", "item_id"])["units_sold"].shift(7)

    df["rolling_7_avg"] = (
        df.groupby(["store_id", "item_id"])["units_sold"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )

    df[["lag_1", "lag_7", "rolling_7_avg"]] = df[["lag_1", "lag_7", "rolling_7_avg"]].fillna(0)

    df = pd.get_dummies(df, columns=["category", "day_of_week"], drop_first=True)

    return df


def run_pipeline():
    df = load_and_merge()
    df = handle_missing_and_outliers(df)
    df = engineer_features(df)
    df.to_csv("data/processed_sales.csv", index=False)
    print(f"Processed dataset saved: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()