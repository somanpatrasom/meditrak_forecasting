import pandas as pd
import numpy as np
import joblib

def build_future_calendar(start_date, days=30):
    """Builds a simple future calendar. Replace is_holiday/is_promo
    with your actual known upcoming dates when available."""
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.day_name()
    cal["is_weekend"] = (cal["date"].dt.weekday >= 5).astype(int)
    cal["is_holiday"] = 0   # plug in known future holidays here
    cal["is_promo"] = 0     # plug in planned promotions here
    return cal


def predict_next_month():
    bundle = joblib.load("models/demand_model.pkl")
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    items = pd.read_csv("data/item_info.csv")
    sales = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
    stores = sales["store_id"].unique()

    last_date = sales["date"].max()
    future_cal = build_future_calendar(last_date + pd.Timedelta(days=1), days=30)

    results = []

    for store in stores:
        for _, item in items.iterrows():
            item_id = item["item_id"]
            hist = sales[(sales["store_id"] == store) & (sales["item_id"] == item_id)]
            hist = hist.sort_values("date")

            # seed recent history for lag/rolling features
            recent_sales = list(hist["units_sold"].tail(7))
            if len(recent_sales) == 0:
                recent_sales = [0]

            for _, row in future_cal.iterrows():
                lag_1 = recent_sales[-1]
                lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales[0]
                rolling_7 = np.mean(recent_sales[-7:])

                record = {
                    "base_price": item["base_price"],
                    "is_weekend": row["is_weekend"],
                    "is_holiday": row["is_holiday"],
                    "is_promo": row["is_promo"],
                    "day_of_week_num": row["date"].dayofweek,
                    "month": row["date"].month,
                    "year": row["date"].year,
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "rolling_7_avg": rolling_7,
                }

                # one-hot encode category & day_of_week to match training columns
                for col in feature_cols:
                    if col.startswith("category_") and col == f"category_{item['category']}":
                        record[col] = 1
                    elif col.startswith("day_of_week_") and col == f"day_of_week_{row['day_of_week']}":
                        record[col] = 1
                    elif col not in record:
                        record[col] = 0

                X_row = pd.DataFrame([record])[feature_cols]
                pred = max(0, model.predict(X_row)[0])

                results.append({
                    "date": row["date"],
                    "store_id": store,
                    "item_id": item_id,
                    "category": item["category"],
                    "predicted_units": round(pred, 1)
                })

                recent_sales.append(pred)

    forecast_df = pd.DataFrame(results)

    # Monthly total per store/item = the actual "inventory list to order"
    inventory_list = (
        forecast_df.groupby(["store_id", "item_id", "category"])["predicted_units"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_units": "predicted_monthly_demand"})
    )
    inventory_list["predicted_monthly_demand"] = inventory_list["predicted_monthly_demand"].round(0)
    inventory_list = inventory_list.sort_values(
        ["store_id", "predicted_monthly_demand"], ascending=[True, False]
    )

    inventory_list.to_csv("data/predicted_inventory_next_month.csv", index=False)
    forecast_df.to_csv("data/daily_forecast_next_month.csv", index=False)
    print("Saved predicted_inventory_next_month.csv")
    return inventory_list


if __name__ == "__main__":
    predict_next_month()