import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Item Info

items = pd.DataFrame({
    "item_id": [f"MED{i:03d}" for i in range(1, 21)],
    "category": np.random.choice(
        ["Antibiotics", "Painkillers", "Vitamins", "Diabetes", "Cardiac", "Antacids"],
        size=20
    ),
    "base_price": np.round(np.random.uniform(10, 500, size=20), 2)
})
items.to_csv("data/item_info.csv", index=False)

# 2. Calendar Data

dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
calendar = pd.DataFrame({"date": dates})
calendar["day_of_week"] = calendar["date"].dt.day_name()
calendar["is_weekend"] = calendar["date"].dt.weekday >= 5


holidays = ["2023-01-26", "2023-08-15", "2023-10-02", "2023-12-25",
            "2024-01-26", "2024-08-15", "2024-10-02", "2024-12-25"]
calendar["is_holiday"] = calendar["date"].astype(str).isin(holidays)

calendar["is_promo"] = np.random.choice([0, 1], size=len(calendar), p=[0.9, 0.1])
calendar.to_csv("data/calendar_data.csv", index=False)

# 3. Sales Transaction Data

stores = [f"STORE{i}" for i in range(1, 6)]
rows = []

for date in dates:
    cal_row = calendar[calendar["date"] == date].iloc[0]
    for store in stores:
        for _, item in items.iterrows():
            base_demand = np.random.poisson(lam=15)
            # seasonal boost for winter months (flu/cold medicines)
            if date.month in [11, 12, 1, 2] and item["category"] in ["Antibiotics", "Vitamins"]:
                base_demand += np.random.poisson(lam=8)
            # weekend dip
            if cal_row["is_weekend"]:
                base_demand = int(base_demand * 0.8)
            # holiday spike (people stock up before holidays)
            if cal_row["is_holiday"]:
                base_demand = int(base_demand * 1.5)
            # promo boost
            if cal_row["is_promo"]:
                base_demand = int(base_demand * 1.3)

            units_sold = max(0, base_demand + np.random.randint(-3, 3))
            rows.append([date, store, item["item_id"], units_sold])

sales = pd.DataFrame(rows, columns=["date", "store_id", "item_id", "units_sold"])
sales.to_csv("data/sales_data.csv", index=False)

print("Data generated:")
print(f"  sales_data.csv: {sales.shape}")
print(f"  calendar_data.csv: {calendar.shape}")
print(f"  item_info.csv: {items.shape}")