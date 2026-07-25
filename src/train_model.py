import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    df = pd.read_csv("data/processed_sales.csv", parse_dates=["date"])

    # Target variable
    y = df["units_sold"]

    # Drop columns that shouldn't be model inputs
    drop_cols = ["units_sold", "date", "store_id", "item_id"]
    X = df.drop(columns=drop_cols)

    # Time-based split: train on earlier data, test on the most recent 20%
    # (this mimics real forecasting -- never test on data "before" training data)
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # --- Model: Random Forest Regressor ---
    # Chosen over plain linear regression because demand has non-linear
    # interactions (holiday x category, promo x weekend, etc.)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=== Model Performance ===")
    print(f"MAE  (avg units off per prediction): {mae:.2f}")
    print(f"RMSE (penalizes big misses more):    {rmse:.2f}")
    print(f"R²   (variance explained):           {r2:.3f}")

    # Save evaluation report
    with open("models/performance_report.txt", "w", encoding="utf-8") as f:
        f.write("MEDITRAK DEMAND FORECASTING - MODEL PERFORMANCE REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f} units\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} units\n")
        f.write(f"R² Score: {r2:.3f}\n")
        f.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # --- Feature importance (Sales Trend Analysis Model) ---
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh")
    plt.title("Top Drivers of Medicine Demand")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    print("Saved feature_importance.png (trend analysis)")

    # --- Actual vs Predicted plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, preds, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.title("Actual vs Predicted Demand")
    plt.tight_layout()
    plt.savefig("models/actual_vs_predicted.png")

    # Save model + the exact feature column order it expects
    joblib.dump({"model": model, "feature_columns": list(X.columns)},
                "models/demand_model.pkl")
    print("Model saved to models/demand_model.pkl")

    return model


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train()