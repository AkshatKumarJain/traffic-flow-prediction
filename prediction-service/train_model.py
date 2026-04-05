import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "traffic.csv"
MODEL_PATH = Path(__file__).resolve().parent / "traffic_model.pkl"
METADATA_PATH = Path(__file__).resolve().parent / "model_metadata.json"
SPLIT_DATE = datetime(2017, 1, 1)

FEATURE_NAMES = [
    "hour",
    "day",
    "month",
    "weekday",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_2",
    "rolling_mean",
    "Junction_2",
    "Junction_3",
    "Junction_4",
]


def detect_delimiter(header_line: str) -> str:
    return "\t" if "\t" in header_line else ","


def load_rows():
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        delimiter = detect_delimiter(header)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = []

        for row in reader:
            rows.append(
                {
                    "DateTime": datetime.strptime(row["DateTime"], "%m/%d/%Y %H:%M"),
                    "Junction": int(row["Junction"]),
                    "Vehicles": float(row["Vehicles"]),
                }
            )

    rows.sort(key=lambda item: (item["Junction"], item["DateTime"]))
    return rows


def build_datasets(rows):
    history = defaultdict(lambda: deque(maxlen=3))
    x_train, y_train, x_test, y_test = [], [], [], []

    for row in rows:
        junction_history = history[row["Junction"]]

        if len(junction_history) >= 2:
            dt = row["DateTime"]
            lag_1 = junction_history[-1]
            lag_2 = junction_history[-2]
            rolling_mean = sum(junction_history) / len(junction_history)

            features = [
                dt.hour,
                dt.day,
                dt.month,
                dt.weekday(),
                math.sin(2 * math.pi * dt.hour / 24),
                math.cos(2 * math.pi * dt.hour / 24),
                lag_1,
                lag_2,
                rolling_mean,
                1 if row["Junction"] == 2 else 0,
                1 if row["Junction"] == 3 else 0,
                1 if row["Junction"] == 4 else 0,
            ]

            if dt < SPLIT_DATE:
                x_train.append(features)
                y_train.append(row["Vehicles"])
            else:
                x_test.append(features)
                y_test.append(row["Vehicles"])

        junction_history.append(row["Vehicles"])

    return x_train, y_train, x_test, y_test


def train():
    rows = load_rows()
    x_train, y_train, x_test, y_test = build_datasets(rows)

    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_type": type(model).__name__,
                "feature_names": FEATURE_NAMES,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "mae": mae,
                "rmse": rmse,
                "split_date": SPLIT_DATE.strftime("%Y-%m-%d"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train()
