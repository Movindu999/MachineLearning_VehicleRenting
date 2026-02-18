import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from data_utils import load_csvs, is_season, is_weekend

# Cache trained models globally
TRAINED_MODELS = {}


def ensure_columns(df, cols):
    """Ensure all specified columns exist; add missing ones as 0."""
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    return df


# ---------------------------------------------------
# 1️⃣ Linear Regression – Monthly Revenue
# ---------------------------------------------------
def train_linear_regression_revenue():
    _, _, monthly = load_csvs()

    if monthly is None or monthly.empty or len(monthly) < 5:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    X = monthly[["TotalRentals", "AvgDailyPrice", "SeasonDays", "WeekendDays", "HighDemandDays"]].fillna(0)
    y = monthly["RevenueLKR"].fillna(0)

    if len(X) < 3:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # ✅ added

    return {
        "model": model,
        "r2": round(r2, 4),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "status": "trained"
    }


# ---------------------------------------------------
# 2️⃣ Logistic Regression – Daily Demand
# ---------------------------------------------------
def train_logistic_regression_demand():
    _, daily, _ = load_csvs()

    if daily is None or daily.empty or len(daily) < 5:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays", "DemandHigh"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays"]].fillna(0)
    y = daily["DemandHigh"].fillna(0)

    if len(X) < 3:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    # Ensure there are at least 2 classes (avoid solver error)
    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": "only one class in target"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": round(acc, 4),
        "status": "trained"
    }


# ---------------------------------------------------
# 3️⃣ KNN Regression – Daily Rentals Count
# ---------------------------------------------------
def train_knn_regressor_rentals():
    _, daily, _ = load_csvs()

    if daily is None or daily.empty or len(daily) < 10:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = daily["TotalRentals"].fillna(0)

    if len(X) < 5:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # ✅ added

    return {
        "model": model,
        "r2": round(r2, 4),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "status": "trained"
    }


# ---------------------------------------------------
# 4️⃣ SVM – Vehicle Type Classifier
# ---------------------------------------------------
def train_svm_classifier_vehicle_type():
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty or len(transactions) < 10:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    transactions = ensure_columns(transactions, ["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR", "VehicleType"])

    le_vehicle = LabelEncoder()
    y = le_vehicle.fit_transform(transactions["VehicleType"])

    X = transactions[["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR"]].fillna(0)

    if len(X) < 5 or len(np.unique(y)) < 2:
        return {"model": None, "accuracy": 0, "status": "insufficient vehicle classes"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": round(acc, 4),
        "le_vehicle": le_vehicle,
        "status": "trained"
    }


# ---------------------------------------------------
# 5️⃣ Decision Tree – Risk Prediction
# ---------------------------------------------------
def train_decision_tree_risk():
    _, daily, _ = load_csvs()

    if daily is None or daily.empty or len(daily) < 5:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])

    daily["RiskHigh"] = (
        ((daily["IsSeason"] == 1) & (daily["TotalRentals"] >= 6)) |
        ((daily["IsSeason"] == 0) & (daily["TotalRentals"] >= 4))
    ).astype(int)

    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = daily["RiskHigh"].fillna(0)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": "only one class in target"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": round(acc, 4),
        "status": "trained"
    }


# ---------------------------------------------------
# Train All Models
# ---------------------------------------------------
def train_all_models():
    global TRAINED_MODELS

    TRAINED_MODELS = {
        "linear_reg": train_linear_regression_revenue(),
        "logistic_reg": train_logistic_regression_demand(),
        "knn_reg": train_knn_regressor_rentals(),
        "svm_classifier": train_svm_classifier_vehicle_type(),
        "decision_tree": train_decision_tree_risk()
    }

    return TRAINED_MODELS


# ---------------------------------------------------
# ✅ PREDICT DAILY
# ---------------------------------------------------
def predict_daily(date_str, vehicle_type="All"):
    if not TRAINED_MODELS or "logistic_reg" not in TRAINED_MODELS:
        train_all_models()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    is_season_val = is_season(date_obj)
    is_weekend_val = is_weekend(date_obj)

    _, daily, _ = load_csvs()
    if daily is None or daily.empty:
        avg_price = 2500
        avg_rental_days = 2
    else:
        avg_price = float(daily.get("AvgDailyPrice", pd.Series([2500])).mean())
        avg_rental_days = float(daily.get("AvgRentalDays", pd.Series([2])).mean())

    # demand
    logistic_model = TRAINED_MODELS["logistic_reg"].get("model")
    if logistic_model:
        X_demo = np.array([[is_season_val, is_weekend_val, avg_price, avg_rental_days]])
        demand_pred = int(logistic_model.predict(X_demo)[0])
        tomorrow_demand = "High" if demand_pred == 1 else "Low"
    else:
        tomorrow_demand = "--"

    # count
    knn_model = TRAINED_MODELS["knn_reg"].get("model")
    if knn_model:
        X_knn = np.array([[is_season_val, is_weekend_val, avg_price]])
        count_pred = float(knn_model.predict(X_knn)[0])
        predicted_count = int(round(max(count_pred, 0)))
    else:
        predicted_count = "--"

    # price
    recommended_price = int(avg_price * (1.2 if is_season_val == 1 else 0.9))

    return {
        "tomorrow_demand": tomorrow_demand,
        "predicted_count": predicted_count,
        "recommended_price": recommended_price
    }


# ---------------------------------------------------
# ✅ PREDICT MONTHLY
# ---------------------------------------------------
def predict_monthly(month_str):
    if not TRAINED_MODELS or "linear_reg" not in TRAINED_MODELS:
        train_all_models()

    year, month = map(int, month_str.split("-"))
    is_season_val = 1 if (month >= 10 or month <= 3) else 0

    _, _, monthly = load_csvs()
    if monthly is None or monthly.empty:
        return {
            "pred_revenue": "--",
            "pred_demand": "--",
            "pred_rental_count": "--",
            "pred_vehicle_type": "--",
            "pred_risk": "--"
        }

    avg_total_rentals = float(monthly["TotalRentals"].mean())
    avg_daily_price = float(monthly["AvgDailyPrice"].mean())
    avg_season_days = float(monthly["SeasonDays"].mean())
    avg_weekend_days = float(monthly["WeekendDays"].mean())
    avg_high_demand_days = float(monthly["HighDemandDays"].mean())

    # revenue
    linear_model = TRAINED_MODELS["linear_reg"].get("model")
    if linear_model:
        X_linear = np.array([[avg_total_rentals, avg_daily_price, avg_season_days, avg_weekend_days, avg_high_demand_days]])
        revenue_pred = float(linear_model.predict(X_linear)[0])
        pred_revenue = int(max(revenue_pred, 0))
    else:
        pred_revenue = "--"

    # demand
    logistic_model = TRAINED_MODELS["logistic_reg"].get("model")
    if logistic_model:
        is_weekend_avg = avg_weekend_days / 30
        X_logistic = np.array([[is_season_val, is_weekend_avg, avg_daily_price, 2]])
        demand_pred = int(logistic_model.predict(X_logistic)[0])
        pred_demand = "High" if demand_pred == 1 else "Low"
    else:
        pred_demand = "--"

    # rentals count (monthly approx)
    knn_model = TRAINED_MODELS["knn_reg"].get("model")
    if knn_model:
        is_weekend_avg = avg_weekend_days / 30
        X_knn = np.array([[is_season_val, is_weekend_avg, avg_daily_price]])
        count_pred = float(knn_model.predict(X_knn)[0])
        pred_rental_count = int(round(max(count_pred * 30, 0)))
    else:
        pred_rental_count = "--"

    # vehicle type
    svm_model = TRAINED_MODELS["svm_classifier"].get("model")
    le_vehicle = TRAINED_MODELS["svm_classifier"].get("le_vehicle")
    if svm_model and le_vehicle:
        X_svm = np.array([[is_season_val, avg_weekend_days / 30, 2, avg_daily_price]])
        idx = int(svm_model.predict(X_svm)[0])
        pred_vehicle_type = le_vehicle.inverse_transform([idx])[0]
    else:
        pred_vehicle_type = "--"

    # risk
    dt_model = TRAINED_MODELS["decision_tree"].get("model")
    if dt_model:
        X_dt = np.array([[is_season_val, avg_weekend_days / 30, avg_daily_price]])
        risk_pred = int(dt_model.predict(X_dt)[0])
        pred_risk = "Yes" if risk_pred == 1 else "No"
    else:
        pred_risk = "--"

    return {
        "pred_revenue": pred_revenue,
        "pred_demand": pred_demand,
        "pred_rental_count": pred_rental_count,
        "pred_vehicle_type": pred_vehicle_type,
        "pred_risk": pred_risk
    }
