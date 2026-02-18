import pandas as pd
import numpy as np
import math
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

VEHICLE_TYPES = ["All", "Bike", "Car", "Tuk Tuk"]


def ensure_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    return df


def build_daily_from_transactions(transactions: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    """
    Build a daily dataset from transactions filtered by vehicle_type.
    Output columns match what models expect.
    """
    if transactions is None or transactions.empty:
        return pd.DataFrame()

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"])

    if vehicle_type != "All":
        tx = tx[tx["VehicleType"] == vehicle_type]

    if tx.empty:
        return pd.DataFrame()

    daily = tx.groupby("Date").agg(
        TotalRentals=("RentalID", "count"),
        RevenueLKR=("TotalPriceLKR", "sum"),
        AvgDailyPrice=("DailyPriceLKR", "mean"),
        AvgRentalDays=("RentalDays", "mean"),
        IsSeason=("IsSeason", "max"),
        IsWeekend=("IsWeekend", "max"),
    ).reset_index()

    daily["DemandHigh"] = (daily["TotalRentals"] >= 6).astype(int)
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month

    return daily


# ---------------------------------------------------
# 1) Linear Regression – Monthly Revenue (All only)
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
    rmse = np.sqrt(mse)

    return {
        "model": model,
        "r2": round(r2, 4),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "status": "trained"
    }


# ---------------------------------------------------
# 2) Logistic Regression – Demand (per vehicle type)
# ---------------------------------------------------
def train_logistic_regression_demand(vehicle_type="All"):
    transactions, daily_all, _ = load_csvs()

    if vehicle_type == "All":
        daily = daily_all
    else:
        daily = build_daily_from_transactions(transactions, vehicle_type)

    if daily is None or daily.empty or len(daily) < 8:
        return {"model": None, "accuracy": 0, "status": f"insufficient data ({vehicle_type})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays", "DemandHigh"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays"]].fillna(0)
    y = daily["DemandHigh"].fillna(0)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": f"only one class ({vehicle_type})"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": round(acc, 4), "status": "trained"}


# ---------------------------------------------------
# 3) KNN Regression – Rentals Count (per vehicle type)
# ---------------------------------------------------
def train_knn_regressor_rentals(vehicle_type="All"):
    transactions, daily_all, _ = load_csvs()

    if vehicle_type == "All":
        daily = daily_all
    else:
        daily = build_daily_from_transactions(transactions, vehicle_type)

    if daily is None or daily.empty or len(daily) < 10:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": f"insufficient data ({vehicle_type})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = daily["TotalRentals"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "model": model,
        "r2": round(r2, 4),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "status": "trained"
    }


# ---------------------------------------------------
# 4) SVM – Vehicle Type (All only)
# ---------------------------------------------------
def train_svm_classifier_vehicle_type():
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty or len(transactions) < 20:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    transactions = ensure_columns(transactions, ["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR", "VehicleType"])

    le_vehicle = LabelEncoder()
    y = le_vehicle.fit_transform(transactions["VehicleType"])
    X = transactions[["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR"]].fillna(0)

    if len(np.unique(y)) < 2:
        return {"model": None, "accuracy": 0, "status": "insufficient vehicle classes"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": round(acc, 4), "le_vehicle": le_vehicle, "status": "trained"}


# ---------------------------------------------------
# 5) Decision Tree – Risk (All only)
# ---------------------------------------------------
def train_decision_tree_risk():
    _, daily, _ = load_csvs()
    if daily is None or daily.empty or len(daily) < 10:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])
    daily["RiskHigh"] = (
        ((daily["IsSeason"] == 1) & (daily["TotalRentals"] >= 6)) |
        ((daily["IsSeason"] == 0) & (daily["TotalRentals"] >= 4))
    ).astype(int)

    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = daily["RiskHigh"].fillna(0)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": "only one class"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": round(acc, 4), "status": "trained"}


def train_all_models():
    global TRAINED_MODELS

    logistic_by_type = {}
    knn_by_type = {}

    for vt in VEHICLE_TYPES:
        logistic_by_type[vt] = train_logistic_regression_demand(vt)
        knn_by_type[vt] = train_knn_regressor_rentals(vt)

    TRAINED_MODELS = {
        "linear_reg": train_linear_regression_revenue(),
        "logistic_reg": logistic_by_type,
        "knn_reg": knn_by_type,
        "svm_classifier": train_svm_classifier_vehicle_type(),
        "decision_tree": train_decision_tree_risk()
    }
    return TRAINED_MODELS


def predict_daily(date_str, vehicle_type="All"):
    if not TRAINED_MODELS:
        train_all_models()

    vehicle_type = vehicle_type if vehicle_type in VEHICLE_TYPES else "All"

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    is_season_val = is_season(date_obj)
    is_weekend_val = is_weekend(date_obj)

    transactions, daily_all, _ = load_csvs()
    if vehicle_type == "All":
        daily = daily_all
    else:
        daily = build_daily_from_transactions(transactions, vehicle_type)

    if daily is None or daily.empty:
        avg_price = 2500
        avg_rental_days = 2
    else:
        avg_price = float(daily["AvgDailyPrice"].mean())
        avg_rental_days = float(daily.get("AvgRentalDays", pd.Series([2])).mean())

    # Logistic model (by type)
    logistic_model = TRAINED_MODELS["logistic_reg"].get(vehicle_type, {}).get("model")
    if logistic_model:
        X_demo = np.array([[is_season_val, is_weekend_val, avg_price, avg_rental_days]])
        demand_pred = int(logistic_model.predict(X_demo)[0])
        tomorrow_demand = "High" if demand_pred == 1 else "Low"
    else:
        tomorrow_demand = "--"

    # KNN model (by type)
    knn_model = TRAINED_MODELS["knn_reg"].get(vehicle_type, {}).get("model")
    if knn_model:
        X_knn = np.array([[is_season_val, is_weekend_val, avg_price]])
        count_pred = float(knn_model.predict(X_knn)[0])
        predicted_count = int(round(max(count_pred, 0)))
    else:
        predicted_count = "--"

    # Recommended price
    type_multiplier = {"All": 1.0, "Bike": 0.9, "Car": 1.2, "Tuk Tuk": 1.0}
    season_factor = 1.2 if is_season_val == 1 else 0.9
    recommended_price = int(avg_price * season_factor * type_multiplier.get(vehicle_type, 1.0))

    return {
        "vehicle_type": vehicle_type,
        "tomorrow_demand": tomorrow_demand,
        "predicted_count": predicted_count,
        "recommended_price": recommended_price
    }


def predict_monthly(month_str: str):
    """
    Month format: YYYY-MM
    Returns keys matching your frontend:
    pred_revenue, pred_demand, pred_rental_count, pred_vehicle_type, pred_risk
    """
    if not TRAINED_MODELS:
        train_all_models()

    month_dt = datetime.strptime(month_str, "%Y-%m")
    year = month_dt.year
    month = month_dt.month

    season_val = 1 if (month >= 10 or month <= 3) else 0

    # month day range
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    days = pd.date_range(start, end, freq="D")

    weekend_days = int((days.weekday >= 5).sum())
    season_days = int(season_val * len(days))

    transactions, daily, monthly = load_csvs()

    # fallback averages
    avg_price = 2500.0
    avg_rental_days = 2.0
    avg_total_rentals_per_day = 4.0

    if daily is not None and not daily.empty:
        if "AvgDailyPrice" in daily.columns:
            avg_price = float(pd.to_numeric(daily["AvgDailyPrice"], errors="coerce").dropna().mean() or avg_price)
        if "AvgRentalDays" in daily.columns:
            avg_rental_days = float(pd.to_numeric(daily["AvgRentalDays"], errors="coerce").dropna().mean() or avg_rental_days)
        if "TotalRentals" in daily.columns:
            avg_total_rentals_per_day = float(pd.to_numeric(daily["TotalRentals"], errors="coerce").dropna().mean() or avg_total_rentals_per_day)

    # ---- Linear Regression monthly revenue
    lin = TRAINED_MODELS.get("linear_reg", {}).get("model")
    if lin is not None:
        est_total_rentals = int(round(avg_total_rentals_per_day * len(days)))
        est_high_demand_days = int(round(0.35 * len(days)))

        X_lin = pd.DataFrame([{
            "TotalRentals": est_total_rentals,
            "AvgDailyPrice": avg_price,
            "SeasonDays": season_days,
            "WeekendDays": weekend_days,
            "HighDemandDays": est_high_demand_days
        }])

        pred_revenue_val = float(lin.predict(X_lin)[0])
        pred_revenue = f"Rs {int(max(pred_revenue_val, 0)):,}"
    else:
        pred_revenue = "--"

    # ---- Logistic demand (All)
    log = TRAINED_MODELS.get("logistic_reg", {}).get("All", {}).get("model")
    if log is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_log = np.array([[season_val, is_weekend_avg, avg_price, avg_rental_days]])
        demand_pred = int(log.predict(X_log)[0])
        pred_demand = "High" if demand_pred == 1 else "Low"
    else:
        pred_demand = "--"

    # ---- KNN rental count (All)
    knn = TRAINED_MODELS.get("knn_reg", {}).get("All", {}).get("model")
    if knn is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_knn = np.array([[season_val, is_weekend_avg, avg_price]])
        per_day = float(knn.predict(X_knn)[0])
        pred_rental_count_val = int(max(round(per_day * len(days)), 0))
        pred_rental_count = str(pred_rental_count_val)
    else:
        pred_rental_count = "--"

    # ---- SVM vehicle type recommendation
    svm_pack = TRAINED_MODELS.get("svm_classifier", {})
    svm = svm_pack.get("model")
    le_vehicle = svm_pack.get("le_vehicle")

    if svm is not None and le_vehicle is not None:
        is_weekend_flag = 1 if (weekend_days / max(len(days), 1)) > 0.25 else 0
        X_svm = np.array([[season_val, is_weekend_flag, avg_rental_days, avg_price]])
        pred_class = int(svm.predict(X_svm)[0])
        pred_vehicle_type = str(le_vehicle.inverse_transform([pred_class])[0])
    else:
        pred_vehicle_type = "--"

    # ---- Decision tree risk
    dt = TRAINED_MODELS.get("decision_tree", {}).get("model")
    if dt is not None:
        is_weekend_flag = 1 if (weekend_days / max(len(days), 1)) > 0.25 else 0
        X_dt = np.array([[season_val, is_weekend_flag, avg_price]])
        risk_pred = int(dt.predict(X_dt)[0])
        pred_risk = "High Risk" if risk_pred == 1 else "Low Risk"
    else:
        pred_risk = "--"

    return {
        "month": month_str,
        "pred_revenue": pred_revenue,
        "pred_demand": pred_demand,
        "pred_rental_count": pred_rental_count,
        "pred_vehicle_type": pred_vehicle_type,
        "pred_risk": pred_risk
    }
