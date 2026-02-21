import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_utils import load_csvs, is_season, is_weekend, normalize_vehicle_type

TRAINED_MODELS = {}

VEHICLE_TYPES = ["All", "Bike", "Car", "Tuk Tuk"]
REAL_TYPES = ["Bike", "Car", "Tuk Tuk"]


def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df


def _demand_threshold(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1
    thr = float(s.quantile(0.70))
    return max(1, int(round(thr)))


def pretty_vehicle_type(v: str) -> str:
    return normalize_vehicle_type(v)


def build_daily_from_transactions(transactions: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    if transactions is None or transactions.empty:
        return pd.DataFrame()

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])

    if "VehicleType" in tx.columns:
        tx["VehicleType"] = tx["VehicleType"].apply(normalize_vehicle_type)

    vt = normalize_vehicle_type(vehicle_type)
    if vt != "All":
        tx = tx[tx["VehicleType"] == vt]

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

    thr = _demand_threshold(daily["TotalRentals"])
    daily["DemandHigh"] = (daily["TotalRentals"] >= thr).astype(int)

    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    return daily


# ============================================================
# 1) MULTIPLE LINEAR REGRESSION (Monthly Revenue)  ✅ lecture style
# ============================================================
def train_linear_regression_revenue():
    _, _, monthly = load_csvs()

    if monthly is None or monthly.empty or len(monthly) < 8:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    monthly = ensure_columns(monthly, ["TotalRentals", "AvgDailyPrice", "SeasonDays", "WeekendDays", "HighDemandDays", "RevenueLKR"])

    X = monthly[["TotalRentals", "AvgDailyPrice", "SeasonDays", "WeekendDays", "HighDemandDays"]].fillna(0)
    y = monthly["RevenueLKR"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {"model": model, "r2": round(r2, 4), "mse": round(mse, 2), "rmse": round(rmse, 2), "status": "trained"}


# ============================================================
# 2) LOGISTIC REGRESSION (Demand High/Low) ✅ lecture style
# - with StandardScaler (like your notebooks)
# ============================================================
def train_logistic_regression_demand(vehicle_type="All"):
    transactions, daily_all, _ = load_csvs()
    vt = normalize_vehicle_type(vehicle_type)

    if vt == "All":
        daily = daily_all.copy() if daily_all is not None else None
        if daily is not None and not daily.empty:
            thr = _demand_threshold(daily["TotalRentals"])
            daily["DemandHigh"] = (pd.to_numeric(daily["TotalRentals"], errors="coerce").fillna(0) >= thr).astype(int)
    else:
        daily = build_daily_from_transactions(transactions, vt)

    if daily is None or daily.empty or len(daily) < 12:
        return {"model": None, "scaler": None, "accuracy": 0, "status": f"insufficient data ({vt})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays", "DemandHigh"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays"]].fillna(0)
    y = daily["DemandHigh"].fillna(0).astype(int)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "scaler": None, "accuracy": 0, "status": f"only one class ({vt})"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "scaler": scaler, "accuracy": round(acc, 4), "status": "trained"}


# ============================================================
# 3) KNN REGRESSION (Daily Rentals Count) ✅ lecture style
# ============================================================
def train_knn_regressor_rentals(vehicle_type="All"):
    transactions, daily_all, _ = load_csvs()
    vt = normalize_vehicle_type(vehicle_type)

    daily = daily_all if vt == "All" else build_daily_from_transactions(transactions, vt)

    if daily is None or daily.empty or len(daily) < 15:
        return {"model": None, "scaler": None, "r2": 0, "mse": 0, "rmse": 0, "status": f"insufficient data ({vt})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = pd.to_numeric(daily["TotalRentals"], errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=7)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {"model": model, "scaler": scaler, "r2": round(r2, 4), "mse": round(mse, 2), "rmse": round(rmse, 2), "status": "trained"}


# ============================================================
# 4) SVM CLASSIFIER (Vehicle Type) ✅ lecture style
# ============================================================
def train_svm_classifier_vehicle_type():
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty or len(transactions) < 60:
        return {"model": None, "scaler": None, "accuracy": 0, "status": "insufficient data"}

    transactions = ensure_columns(transactions, ["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR", "VehicleType"])
    transactions["VehicleType"] = transactions["VehicleType"].apply(normalize_vehicle_type)

    X = transactions[["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR"]].fillna(0)
    y_text = transactions["VehicleType"].astype(str)

    le_vehicle = LabelEncoder()
    y = le_vehicle.fit_transform(y_text)

    if len(np.unique(y)) < 2:
        return {"model": None, "scaler": None, "accuracy": 0, "status": "insufficient vehicle classes"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "scaler": scaler, "accuracy": round(acc, 4), "le_vehicle": le_vehicle, "status": "trained"}


# ============================================================
# 5) DECISION TREE CLASSIFICATION (Risk High/Low) ✅ lecture style
# ============================================================
def train_decision_tree_risk():
    _, daily, _ = load_csvs()

    if daily is None or daily.empty or len(daily) < 20:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])

    thr = _demand_threshold(daily["TotalRentals"])
    daily["RiskHigh"] = (pd.to_numeric(daily["TotalRentals"], errors="coerce").fillna(0) >= thr).astype(int)

    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = daily["RiskHigh"].fillna(0).astype(int)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": "only one class"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": round(acc, 4), "status": "trained"}


# ============================================================
# TRAIN ALL
# ============================================================
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


# ============================================================
# HELPERS FOR PREDICT
# ============================================================
def _avg_stats_for_type(transactions, daily_all, vehicle_type):
    vt = normalize_vehicle_type(vehicle_type)
    daily = daily_all if vt == "All" else build_daily_from_transactions(transactions, vt)

    if daily is None or daily.empty:
        return 2500.0, 2.0

    avg_price = float(pd.to_numeric(daily["AvgDailyPrice"], errors="coerce").dropna().mean() or 2500.0)
    avg_days = float(pd.to_numeric(daily.get("AvgRentalDays", pd.Series([2])), errors="coerce").dropna().mean() or 2.0)
    return avg_price, avg_days


def _recommended_price(avg_price, is_season_val, vehicle_type):
    vt = normalize_vehicle_type(vehicle_type)
    type_multiplier = {"Bike": 0.9, "Car": 1.25, "Tuk Tuk": 1.0, "All": 1.0}
    season_factor = 1.2 if is_season_val == 1 else 0.9
    return int(round(avg_price * season_factor * type_multiplier.get(vt, 1.0)))


# ============================================================
# DAILY PREDICT
# ============================================================
def predict_daily(date_str, vehicle_type="All"):
    if not TRAINED_MODELS:
        train_all_models()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    is_season_val = is_season(date_obj)
    is_weekend_val = is_weekend(date_obj)

    transactions, daily_all, _ = load_csvs()

    vt = normalize_vehicle_type(vehicle_type)
    if vt not in VEHICLE_TYPES:
        vt = "All"

    avg_price, avg_days = _avg_stats_for_type(transactions, daily_all, vt)

    # Logistic (Demand)
    pack_log = TRAINED_MODELS["logistic_reg"].get(vt, {})
    log_model = pack_log.get("model")
    log_scaler = pack_log.get("scaler")

    if log_model is not None and log_scaler is not None:
        X_log = np.array([[is_season_val, is_weekend_val, avg_price, avg_days]])
        X_log_s = log_scaler.transform(X_log)
        demand_pred = int(log_model.predict(X_log_s)[0])
        demand = "High" if demand_pred == 1 else "Low"
    else:
        demand = "--"

    # KNN (Rentals count)
    pack_knn = TRAINED_MODELS["knn_reg"].get(vt, {})
    knn_model = pack_knn.get("model")
    knn_scaler = pack_knn.get("scaler")

    if knn_model is not None and knn_scaler is not None:
        X_knn = np.array([[is_season_val, is_weekend_val, avg_price]])
        X_knn_s = knn_scaler.transform(X_knn)
        count_pred = float(knn_model.predict(X_knn_s)[0])
        count = int(round(max(count_pred, 0)))
    else:
        count = "--"

    price = _recommended_price(avg_price, is_season_val, vt)
    revenue = int(round((0 if count == "--" else count) * price))

    return {
        "date": date_str,
        "vehicle_type": pretty_vehicle_type(vt),
        "predicted_demand": demand,
        "predicted_count": count,
        "recommended_price": price,
        "predicted_revenue": revenue
    }


# ============================================================
# MONTHLY PREDICT
# ============================================================
def predict_monthly(month_str: str):
    if not TRAINED_MODELS:
        train_all_models()

    month_dt = datetime.strptime(month_str, "%Y-%m")
    year = month_dt.year
    month = month_dt.month
    season_val = 1 if (month >= 10 or month <= 3) else 0

    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    days = pd.date_range(start, end, freq="D")
    weekend_days = int((days.weekday >= 5).sum())
    season_days = int(season_val * len(days))

    _, daily, _ = load_csvs()

    avg_price_all = 2500.0
    avg_days_all = 2.0
    avg_total_rentals_per_day = 4.0

    if daily is not None and not daily.empty:
        avg_price_all = float(pd.to_numeric(daily["AvgDailyPrice"], errors="coerce").dropna().mean() or avg_price_all)
        avg_days_all = float(pd.to_numeric(daily.get("AvgRentalDays", pd.Series([2])), errors="coerce").dropna().mean() or avg_days_all)
        avg_total_rentals_per_day = float(pd.to_numeric(daily["TotalRentals"], errors="coerce").dropna().mean() or avg_total_rentals_per_day)

    # Linear Regression (Revenue)
    lin = TRAINED_MODELS.get("linear_reg", {}).get("model")
    if lin is not None:
        est_total_rentals = int(round(avg_total_rentals_per_day * len(days)))
        est_high_demand_days = int(round(0.35 * len(days)))

        X_lin = pd.DataFrame([{
            "TotalRentals": est_total_rentals,
            "AvgDailyPrice": avg_price_all,
            "SeasonDays": season_days,
            "WeekendDays": weekend_days,
            "HighDemandDays": est_high_demand_days
        }])

        pred_revenue_val = float(lin.predict(X_lin)[0])
        pred_revenue_text = f"Rs {int(max(pred_revenue_val, 0)):,}"
    else:
        pred_revenue_text = "--"

    # Logistic (Demand)
    pack_log = TRAINED_MODELS.get("logistic_reg", {}).get("All", {})
    log_all = pack_log.get("model")
    log_scaler = pack_log.get("scaler")

    if log_all is not None and log_scaler is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_log = np.array([[season_val, is_weekend_avg, avg_price_all, avg_days_all]])
        X_log_s = log_scaler.transform(X_log)
        demand_pred = int(log_all.predict(X_log_s)[0])
        pred_demand = "High" if demand_pred == 1 else "Low"
    else:
        pred_demand = "--"

    # KNN (Monthly rentals)
    pack_knn = TRAINED_MODELS.get("knn_reg", {}).get("All", {})
    knn_all = pack_knn.get("model")
    knn_scaler = pack_knn.get("scaler")

    if knn_all is not None and knn_scaler is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_knn = np.array([[season_val, is_weekend_avg, avg_price_all]])
        X_knn_s = knn_scaler.transform(X_knn)
        per_day = float(knn_all.predict(X_knn_s)[0])
        pred_rental_count_val = int(max(round(per_day * len(days)), 0))
        pred_rental_count = str(pred_rental_count_val)
    else:
        pred_rental_count = "--"

    # ✅ Vehicle Type Recommendation (NO “always Tuk Tuk” issue)
    # Use KNN per-type predicted month rentals and pick best
    per_type_preds = {}
    is_weekend_avg = weekend_days / max(len(days), 1)

    for vt in ["Bike", "Car", "Tuk Tuk"]:
        pack = TRAINED_MODELS.get("knn_reg", {}).get(vt, {})
        m = pack.get("model")
        sc = pack.get("scaler")
        if m is None or sc is None:
            continue
        Xv = np.array([[season_val, is_weekend_avg, avg_price_all]])
        Xv_s = sc.transform(Xv)
        per_day_v = float(m.predict(Xv_s)[0])
        per_month_v = float(max(per_day_v * len(days), 0))
        per_type_preds[vt] = per_month_v

    if per_type_preds:
        best_vt = max(per_type_preds, key=per_type_preds.get)
        pred_vehicle_type = pretty_vehicle_type(best_vt)
    else:
        pred_vehicle_type = "--"

    # Decision Tree (Risk)
    dt = TRAINED_MODELS.get("decision_tree", {}).get("model")
    if dt is not None:
        is_weekend_flag = 1 if (weekend_days / max(len(days), 1)) > 0.25 else 0
        X_dt = np.array([[season_val, is_weekend_flag, avg_price_all]])
        risk_pred = int(dt.predict(X_dt)[0])
        pred_risk = "High Risk" if risk_pred == 1 else "Low Risk"
    else:
        pred_risk = "--"

    return {
        "month": month_str,
        "pred_revenue": pred_revenue_text,
        "pred_demand": pred_demand,
        "pred_rental_count": pred_rental_count,
        "pred_vehicle_type": pred_vehicle_type,
        "pred_risk": pred_risk
    }