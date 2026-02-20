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
from sklearn.pipeline import Pipeline

from data_utils import load_csvs, is_season, is_weekend

TRAINED_MODELS = {}

VEHICLE_TYPES = ["All", "Bike", "Car", "Tuk Tuk"]
REAL_TYPES = ["Bike", "Car", "Tuk Tuk"]


def ensure_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    return df


def _demand_threshold(series):
    """Dynamic threshold to avoid single-class (better than hard-coded 6)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1
    thr = float(s.quantile(0.70))
    return max(1, int(round(thr)))


def build_daily_from_transactions(transactions: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    """Build daily dataset from transactions filtered by vehicle_type."""
    if transactions is None or transactions.empty:
        return pd.DataFrame()

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])

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

    thr = _demand_threshold(daily["TotalRentals"])
    daily["DemandHigh"] = (daily["TotalRentals"] >= thr).astype(int)

    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    return daily


# ---------------------------------------------------
# 1) Linear Regression – Monthly Revenue (All only)
# ---------------------------------------------------
def train_linear_regression_revenue():
    _, _, monthly = load_csvs()

    if monthly is None or monthly.empty or len(monthly) < 8:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    X = monthly[["TotalRentals", "AvgDailyPrice", "SeasonDays", "WeekendDays", "HighDemandDays"]].fillna(0)
    y = monthly["RevenueLKR"].fillna(0)

    if len(X) < 5:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": "insufficient data"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {"model": model, "r2": round(r2, 4), "mse": round(mse, 2), "rmse": round(rmse, 2), "status": "trained"}


# ---------------------------------------------------
# 2) Logistic Regression – Demand (per vehicle type)
# ---------------------------------------------------
def train_logistic_regression_demand(vehicle_type="All"):
    transactions, daily_all, _ = load_csvs()

    if vehicle_type == "All":
        daily = daily_all.copy() if daily_all is not None else None
        if daily is not None and not daily.empty:
            thr = _demand_threshold(daily["TotalRentals"])
            daily["DemandHigh"] = (pd.to_numeric(daily["TotalRentals"], errors="coerce").fillna(0) >= thr).astype(int)
    else:
        daily = build_daily_from_transactions(transactions, vehicle_type)

    if daily is None or daily.empty or len(daily) < 12:
        return {"model": None, "accuracy": 0, "status": f"insufficient data ({vehicle_type})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays", "DemandHigh"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice", "AvgRentalDays"]].fillna(0)
    y = daily["DemandHigh"].fillna(0).astype(int)

    if len(pd.Series(y).unique()) < 2:
        return {"model": None, "accuracy": 0, "status": f"only one class ({vehicle_type})"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"))
    ])
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

    if daily is None or daily.empty or len(daily) < 15:
        return {"model": None, "r2": 0, "mse": 0, "rmse": 0, "status": f"insufficient data ({vehicle_type})"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])
    X = daily[["IsSeason", "IsWeekend", "AvgDailyPrice"]].fillna(0)
    y = pd.to_numeric(daily["TotalRentals"], errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=7))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {"model": model, "r2": round(r2, 4), "mse": round(mse, 2), "rmse": round(rmse, 2), "status": "trained"}


# ---------------------------------------------------
# 4) SVM – Vehicle Type (All only)
# ---------------------------------------------------
def train_svm_classifier_vehicle_type():
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty or len(transactions) < 60:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    transactions = ensure_columns(transactions, ["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR", "VehicleType"])

    le_vehicle = LabelEncoder()
    y = le_vehicle.fit_transform(transactions["VehicleType"].astype(str))
    X = transactions[["IsSeason", "IsWeekend", "RentalDays", "DailyPriceLKR"]].fillna(0)

    if len(np.unique(y)) < 2:
        return {"model": None, "accuracy": 0, "status": "insufficient vehicle classes"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=42))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": round(acc, 4), "le_vehicle": le_vehicle, "status": "trained"}


# ---------------------------------------------------
# 5) Decision Tree – Risk (All only)
# ---------------------------------------------------
def train_decision_tree_risk():
    _, daily, _ = load_csvs()
    if daily is None or daily.empty or len(daily) < 20:
        return {"model": None, "accuracy": 0, "status": "insufficient data"}

    daily = ensure_columns(daily, ["IsSeason", "IsWeekend", "AvgDailyPrice", "TotalRentals"])

    # dynamic risk threshold
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


def _avg_stats_for_type(transactions, daily_all, vehicle_type):
    if vehicle_type == "All":
        daily = daily_all
    else:
        daily = build_daily_from_transactions(transactions, vehicle_type)

    if daily is None or daily.empty:
        return 2500.0, 2.0  # fallback
    avg_price = float(pd.to_numeric(daily["AvgDailyPrice"], errors="coerce").dropna().mean() or 2500.0)
    avg_days = float(pd.to_numeric(daily.get("AvgRentalDays", pd.Series([2])), errors="coerce").dropna().mean() or 2.0)
    return avg_price, avg_days


def _recommended_price(avg_price, is_season_val, vehicle_type):
    type_multiplier = {"Bike": 0.9, "Car": 1.25, "Tuk Tuk": 1.0, "All": 1.0}
    season_factor = 1.2 if is_season_val == 1 else 0.9
    return int(round(avg_price * season_factor * type_multiplier.get(vehicle_type, 1.0)))


def predict_daily(date_str, vehicle_type="All"):
    """Predict for a given date. If vehicle_type=All -> returns per_type + top + revenue."""
    if not TRAINED_MODELS:
        train_all_models()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    is_season_val = is_season(date_obj)
    is_weekend_val = is_weekend(date_obj)

    transactions, daily_all, _ = load_csvs()

    vehicle_type = vehicle_type if vehicle_type in VEHICLE_TYPES else "All"

    # ✅ All: predict each type separately
    if vehicle_type == "All":
        per_type = {}
        total_count = 0
        total_revenue = 0

        for vt in REAL_TYPES:
            avg_price, avg_days = _avg_stats_for_type(transactions, daily_all, vt)

            log_model = TRAINED_MODELS["logistic_reg"].get(vt, {}).get("model")
            if log_model:
                X_log = np.array([[is_season_val, is_weekend_val, avg_price, avg_days]])
                demand_pred = int(log_model.predict(X_log)[0])
                demand = "High" if demand_pred == 1 else "Low"
            else:
                demand = "--"

            knn_model = TRAINED_MODELS["knn_reg"].get(vt, {}).get("model")
            if knn_model:
                X_knn = np.array([[is_season_val, is_weekend_val, avg_price]])
                count_pred = float(knn_model.predict(X_knn)[0])
                count = int(round(max(count_pred, 0)))
            else:
                count = 0

            price = _recommended_price(avg_price, is_season_val, vt)
            revenue = int(round(count * price))

            per_type[vt] = {
                "demand": demand,
                "predicted_count": count,
                "recommended_price": price,
                "predicted_revenue": revenue
            }

            total_count += count
            total_revenue += revenue

        top_vehicle = max(per_type.keys(), key=lambda k: per_type[k]["predicted_count"]) if per_type else "--"

        return {
            "date": date_str,
            "vehicle_type": "All",
            "predicted_count": total_count,
            "predicted_revenue": total_revenue,
            "top_vehicle": top_vehicle,
            "per_type": per_type
        }

    # ✅ Single type (Bike/Car/Tuk Tuk)
    avg_price, avg_days = _avg_stats_for_type(transactions, daily_all, vehicle_type)

    log_model = TRAINED_MODELS["logistic_reg"].get(vehicle_type, {}).get("model")
    if log_model:
        X_log = np.array([[is_season_val, is_weekend_val, avg_price, avg_days]])
        demand_pred = int(log_model.predict(X_log)[0])
        demand = "High" if demand_pred == 1 else "Low"
    else:
        demand = "--"

    knn_model = TRAINED_MODELS["knn_reg"].get(vehicle_type, {}).get("model")
    if knn_model:
        X_knn = np.array([[is_season_val, is_weekend_val, avg_price]])
        count_pred = float(knn_model.predict(X_knn)[0])
        count = int(round(max(count_pred, 0)))
    else:
        count = "--"

    price = _recommended_price(avg_price, is_season_val, vehicle_type)
    revenue = int(round((0 if count == "--" else count) * price))

    return {
        "date": date_str,
        "vehicle_type": vehicle_type,
        "tomorrow_demand": demand,          # keep frontend compatibility
        "predicted_count": count,
        "recommended_price": price,
        "predicted_revenue": revenue
    }


def predict_monthly(month_str: str):
    """Month: YYYY-MM -> returns totals + per_type + top_vehicle."""
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

    transactions, daily, _ = load_csvs()

    # fallback global averages
    avg_price_all = 2500.0
    avg_days_all = 2.0
    avg_total_rentals_per_day = 4.0

    if daily is not None and not daily.empty:
        avg_price_all = float(pd.to_numeric(daily["AvgDailyPrice"], errors="coerce").dropna().mean() or avg_price_all)
        avg_days_all = float(pd.to_numeric(daily.get("AvgRentalDays", pd.Series([2])), errors="coerce").dropna().mean() or avg_days_all)
        avg_total_rentals_per_day = float(pd.to_numeric(daily["TotalRentals"], errors="coerce").dropna().mean() or avg_total_rentals_per_day)

    # ---- Linear Regression monthly revenue
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

    # ---- Demand (All)
    log_all = TRAINED_MODELS.get("logistic_reg", {}).get("All", {}).get("model")
    if log_all is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_log = np.array([[season_val, is_weekend_avg, avg_price_all, avg_days_all]])
        demand_pred = int(log_all.predict(X_log)[0])
        pred_demand = "High" if demand_pred == 1 else "Low"
    else:
        pred_demand = "--"

    # ---- KNN rental count (All) -> month total
    knn_all = TRAINED_MODELS.get("knn_reg", {}).get("All", {}).get("model")
    if knn_all is not None:
        is_weekend_avg = weekend_days / max(len(days), 1)
        X_knn = np.array([[season_val, is_weekend_avg, avg_price_all]])
        per_day = float(knn_all.predict(X_knn)[0])
        pred_rental_count_val = int(max(round(per_day * len(days)), 0))
        pred_rental_count = str(pred_rental_count_val)
    else:
        pred_rental_count = "--"

    # ---- SVM vehicle type recommendation (rough)
    svm_pack = TRAINED_MODELS.get("svm_classifier", {})
    svm = svm_pack.get("model")
    le_vehicle = svm_pack.get("le_vehicle")

    if svm is not None and le_vehicle is not None:
        is_weekend_flag = 1 if (weekend_days / max(len(days), 1)) > 0.25 else 0
        X_svm = np.array([[season_val, is_weekend_flag, avg_days_all, avg_price_all]])
        pred_class = int(svm.predict(X_svm)[0])
        pred_vehicle_type = str(le_vehicle.inverse_transform([pred_class])[0])
    else:
        pred_vehicle_type = "--"

    # ---- Decision tree risk
    dt = TRAINED_MODELS.get("decision_tree", {}).get("model")
    if dt is not None:
        is_weekend_flag = 1 if (weekend_days / max(len(days), 1)) > 0.25 else 0
        X_dt = np.array([[season_val, is_weekend_flag, avg_price_all]])
        risk_pred = int(dt.predict(X_dt)[0])
        pred_risk = "High Risk" if risk_pred == 1 else "Low Risk"
    else:
        pred_risk = "--"

    # ✅ Per-type month estimates using per-type KNN models
    per_type = {}
    total_rev_est = 0
    top_vehicle = "--"
    top_count = -1

    for vt in REAL_TYPES:
        avg_price_vt, avg_days_vt = _avg_stats_for_type(transactions, daily, vt)
        knn_vt = TRAINED_MODELS.get("knn_reg", {}).get(vt, {}).get("model")

        if knn_vt is not None:
            is_weekend_avg = weekend_days / max(len(days), 1)
            X_knn = np.array([[season_val, is_weekend_avg, avg_price_vt]])
            per_day = float(knn_vt.predict(X_knn)[0])
            count_month = int(max(round(per_day * len(days)), 0))
        else:
            count_month = 0

        price = _recommended_price(avg_price_vt, season_val, vt)
        rev = int(round(count_month * price))
        per_type[vt] = {"predicted_count": count_month, "recommended_price": price, "predicted_revenue": rev}
        total_rev_est += rev

        if count_month > top_count:
            top_count = count_month
            top_vehicle = vt

    return {
        "month": month_str,
        "pred_revenue": pred_revenue_text,      # from LinearReg (All)
        "pred_demand": pred_demand,
        "pred_rental_count": pred_rental_count, # All count
        "pred_vehicle_type": pred_vehicle_type, # SVM
        "pred_risk": pred_risk,
        # ✅ new fields (frontend can show later)
        "per_type": per_type,
        "top_vehicle": top_vehicle,
        "revenue_estimate_from_types": f"Rs {int(max(total_rev_est, 0)):,}"
    }
