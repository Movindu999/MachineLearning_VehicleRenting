import pandas as pd
import os
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "rentals_transactions_realistic.csv")
DAILY_CSV = os.path.join(DATA_DIR, "rentals_daily_realistic.csv")
MONTHLY_CSV = os.path.join(DATA_DIR, "rentals_monthly_realistic.csv")


def normalize_vehicle_type(v: str) -> str:
    """Normalize vehicle type names to one consistent format."""
    if v is None:
        return "All"
    s = str(v).strip()

    # common variants -> one format
    s_low = s.lower().replace("-", "").replace(" ", "")
    if s_low in ["tuktuk", "tuk2", "tuk"]:
        return "Tuk Tuk"
    if s_low in ["bike", "motorbike", "motorcycle"]:
        return "Bike"
    if s_low in ["car", "cars"]:
        return "Car"
    if s_low == "all":
        return "All"

    # fallback original (title-case maybe)
    return s.title()


def load_csvs():
    """Load (transactions, daily, monthly) safely + normalize VehicleType columns."""
    def _safe_read(path):
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    transactions = _safe_read(TRANSACTIONS_CSV)
    daily = _safe_read(DAILY_CSV)
    monthly = _safe_read(MONTHLY_CSV)

    # ✅ Normalize transactions VehicleType
    if transactions is not None and not transactions.empty and "VehicleType" in transactions.columns:
        transactions["VehicleType"] = transactions["VehicleType"].apply(normalize_vehicle_type)

    # ✅ If daily has per-day vehicle types (usually it doesn't), keep safe
    if daily is not None and not daily.empty and "VehicleType" in daily.columns:
        daily["VehicleType"] = daily["VehicleType"].apply(normalize_vehicle_type)

    return transactions, daily, monthly


def is_season(date_obj):
    month = date_obj.month
    return 1 if month >= 10 or month <= 3 else 0


def is_weekend(date_obj):
    return 1 if date_obj.weekday() >= 5 else 0


def append_transaction(record_dict):
    """
    Append new transaction to CSV.
    record_dict keys:
      date, vehicle_type, customer_type, rental_days, daily_price, notes(optional)
    """
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty:
        next_rental_id = 1
        transactions = pd.DataFrame(columns=[
            "RentalID", "Date", "Year", "Month", "IsSeason", "IsWeekend", "VehicleType",
            "RentalDays", "DailyPriceLKR", "TotalPriceLKR", "CustomerType", "Notes"
        ])
    else:
        next_rental_id = int(pd.to_numeric(transactions["RentalID"], errors="coerce").max()) + 1

    date_obj = datetime.strptime(record_dict["date"], "%Y-%m-%d")
    vtype = normalize_vehicle_type(record_dict.get("vehicle_type", "All"))

    new_record = {
        "RentalID": next_rental_id,
        "Date": record_dict["date"],
        "Year": date_obj.year,
        "Month": date_obj.month,
        "IsSeason": is_season(date_obj),
        "IsWeekend": is_weekend(date_obj),
        "VehicleType": vtype,
        "RentalDays": int(record_dict["rental_days"]),
        "DailyPriceLKR": float(record_dict["daily_price"]),
        "TotalPriceLKR": int(record_dict["rental_days"]) * float(record_dict["daily_price"]),
        "CustomerType": record_dict.get("customer_type", "Local"),
        "Notes": record_dict.get("notes", "")
    }

    transactions = pd.concat([transactions, pd.DataFrame([new_record])], ignore_index=True)
    transactions.to_csv(TRANSACTIONS_CSV, index=False)
    return new_record


def rebuild_daily_monthly():
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty:
        print("No transactions to rebuild from")
        return None, None

    transactions["Date"] = pd.to_datetime(transactions["Date"], errors="coerce")
    transactions = transactions.dropna(subset=["Date"])

    # ensure columns
    if "CustomerType" not in transactions.columns:
        transactions["CustomerType"] = "Local"
    transactions["CustomerType"] = transactions["CustomerType"].fillna("Local")

    if "VehicleType" in transactions.columns:
        transactions["VehicleType"] = transactions["VehicleType"].apply(normalize_vehicle_type)

    # DAILY
    daily = transactions.groupby("Date").agg(
        TotalRentals=("RentalID", "count"),
        RevenueLKR=("TotalPriceLKR", "sum"),
        AvgDailyPrice=("DailyPriceLKR", "mean"),
        AvgRentalDays=("RentalDays", "mean"),
        ForeignerShare=("CustomerType", lambda x: (x == "Foreigner").mean()),
        IsSeason=("IsSeason", "max"),
        IsWeekend=("IsWeekend", "max")
    ).reset_index()

    # dynamic threshold
    thr = float(pd.to_numeric(daily["TotalRentals"], errors="coerce").dropna().quantile(0.70) or 1)
    thr = max(1, int(round(thr)))
    daily["DemandHigh"] = (daily["TotalRentals"] >= thr).astype(int)
    daily["DemandLabel"] = daily["DemandHigh"].apply(lambda x: "High" if x == 1 else "Low")

    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    daily.to_csv(DAILY_CSV, index=False)

    # MONTHLY
    daily_for_monthly = daily.copy()
    daily_for_monthly["YearMonth"] = daily_for_monthly["Date"].dt.to_period("M")

    monthly = daily_for_monthly.groupby("YearMonth").agg(
        TotalRentals=("TotalRentals", "sum"),
        RevenueLKR=("RevenueLKR", "sum"),
        AvgDailyPrice=("AvgDailyPrice", "mean"),
        ForeignerShare=("ForeignerShare", "mean"),
        SeasonDays=("IsSeason", "sum"),
        WeekendDays=("IsWeekend", "sum"),
        HighDemandDays=("DemandHigh", "sum")
    ).reset_index()

    monthly["Month"] = monthly["YearMonth"].astype(str)
    monthly = monthly.drop("YearMonth", axis=1)
    monthly.to_csv(MONTHLY_CSV, index=False)

    return daily, monthly