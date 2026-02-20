import pandas as pd
import os
from datetime import datetime

# ✅ File paths (keep CSVs in same folder as this file)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "rentals_transactions_realistic.csv")
DAILY_CSV = os.path.join(DATA_DIR, "rentals_daily_realistic.csv")
MONTHLY_CSV = os.path.join(DATA_DIR, "rentals_monthly_realistic.csv")


def load_csvs():
    """Load (transactions, daily, monthly) safely."""
    def _safe_read(path):
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    transactions = _safe_read(TRANSACTIONS_CSV)
    daily = _safe_read(DAILY_CSV)
    monthly = _safe_read(MONTHLY_CSV)
    return transactions, daily, monthly


def is_season(date_obj):
    """Peak season Oct-Mar"""
    month = date_obj.month
    return 1 if month >= 10 or month <= 3 else 0


def is_weekend(date_obj):
    """Saturday=5 Sunday=6"""
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

    new_record = {
        "RentalID": next_rental_id,
        "Date": record_dict["date"],
        "Year": date_obj.year,
        "Month": date_obj.month,
        "IsSeason": is_season(date_obj),
        "IsWeekend": is_weekend(date_obj),
        "VehicleType": record_dict["vehicle_type"],
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
    """Rebuild daily + monthly datasets from transactions CSV."""
    transactions, _, _ = load_csvs()

    if transactions is None or transactions.empty:
        print("No transactions to rebuild from")
        return None, None

    transactions["Date"] = pd.to_datetime(transactions["Date"], errors="coerce")
    transactions = transactions.dropna(subset=["Date"])

    # if CustomerType missing/NaN -> treat as Local
    if "CustomerType" not in transactions.columns:
        transactions["CustomerType"] = "Local"
    transactions["CustomerType"] = transactions["CustomerType"].fillna("Local")

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

    # ✅ dynamic threshold (helps avoid "only one class" errors)
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
