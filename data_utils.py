import pandas as pd
import os
from datetime import datetime

# File paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "rentals_transactions_realistic.csv")
DAILY_CSV = os.path.join(DATA_DIR, "rentals_daily_realistic.csv")
MONTHLY_CSV = os.path.join(DATA_DIR, "rentals_monthly_realistic.csv")


def load_csvs():
    """
    Load three CSV files (transactions, daily, monthly).
    
    Returns:
        tuple: (transactions_df, daily_df, monthly_df)
    """
    try:
        transactions = pd.read_csv(TRANSACTIONS_CSV)
        daily = pd.read_csv(DAILY_CSV)
        monthly = pd.read_csv(MONTHLY_CSV)
        return transactions, daily, monthly
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return None, None, None


def is_season(date_obj):
    """
    Check if date falls in peak season (Oct-Mar).
    
    Args:
        date_obj: datetime object
        
    Returns:
        int: 1 if Oct-Mar, else 0
    """
    month = date_obj.month
    return 1 if month >= 10 or month <= 3 else 0


def is_weekend(date_obj):
    """
    Check if date is weekend (Saturday=5, Sunday=6).
    
    Args:
        date_obj: datetime object
        
    Returns:
        int: 1 if weekend, else 0
    """
    return 1 if date_obj.weekday() >= 5 else 0


def append_transaction(record_dict):
    """
    Append a new transaction record to the transactions CSV.
    
    Args:
        record_dict: dict with keys:
            - date (YYYY-MM-DD string)
            - vehicle_type (Bike, Car, Tuk Tuk)
            - customer_type (Local, Foreigner)
            - rental_days (int)
            - daily_price (float, in LKR)
            - notes (str, optional)
    
    Returns:
        dict: New record with computed fields
    """
    transactions, _, _ = load_csvs()
    
    if transactions is None or transactions.empty:
        next_rental_id = 1
    else:
        next_rental_id = transactions["RentalID"].max() + 1
    
    # Parse date
    date_obj = datetime.strptime(record_dict["date"], "%Y-%m-%d")
    
    # Compute derived fields
    new_record = {
        "RentalID": next_rental_id,
        "Date": record_dict["date"],
        "Year": date_obj.year,
        "Month": date_obj.month,
        "IsSeason": is_season(date_obj),
        "IsWeekend": is_weekend(date_obj),
        "VehicleType": record_dict["vehicle_type"],
        "RentalDays": record_dict["rental_days"],
        "DailyPriceLKR": record_dict["daily_price"],
        "TotalPriceLKR": record_dict["rental_days"] * record_dict["daily_price"],
        "CustomerType": record_dict["customer_type"],
        "Notes": record_dict.get("notes", "")
    }
    
    # Append to DataFrame
    new_row = pd.DataFrame([new_record])
    if transactions is None or transactions.empty:
        transactions = new_row
    else:
        transactions = pd.concat([transactions, new_row], ignore_index=True)
    
    # Save
    transactions.to_csv(TRANSACTIONS_CSV, index=False)
    
    return new_record


def rebuild_daily_monthly():
    """
    Rebuild daily and monthly datasets from the transactions CSV.
    Updates both daily_csv and monthly_csv.
    """
    transactions, _, _ = load_csvs()
    
    if transactions is None or transactions.empty:
        print("No transactions to rebuild from")
        return None, None
    
    # Ensure Date is datetime
    transactions["Date"] = pd.to_datetime(transactions["Date"])
    
    # ========== BUILD DAILY DATASET ==========
    daily = transactions.groupby("Date").agg(
        TotalRentals=("RentalID", "count"),
        RevenueLKR=("TotalPriceLKR", "sum"),
        AvgDailyPrice=("DailyPriceLKR", "mean"),
        AvgRentalDays=("RentalDays", "mean"),
        ForeignerShare=("CustomerType", lambda x: (x == "Foreigner").mean()),
        IsSeason=("IsSeason", "max"),
        IsWeekend=("IsWeekend", "max")
    ).reset_index()
    
    # Create demand labels
    daily["DemandHigh"] = (daily["TotalRentals"] >= 6).astype(int)
    daily["DemandLabel"] = daily["DemandHigh"].apply(lambda x: "High" if x == 1 else "Low")
    
    # Add Year, Month from Date
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    
    # Save daily
    daily.to_csv(DAILY_CSV, index=False)
    
    # ========== BUILD MONTHLY DATASET ==========
    # Create period for grouping
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
    
    # Format month as YYYY-MM string
    monthly["Month"] = monthly["YearMonth"].astype(str)
    monthly = monthly.drop("YearMonth", axis=1)
    
    # Save monthly
    monthly.to_csv(MONTHLY_CSV, index=False)
    
    return daily, monthly
