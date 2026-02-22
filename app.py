from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import traceback
import numpy as np
import pandas as pd

from data_utils import append_transaction, rebuild_daily_monthly, load_csvs
from ml_train_predict import train_all_models, predict_daily, predict_monthly

app = Flask(__name__)
CORS(app)


def to_native(x):
    """Convert numpy/pandas types to JSON-safe python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_native(v) for v in x]
    if isinstance(x, tuple):
        return [to_native(v) for v in x]
    return x


print("Training ML models on startup...")
models = train_all_models()
print("Models trained successfully!")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ✅ NEW: KPI Summary (Today + This month + per vehicle counts)
@app.route("/stats/summary", methods=["GET"])
def stats_summary():
    try:
        transactions, _, _ = load_csvs()
        if transactions is None or transactions.empty:
            return jsonify({
                "today": {"total": 0, "by_type": {"Bike": 0, "Car": 0, "Tuk Tuk": 0}},
                "month": {"total": 0, "by_type": {"Bike": 0, "Car": 0, "Tuk Tuk": 0}},
            }), 200

        # ensure Date is datetime
        transactions = transactions.copy()
        transactions["Date"] = pd.to_datetime(transactions["Date"], errors="coerce")
        transactions = transactions.dropna(subset=["Date"])

        today = pd.Timestamp.now().normalize()
        year = today.year
        month = today.month

        # TODAY
        tdf = transactions[transactions["Date"].dt.normalize() == today]
        today_total = int(len(tdf))
        today_by_type = tdf["VehicleType"].value_counts().to_dict()

        # MONTH
        mdf = transactions[(transactions["Date"].dt.year == year) & (transactions["Date"].dt.month == month)]
        month_total = int(len(mdf))
        month_by_type = mdf["VehicleType"].value_counts().to_dict()

        # ensure keys exist
        def _fill(d):
            return {
                "Bike": int(d.get("Bike", 0)),
                "Car": int(d.get("Car", 0)),
                "Tuk Tuk": int(d.get("Tuk Tuk", 0)),
            }

        return jsonify({
            "today": {"total": today_total, "by_type": _fill(today_by_type)},
            "month": {"total": month_total, "by_type": _fill(month_by_type)},
        }), 200

    except Exception as e:
        print("Error in /stats/summary:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ✅ Return Monthly dataset for chart
@app.route("/data/monthly", methods=["GET"])
def get_monthly_data():
    try:
        _, _, monthly = load_csvs()
        if monthly is None or monthly.empty:
            return jsonify([]), 200

        # Ensure Month + TotalRentals exist
        if "Month" not in monthly.columns or "TotalRentals" not in monthly.columns:
            return jsonify([]), 200

        out = []
        for _, row in monthly.iterrows():
            month_str = str(row["Month"])  # "YYYY-MM"
            try:
                dt = datetime.strptime(month_str, "%Y-%m")
                is_season = 1 if (dt.month >= 10 or dt.month <= 3) else 0
            except:
                is_season = 0

            total_rentals = pd.to_numeric(row["TotalRentals"], errors="coerce")
            total_rentals = int(total_rentals) if pd.notna(total_rentals) else 0

            out.append({
                "month": month_str,
                "total_rentals": total_rentals,
                "is_season": bool(is_season)
            })

        out.sort(key=lambda x: x["month"])
        return jsonify(out), 200

    except Exception as e:
        print("Error in /data/monthly:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/add_rental", methods=["POST"])
def add_rental():
    try:
        data = request.get_json(force=True)

        required_fields = ["date", "vehicle_type", "customer_type", "rental_days", "daily_price"]
        for f in required_fields:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        try:
            datetime.strptime(data["date"], "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        new_record = append_transaction({
            "date": data["date"],
            "vehicle_type": data["vehicle_type"],
            "customer_type": data["customer_type"],
            "rental_days": int(data["rental_days"]),
            "daily_price": float(data["daily_price"]),
            "notes": data.get("notes", "")
        })

        # rebuild aggregates
        rebuild_daily_monthly()

        # retrain models
        global models
        models = train_all_models()

        return jsonify({
            "message": "saved",
            "rental_id": int(new_record.get("RentalID")) if new_record.get("RentalID") is not None else None
        }), 201

    except Exception as e:
        print("Error in /add_rental:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict/daily", methods=["POST"])
def predict_daily_route():
    try:
        data = request.get_json(force=True)
        if "date" not in data:
            return jsonify({"error": "Missing field: date"}), 400

        try:
            datetime.strptime(data["date"], "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        vehicle_type = data.get("vehicle_type", "All")
        preds = predict_daily(data["date"], vehicle_type)

        # optional confidence fallback
        if isinstance(preds, dict) and "confidence" not in preds:
            preds["confidence"] = 85

        return jsonify(to_native(preds)), 200

    except Exception as e:
        print("Error in /predict/daily:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict/monthly", methods=["POST"])
def predict_monthly_route():
    try:
        data = request.get_json(force=True)
        if "month" not in data:
            return jsonify({"error": "Missing field: month"}), 400

        try:
            datetime.strptime(data["month"], "%Y-%m")
        except ValueError:
            return jsonify({"error": "Invalid month format. Use YYYY-MM"}), 400

        preds = predict_monthly(data["month"])
        return jsonify(to_native(preds)), 200

    except Exception as e:
        print("Error in /predict/monthly:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/models/info", methods=["GET"])
def models_info():
    """
    Returns model metrics for dashboard.

    - linear_reg, svm_classifier, decision_tree => normal dict
    - logistic_reg, knn_reg => dict by vehicle type
      For metrics table, show ONLY "All".

    IMPORTANT:
    Remove non-JSON objects like `model`, `scaler`, `le_vehicle`.
    """
    try:
        info = {}

        def clean(d):
            if not isinstance(d, dict):
                return {}
            drop = {"model", "scaler", "le_vehicle"}
            return {k: v for k, v in d.items() if k not in drop}

        for model_name, model_data in models.items():

            if not model_data:
                info[model_name] = {"status": "not available", "metrics": {}}
                continue

            # dict by vehicle type
            if model_name in ["logistic_reg", "knn_reg"] and isinstance(model_data, dict):
                chosen = model_data.get("All") or next(iter(model_data.values()))
                cm = clean(chosen)
                info[model_name] = {
                    "status": chosen.get("status", "unknown") if isinstance(chosen, dict) else "unknown",
                    "metrics": to_native(cm)
                }
                continue

            cm = clean(model_data)
            info[model_name] = {
                "status": model_data.get("status", "unknown") if isinstance(model_data, dict) else "unknown",
                "metrics": to_native(cm)
            }

        return jsonify(info), 200

    except Exception as e:
        print("Error in /models/info:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Vehicle Rental ML Dashboard Backend...")
    print("Flask app running on http://localhost:5000")
    app.run(debug=True, port=5000, host="0.0.0.0")