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


# ✅ NEW: Return Monthly dataset for chart
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

            out.append({
                "month": month_str,
                "total_rentals": int(pd.to_numeric(row["TotalRentals"], errors="coerce") or 0),
                "is_season": bool(is_season)
            })

        # sort by month
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

        rebuild_daily_monthly()

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

        if "confidence" not in preds:
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
    """
    try:
        info = {}

        for model_name, model_data in models.items():

            if not model_data:
                info[model_name] = {"status": "not available", "metrics": {}}
                continue

            if model_name in ["logistic_reg", "knn_reg"] and isinstance(model_data, dict):
                chosen = model_data.get("All") or next(iter(model_data.values()))
                clean_metrics = {k: v for k, v in chosen.items() if k not in ["model", "le_vehicle"]}
                info[model_name] = {
                    "status": chosen.get("status", "unknown"),
                    "metrics": to_native(clean_metrics)
                }
                continue

            clean_metrics = {k: v for k, v in model_data.items() if k not in ["model", "le_vehicle"]}
            info[model_name] = {
                "status": model_data.get("status", "unknown"),
                "metrics": to_native(clean_metrics)
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