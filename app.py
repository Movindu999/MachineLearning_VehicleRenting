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

            # dict-by-type models
            if model_name in ["logistic_reg", "knn_reg"] and isinstance(model_data, dict):
                chosen = model_data.get("All") or next(iter(model_data.values()))
                clean_metrics = {k: v for k, v in chosen.items() if k not in ["model", "le_vehicle"]}
                info[model_name] = {
                    "status": chosen.get("status", "unknown"),
                    "metrics": to_native(clean_metrics)
                }
                continue

            # normal models
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


# ✅ NEW: Scatter data endpoint
@app.route("/chart/monthly_scatter", methods=["GET"])
def monthly_scatter_chart():
    """
    Month vs TotalRentals (scatter) + season grouping.
    Response:
    {
      "labels": ["2024-01","2024-02",...],
      "season": [{"x":0,"y":120,"month":"2024-01"}, ...],
      "non_season": [{"x":3,"y":80,"month":"2024-04"}, ...]
    }
    """
    try:
        _, _, monthly = load_csvs()
        if monthly is None or monthly.empty:
            return jsonify({"labels": [], "season": [], "non_season": []}), 200

        m = monthly.copy()
        m["Month"] = m["Month"].astype(str)

        # Sort by Month safely
        m["MonthDT"] = pd.to_datetime(m["Month"], format="%Y-%m", errors="coerce")
        m = m.dropna(subset=["MonthDT"]).sort_values("MonthDT").reset_index(drop=True)

        labels = m["Month"].tolist()
        season_points = []
        non_season_points = []

        for i, row in m.iterrows():
            month_str = row["Month"]
            total_rentals = float(row.get("TotalRentals", 0) or 0)

            mm = int(month_str.split("-")[1])
            is_season = 1 if (mm >= 10 or mm <= 3) else 0

            pt = {"x": i, "y": total_rentals, "month": month_str}

            if is_season == 1:
                season_points.append(pt)
            else:
                non_season_points.append(pt)

        return jsonify({
            "labels": labels,
            "season": season_points,
            "non_season": non_season_points
        }), 200

    except Exception as e:
        print("Error in /chart/monthly_scatter:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Vehicle Rental ML Dashboard Backend...")
    print("Flask app running on http://localhost:5000")
    app.run(debug=True, port=5000, host="0.0.0.0")
