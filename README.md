# Vehicle Rental ML Dashboard - Python Backend

A complete ML backend for the Vehicle Rental Prediction & Management System using Flask, Pandas, and scikit-learn.

---

## Project Structure

```
c:\Users\user\Desktop\Test2\
├── app.py                               # Flask application
├── ml_train_predict.py                  # ML models training & predictions
├── data_utils.py                        # Data utilities & CSV operations
├── requirements.txt                     # Python dependencies
├── rentals_transactions_realistic.csv   # Transaction-level data
├── rentals_daily_realistic.csv          # Daily aggregated data
└── rentals_monthly_realistic.csv        # Monthly aggregated data
```

---

## Setup Instructions

### 1. Install Python (if not already installed)
Ensure Python 3.7+ is installed on your system.

### 2. Create Virtual Environment (Recommended)
```bash
cd c:\Users\user\Desktop\Test2
python -m venv venv
```

**Activate virtual environment:**
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (CMD):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Backend
```bash
python app.py
```

**Expected Output:**
```
Training ML models on startup...
Models trained successfully!
Starting Vehicle Rental ML Dashboard Backend...
Flask app running on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

---

## API Endpoints

### 1. Health Check
**GET** `/health`

**Response:**
```json
{
  "status": "ok"
}
```

---

### 2. Add New Rental
**POST** `/add_rental`

**Request Body:**
```json
{
  "date": "2026-02-18",
  "vehicle_type": "Car",
  "customer_type": "Foreigner",
  "rental_days": 3,
  "daily_price": 3000,
  "notes": "Optional notes about the rental"
}
```

**Response:**
```json
{
  "message": "saved",
  "rental_id": 124,
  "latest_daily": {
    "Date": "2026-02-18",
    "TotalRentals": 8,
    "RevenueLKR": 45000,
    ...
  },
  "latest_monthly": {
    "Month": "2026-02",
    "TotalRentals": 198,
    ...
  }
}
```

---

### 3. Predict Daily Metrics
**POST** `/predict/daily`

**Request Body:**
```json
{
  "date": "2026-02-19",
  "vehicle_type": "All"
}
```

**Response:**
```json
{
  "tomorrow_demand": "High",
  "predicted_count": 7,
  "recommended_price": 3000
}
```

---

### 4. Predict Monthly Metrics
**POST** `/predict/monthly`

**Request Body:**
```json
{
  "month": "2026-03"
}
```

**Response:**
```json
{
  "pred_revenue": 245000,
  "pred_demand": "High",
  "pred_rental_count": 892,
  "pred_vehicle_type": "Car",
  "pred_risk": "No"
}
```

---

### 5. Get Model Information
**GET** `/models/info`

**Response:**
```json
{
  "linear_reg": {
    "status": "trained",
    "metrics": {
      "r2": 0.8234,
      "mse": 1234.56
    }
  },
  "logistic_reg": {
    "status": "trained",
    "metrics": {
      "accuracy": 0.82
    }
  },
  ...
}
```

---

## ML Models

### 1. Linear Regression (Revenue Prediction)
- **Task:** Predict monthly revenue based on rental counts and pricing
- **Features:** TotalRentals, AvgDailyPrice, SeasonDays, WeekendDays, HighDemandDays
- **Output:** Revenue (LKR)
- **Metric:** R² Score

### 2. Logistic Regression (Demand Classification)
- **Task:** Classify daily demand as High or Low
- **Features:** IsSeason, IsWeekend, AvgDailyPrice, AvgRentalDays, ForeignerShare
- **Output:** High (if ≥6 rentals) or Low (<6 rentals)
- **Metric:** Accuracy

### 3. KNN Regressor (Rental Count)
- **Task:** Predict daily rental count
- **Features:** IsSeason, IsWeekend, AvgDailyPrice, ForeignerShare
- **Output:** Number of rentals (k=5 neighbors)
- **Metric:** R² Score, MSE

### 4. SVM Classifier (Vehicle Type Recommendation)
- **Task:** Recommend best-performing vehicle type for rental conditions
- **Features:** IsSeason, IsWeekend, RentalDays, DailyPriceLKR, CustomerType (one-hot encoded)
- **Output:** Bike, Car, or Tuk Tuk
- **Metric:** Accuracy

### 5. Decision Tree Classifier (Risk Assessment)
- **Task:** Assess vehicle shortage risk
- **Rules:**
  - Risk = Yes if (Peak Season AND Rentals ≥ 6) OR (Off-season AND Rentals ≥ 4)
  - Risk = No otherwise
- **Features:** IsSeason, IsWeekend, AvgDailyPrice, ForeignerShare
- **Output:** Yes or No
- **Metric:** Accuracy (max_depth=5)

---

## Data Files

### rentals_transactions_realistic.csv
Individual rental transaction records.

**Columns:**
```
RentalID, Date, Year, Month, IsSeason, IsWeekend, VehicleType, 
RentalDays, DailyPriceLKR, TotalPriceLKR, CustomerType, Notes
```

### rentals_daily_realistic.csv
Aggregated daily statistics.

**Columns:**
```
Date, TotalRentals, RevenueLKR, AvgDailyPrice, AvgRentalDays, 
ForeignerShare, IsSeason, IsWeekend, DemandHigh, DemandLabel, Year, Month
```

### rentals_monthly_realistic.csv
Aggregated monthly statistics.

**Columns:**
```
Month, TotalRentals, RevenueLKR, AvgDailyPrice, ForeignerShare, 
SeasonDays, WeekendDays, HighDemandDays
```

---

## Frontend Integration

**Connect the HTML dashboard to this backend:**

1. **Add new rental:**
   ```javascript
   fetch('http://localhost:5000/add_rental', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           date: document.getElementById('new_date').value,
           vehicle_type: document.getElementById('new_vehicle').value,
           customer_type: document.getElementById('new_customer_type').value,
           rental_days: document.getElementById('new_days').value,
           daily_price: document.getElementById('new_price').value,
           notes: document.getElementById('new_notes').value
       })
   }).then(res => res.json()).then(data => {
       document.getElementById('save_status').innerHTML = data.message;
   });
   ```

2. **Get daily predictions:**
   ```javascript
   fetch('http://localhost:5000/predict/daily', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           date: document.getElementById('filter_date').value,
           vehicle_type: document.getElementById('filter_vehicle_type').value
       })
   }).then(res => res.json()).then(data => {
       document.getElementById('daily_pred_demand').textContent = data.tomorrow_demand;
       document.getElementById('daily_pred_count').textContent = data.predicted_count;
       document.getElementById('daily_pred_price').textContent = data.recommended_price;
   });
   ```

3. **Get monthly predictions:**
   ```javascript
   fetch('http://localhost:5000/predict/monthly', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           month: document.getElementById('filter_month').value
       })
   }).then(res => res.json()).then(data => {
       document.getElementById('pred_revenue').textContent = data.pred_revenue;
       document.getElementById('pred_demand').textContent = data.pred_demand;
       document.getElementById('pred_rental_count').textContent = data.pred_rental_count;
       document.getElementById('pred_vehicle_type').textContent = data.pred_vehicle_type;
       document.getElementById('pred_shortage').textContent = data.pred_risk;
   });
   ```

---

## Testing the Backend

### Using PowerShell/cURL:

**1. Health Check:**
```powershell
curl -X GET http://localhost:5000/health
```

**2. Add Rental:**
```powershell
curl -X POST http://localhost:5000/add_rental `
  -H "Content-Type: application/json" `
  -d '{
    "date": "2026-02-18",
    "vehicle_type": "Car",
    "customer_type": "Foreigner",
    "rental_days": 3,
    "daily_price": 3000,
    "notes": "Business trip"
  }'
```

**3. Daily Prediction:**
```powershell
curl -X POST http://localhost:5000/predict/daily `
  -H "Content-Type: application/json" `
  -d '{"date": "2026-02-19", "vehicle_type": "All"}'
```

**4. Monthly Prediction:**
```powershell
curl -X POST http://localhost:5000/predict/monthly `
  -H "Content-Type: application/json" `
  -d '{"month": "2026-03"}'
```

---

## Troubleshooting

### Port 5000 Already in Use
```powershell
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Module Not Found Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### CSV File Not Found
Ensure the three CSV files are in the same directory as `app.py`.

### Models Not Training
Check that CSV files have sufficient data (minimum 5-10 rows). Empty files will cause issues.

---

## Notes for Viva/Presentation

**Key Points:**
1. **Data Pipeline:** Transactions → Daily → Monthly aggregation
2. **Derived Fields:** Season (Oct-Mar), Weekend detection, Demand labeling based on rentals ≥6
3. **Feature Engineering:** One-hot encoding for categorical variables, normalization via sklearn
4. **Model Selection:** Each model solves a different prediction task
5. **API Design:** RESTful endpoints for seamless frontend integration
6. **Extensibility:** Easy to add new models or modify features

**Common Q&A:**
- "Why 5 models?" → Different tasks require different algorithms (regression vs classification)
- "How is demand defined?" → Threshold-based: ≥6 rentals = High demand
- "Why rebuild daily/monthly?" → To keep aggregated datasets in sync with transactions
- "What happens if data is insufficient?" → Models return "--" (not available) and can be retrained later

---

## License & Credits

Created for Vehicle Rental ML Dashboard Project - February 2026.

---

**Happy Coding! 🚗📊**
