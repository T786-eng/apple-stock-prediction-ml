# 📈 Apple Stock Price Prediction

## 📌 Overview
This project predicts the future stock price of Apple (AAPL) using **Machine Learning (Linear Regression)**. It analyzes historical price trends to forecast the next day's closing price.

## ✨ Key Features
* **Hybrid Data Loading:** Automatically detects if `AAPL.csv` exists. If not, it generates realistic synthetic data for testing.
* **Auto-Save Graph:** Automatically generates and saves a high-resolution comparison graph (`stock_prediction.png`).
* **Price Forecasting:** Predicts the exact closing price for the next trading day.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`
* **Editor:** Visual Studio Code

## ⚙️ How to Run
1.  **Install Requirements:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
2.  **(Optional) Real Data:** Download the AAPL historical CSV from Yahoo Finance and rename it to `AAPL.csv`.
3.  **Run Script:**
    ```bash
    python AAPL_stock.py
    ```

## 📊 Output
* **Console:** Displays the model accuracy (R² Score) and the predicted price for tomorrow.
* **Image:** Saves a file named **`stock_prediction.png`** showing the AI's performance.

## 📝 Sample Output
```text
[✔] Model Trained. Accuracy (R²): 0.9850
------------------------------------------
   Today's Close:      $264.50
   PREDICTED Tomorrow: $265.12
------------------------------------------
[✔] Graph saved successfully at: .../stock_prediction.png