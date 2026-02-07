import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# ==========================================
# CONFIGURATION
# ==========================================
GRAPH_FILENAME = "stock_prediction.png"
DATA_FILENAME = "AAPL.csv"

# ==========================================
# 1. DATA LOADING / GENERATION
# ==========================================
def get_data(filename=DATA_FILENAME):
    """
    Tries to load 'AAPL.csv'. If not found, generates synthetic data.
    """
    if os.path.exists(filename):
        print(f"[✔] Loading real data from {filename}...")
        try:
            df = pd.read_csv(filename)
            # Handle Yahoo Finance format
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[['Close']] # We only need the Close price
        except Exception as e:
            print(f"[X] Error reading CSV: {e}")
            exit()
    else:
        print(f"[!] '{filename}' not found. Generating synthetic data...")
        np.random.seed(42)
        days = 500
        date_range = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Simulate a stock trend (Random Walk)
        prices = [150]
        for _ in range(days - 1):
            change = np.random.normal(0, 2) + 0.05
            prices.append(prices[-1] + change)
            
        df = pd.DataFrame({'Close': prices}, index=date_range)
    
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def prepare_data(df):
    """
    Prepares features (X) and target (y).
    """
    df = df.copy()
    
    # Target (y): The price of the NEXT day
    df['Prediction'] = df['Close'].shift(-1)
    
    # Drop the last row (NaN)
    df.dropna(inplace=True)
    
    X = np.array(df[['Close']])
    y = np.array(df['Prediction'])
    
    return X, y, df

# ==========================================
# 3. TRAINING
# ==========================================
def train_model(X, y):
    """
    Trains the Linear Regression model.
    """
    # NO SHUFFLING for Time Series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    
    return model, predictions, X_test, y_test, score

# ==========================================
# 4. VISUALIZATION & SAVING
# ==========================================
def save_and_plot_results(df, y_test, predictions):
    """
    Plots Actual vs Predicted prices and saves the image.
    """
    test_dates = df.index[-len(y_test):]
    
    plt.figure(figsize=(14, 7))
    plt.title('Apple (AAPL) Stock Price Prediction (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    # Actual Data
    plt.plot(test_dates, y_test, label='Actual Price', color='#1f77b4', linewidth=2)
    
    # Predicted Data
    plt.plot(test_dates, predictions, label='AI Prediction', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # SAVE THE PLOT
    save_path = os.path.join(os.getcwd(), GRAPH_FILENAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"[✔] Graph saved successfully at: {save_path}")
    
    # Show plot
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("   AAPL STOCK PREDICTION SYSTEM")
    print("==========================================\n")

    # 1. Load Data
    df = get_data()
    print(f"[✔] Data processed: {len(df)} days of history.")

    # 2. Prepare
    X, y, clean_df = prepare_data(df)

    # 3. Train
    model, preds, X_test, y_test, score = train_model(X, y)
    print(f"[✔] Model Trained. Accuracy (R²): {score:.4f}")

    # 4. Predict Tomorrow
    last_known_price = clean_df['Close'].iloc[-1]
    last_val_reshaped = np.array([[last_known_price]])
    next_price = model.predict(last_val_reshaped)
    
    print("\n------------------------------------------")
    print(f"   Today's Close:      ${last_known_price:.2f}")
    print(f"   PREDICTED Tomorrow: ${next_price[0]:.2f}")
    print("------------------------------------------\n")
    
    # 5. Visualize & Save
    print("[-] Generating graph...")
    save_and_plot_results(clean_df, y_test, preds)