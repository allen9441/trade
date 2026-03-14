from data.loader import DataLoader
from models.lstm import LSTMTrader
from config import TICKERS, FEATURES, SEQUENCE_LENGTH, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, BASE_DIR
import pandas as pd
import numpy as np
import sys
import warnings
from datetime import datetime, timedelta
import json
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

def prepare_data(loader, tickers, start_date, end_date):
    """抓取並處理多資產數據"""
    try:
        market_df = loader.fetch_market_index(start_date, end_date)
    except Exception as e:
        print(f"Error fetching market index: {e}")
        sys.exit(1)

    all_data = {}
    combined_data = []

    for ticker in tickers:
        try:
            df = loader.fetch_data(ticker, start_date, end_date)
            if df.empty or len(df) < 100:
                continue
            df['Vol_Change'] = df['Volume'].pct_change()
            df = loader.add_technical_indicators(df)
            df = pd.merge(df, market_df, on='Date', how='inner')
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            all_data[ticker] = df.copy()
            df['Ticker'] = ticker
            combined_data.append(df)
        except Exception:
            continue

    if not combined_data:
        print("No valid data for any ticker.")
        sys.exit(1)

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.sort_values(by='Date', inplace=True)
    return all_data, combined_df

def main():
    print(f"--- AI Quant Trading Prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    loader = DataLoader()
    tickers = TICKERS
    features = FEATURES

    train_start = (datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    train_end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n[1] Fetching Data ({train_start} to {train_end})...")
    all_data_dict, df_train_combined = prepare_data(loader, tickers, train_start, train_end)

    print("\n[2] Checking for weekend retraining or loading existing model...")
    trader = LSTMTrader(sequence_length=SEQUENCE_LENGTH, features=features)

    # retrain on Mondays or missing model
    if datetime.now().weekday() == 0 or not trader.load_model():
        print("Training model on latest data from scratch...")
        trader.train(df_train_combined, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
        trader.save_model()
    else:
        print("Using cached model weights. Skipping full training.")

    print("\n[3] Generating Signals for Today...")
    latest_signals = []

    for ticker in tickers:
        if ticker not in all_data_dict:
            continue
        df_test = all_data_dict[ticker]
        df_signals = trader.generate_signals(df_test)

        if not df_signals.empty:
            last_row = df_signals.iloc[-1]
            # if signal exists (use pd.notna to avoid NaN comparison issues)
            if pd.notna(last_row['Position']) and last_row['Position'] != 0:
                confidence = last_row.get('lstm_prob_smooth', np.nan)
                if pd.isna(confidence):
                    confidence = last_row.get('lstm_prob', 0.5)
                if pd.isna(confidence):
                    confidence = 0.5
                latest_signals.append({
                    "Date": last_row['Date'].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Close": float(last_row['Close']),
                    "Signal": "BUY" if last_row['Position'] == 1.0 else "SELL",
                    "Confidence": float(confidence)
                })

    # 按照信心度排序
    latest_signals = sorted(latest_signals, key=lambda x: x['Confidence'], reverse=True)

    # 輸出成 JSON 供其他系統或機器人讀取
    output_path = os.path.join(BASE_DIR, "latest_signals.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(latest_signals, f, ensure_ascii=False, indent=4)

    print(f"\n[4] Job Complete. Found {len(latest_signals)} actionable signals today.")
    for sig in latest_signals:
        print(f"[{sig['Signal']}] {sig['Ticker']} - Confidence: {sig['Confidence']:.2f}")

if __name__ == "__main__":
    main()
