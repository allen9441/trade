from data.loader import DataLoader
from models.lstm import LSTMTrader
import pandas as pd
import numpy as np
import sys
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

TICKERS = [
    "0050.TW", "0056.TW", "00878.TW", "00929.TW",
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW",
    "2882.TW", "2891.TW", "2886.TW", "1301.TW", "1303.TW", "2002.TW",
    "2884.TW", "2892.TW", "1216.TW", "2303.TW", "3711.TW", "2885.TW",
    "2880.TW", "3231.TW", "3045.TW", "2883.TW", "5871.TW", "2887.TW",
    "2395.TW", "2412.TW", "2890.TW", "5880.TW", "1101.TW", "2357.TW",
    "2301.TW", "2912.TW", "1326.TW", "2603.TW", "2207.TW", "1304.TW",
    "2324.TW", "6669.TW", "3034.TW", "4938.TW", "3037.TW", "2345.TW",
    "2356.TW", "1590.TW", "5876.TW", "4904.TW", "2379.TW"
]

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

    features = [
        'Open', 'Close', 'High', 'Low',
        'Body_Size', 'Daily_Range',
        'SMA_5', 'SMA_20', 'SMA_60', 'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'OBV',
        'Vol_Change', 'TWII_Return', 'TWII_Volume', 'SP500_Return'
    ]

    train_start = (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    train_end = datetime.now().strftime('%Y-%m-%d')

    print(f"\n[1] Fetching Data ({train_start} to {train_end})...")
    all_data_dict, df_train_combined = prepare_data(loader, tickers, train_start, train_end)

    print("\n[2] Checking for weekend retraining or loading existing model...")
    trader = LSTMTrader(sequence_length=20, features=features)

    # retrain on Mondays or missing model
    if datetime.now().weekday() == 0 or not trader.load_model():
        print("Training model on latest data from scratch...")
        trader.train(df_train_combined, epochs=20, batch_size=256)
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
            # if signal exists
            if last_row['Position'] != 0:
                latest_signals.append({
                    "Date": last_row['Date'].strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Close": float(last_row['Close']),
                    "Signal": "BUY" if last_row['Position'] == 1.0 else "SELL",
                    "Confidence": float(last_row['lstm_prob_smooth'])
                })

    # 按照信心度排序
    latest_signals = sorted(latest_signals, key=lambda x: x['Confidence'], reverse=True)

    # 輸出成 JSON 供其他系統或機器人讀取
    output_path = "latest_signals.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(latest_signals, f, ensure_ascii=False, indent=4)

    print(f"\n[4] Job Complete. Found {len(latest_signals)} actionable signals today.")
    for sig in latest_signals:
        print(f"[{sig['Signal']}] {sig['Ticker']} - Confidence: {sig['Confidence']:.2f}")

if __name__ == "__main__":
    main()
