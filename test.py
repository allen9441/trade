import sys
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from data.loader import DataLoader
from models.lstm import LSTMTrader
from engine.backtest import BacktestEngine

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

def format_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD.")
        sys.exit(1)

def main():
    print("=============================")
    print("  AI Quant Trading Backtest  ")
    print("=============================\n")
    
    train_start_input = input("Enter Training Start Date [YYYYMMDD]: ")
    train_end_input = input("Enter Training End Date [YYYYMMDD]: ")
    test_buffer_start_input = input("Enter Testing Buffer Start Date (to gather sequence data before test) [YYYYMMDD]: ")
    test_start_input = input("Enter Testing Start Date [YYYYMMDD]: ")
    test_end_input = input("Enter Testing End Date [YYYYMMDD]: ")
    enable_short_input = input("Enable Short Selling? (T/F) [Default: F]: ").strip().upper()
    enable_short = True if enable_short_input == 'T' else False
    
    train_start = format_date(train_start_input)
    train_end = format_date(train_end_input)
    test_buffer_start = format_date(test_buffer_start_input)
    test_start = format_date(test_start_input)
    test_end = format_date(test_end_input)
    
    print(f"\n[Settings]")
    print(f"Training Period: {train_start} to {train_end}")
    print(f"Testing Period:  {test_start} to {test_end} (Buffer from: {test_buffer_start})")
    print(f"Initial Capital: $1,000,000 TWD")
    print(f"Short Selling:   {'Enabled' if enable_short else 'Disabled'}")
    print("==================================================\n")
    
    loader = DataLoader()
    features = [
        'Open', 'Close', 'High', 'Low',
        'Body_Size', 'Daily_Range',
        'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'OBV',
        'Vol_Change', 'TWII_Return', 'TWII_Volume', 'SP500_Return'
    ]
    
    # 1. 準備訓練集
    print("[1] Fetching Training Data...")
    try:
        market_df_train = loader.fetch_market_index(train_start, train_end)
    except Exception as e:
        print(f"Failed to fetch market data: {e}")
        return

    combined_train_data = []
    for ticker in TICKERS:
        df = loader.fetch_data(ticker, train_start, train_end)
        if df.empty or len(df) < 100: continue
        df['Vol_Change'] = df['Volume'].pct_change()
        df = loader.add_technical_indicators(df)
        df = pd.merge(df, market_df_train, on='Date', how='inner')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df['Ticker'] = ticker
        combined_train_data.append(df)
        
    if not combined_train_data:
        print("No valid training data found.")
        return
        
    df_train = pd.concat(combined_train_data, ignore_index=True)
    df_train.sort_values(by='Date', inplace=True)
    
    # 2. 訓練模型
    print(f"\n[2] Training 10-Model Ensemble (This will take a while)...")
    trader = LSTMTrader(sequence_length=20, features=features)
    trader.train(df_train, epochs=20, batch_size=256)
    
    # 3. 準備測試集並產生信號
    print("\n[3] Fetching Testing Data & Voting...")
    # 為確保測試期間的技術指標計算正確，需從測試期間的前幾個月開始擷取數據，以便生成足夠的序列數據
    market_df_test = loader.fetch_market_index(test_buffer_start, test_end)
    test_signals = {}
    raw_test_data = {}
    
    for ticker in TICKERS:
        # 同樣需要擷取測試期間前的數據來計算技術指標，確保模型能夠生成信號
        df = loader.fetch_data(ticker, test_buffer_start, test_end)
        if df.empty or len(df) < 30: continue
        df['Vol_Change'] = df['Volume'].pct_change()
        df = loader.add_technical_indicators(df)
        df = pd.merge(df, market_df_test, on='Date', how='inner')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if df.empty: continue
        
        # 儲存完整的測試期間資料，用於建構價格矩陣
        raw_test_data[ticker] = df.copy()
        
        # 生成信號
        df_scored = trader.generate_signals(df)
        
        # 模型在 T 日收盤後產生訊號，在 T+1 日執行交易。
        df_scored['Position'] = df_scored['Position'].shift(1)
        df_scored['confidence'] = df_scored['confidence'].shift(1)
        
        # 只保留測試期間的數據
        df_scored = df_scored[df_scored['Date'] >= test_start].copy()
        
        if not df_scored.empty:
            df_scored['Ticker'] = ticker
            test_signals[ticker] = df_scored
        
    if not test_signals:
        print("No valid testing data to backtest.")
        return
        
    # 合併所有測試信號並準備價格矩陣
    df_test_combined = pd.concat(test_signals.values(), ignore_index=True)
    df_test_combined.sort_values(by='Date', inplace=True)
    
    # 準備價格矩陣，確保每個日期都有所有股票的價格（使用前向填充處理缺失值）
    unique_dates = df_test_combined['Date'].unique()
    price_matrix = pd.DataFrame(index=unique_dates)
    
    for ticker in TICKERS:
        if ticker in raw_test_data:
            df_t = raw_test_data[ticker]
            # 只取測試期間的收盤價
            df_t_test = df_t[df_t['Date'] >= test_start].set_index('Date')['Close']
            price_matrix[ticker] = df_t_test
            
    price_matrix.ffill(inplace=True)
    
    # 4. 執行回測
    print("\n[4] Running Backtest Simulation ($1,000,000 Capital)...")
    
    def ensemble_strategy(row, current_position, current_capital, current_price):
        action = None
        quantity = 0
        
        # 根據投票結果決定買賣行為：1.0 為買入/回補信號，-1.0 為賣出/放空信號
        if row.get('Position') == 1.0:
            if current_position < 0:
                # 之前有放空，現在有買入信號，所以回補
                action = 'COVER'
                quantity = abs(current_position)
            elif current_capital > current_price:
                # 買多
                action = 'BUY'
                alloc = current_capital * 0.20 
                quantity = int(alloc // current_price)
            
        elif row.get('Position') == -1.0:
            if current_position > 0:
                # 之前有買多，現在有賣出信號，所以賣出平倉
                action = 'SELL'
                quantity = current_position
            elif enable_short and current_capital > current_price:
                # 如果開啟放空功能且沒有多單部位
                action = 'SHORT'
                alloc = current_capital * 0.20
                quantity = int(alloc // current_price)
            
        return action, quantity

    engine = BacktestEngine(initial_capital=1000000.0)
    metrics = engine.run_daily_batch(df_test_combined, ensemble_strategy, price_matrix)
    
    # 5. 最終結果
    print("\n==================================================")
    print("                 BACKTEST RESULTS                 ")
    print("==================================================")
    print(f"Total Initial Capital:  ${metrics['Total_Initial_Value']:,.2f}")
    print(f"Total Final Value:      ${metrics['Total_Final_Value']:,.2f}")
    print(f"Total Net Profit:       ${metrics['Total_Net_Profit']:,.2f}")
    print(f"Total Return:           {metrics['Total_Return_Pct']:.2f}%")
    print(f"Max Drawdown:           {metrics['Max_Drawdown_Pct']:.2f}%")
    print(f"Win Rate:               {metrics['Win_Rate_Pct']:.2f}%")
    print(f"Total Trades Executed:  {metrics['Total_Trades']}")
    print(f"Total Fees Paid:        ${metrics['Total_Fees_Paid']:,.2f}")
    print("--------------------------------------------------")
    
    print("\n[Performance by Ticker]")
    ticker_metrics = metrics['Ticker_Metrics']
    sorted_tickers = sorted(ticker_metrics.items(), key=lambda x: x[1]['Net_Profit'], reverse=True)
    
    for ticker, m in sorted_tickers:
        if m['Trades'] > 0:
            print(f"{ticker:<8} | Trades: {m['Trades']:<3} | Profit: ${m['Net_Profit']:>9,.2f} | Fees: ${m['Fees_Paid']:>7,.2f}")
            
    print("\n[Trade History (First 10 & Last 10)]")
    history = metrics['Trade_History']
    
    try:
        with open("testlog.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print("Trade history saved to testlog.json")
    except Exception as e:
        print(f"Failed to save trade history: {e}")

    if not history:
        print("No trades executed.")
    else:
        df_hist = pd.DataFrame(history)
        df_hist['Value'] = df_hist['Price'] * df_hist['Quantity']
        display_cols = ['Date', 'Ticker', 'Action', 'Price', 'Quantity', 'Value', 'Fee']
        
        if len(df_hist) <= 20:
            print(df_hist[display_cols].to_string(index=False))
        else:
            print(df_hist[display_cols].head(10).to_string(index=False))
            print("...")
            print(df_hist[display_cols].tail(10).to_string(index=False))
            
    print("==================================================\n")

if __name__ == "__main__":
    main()