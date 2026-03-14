import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from data.loader import DataLoader
from models.lstm import LSTMTrader
from engine.backtest import BacktestEngine
from config import (
    TICKERS, FEATURES, INITIAL_CAPITAL, SEQUENCE_LENGTH,
    NUM_ENSEMBLE_MODELS, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    BASE_DIR,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    
    parallel_training_input = input("Enable Parallel Multi-Model Training? (T/F) [Default: T]: ").strip().upper()
    parallel_training = False if parallel_training_input == 'F' else True
    
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
    print(f"Training Mode:   {'Parallel' if parallel_training else 'Sequential'}")
    print("==================================================\n")
    
    loader = DataLoader()
    features = FEATURES
    
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
    print(f"\n[2] Training {NUM_ENSEMBLE_MODELS}-Model Ensemble (This will take a while)...")
    trader = LSTMTrader(sequence_length=SEQUENCE_LENGTH, features=features)
    trader.train(df_train, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE, parallel=parallel_training)
    
    # 3. 準備測試集並產生信號
    print("\n[3] Fetching Testing Data & Voting...")
    # 為確保測試期間的技術指標計算正確，需從測試期間的前幾個月開始擷取數據，以便生成足夠的序列數據
    market_df_test = loader.fetch_market_index(test_buffer_start, test_end)
    test_signals = {}
    raw_test_data = {}
    
    for ticker in TICKERS:
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
        
        # 計算預測真實目標
        df_scored['Future_Return'] = df_scored['Close'].pct_change().shift(-1)
        df_scored['True_Target'] = (df_scored['Future_Return'] > 0.002).astype(int)
        
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
    
    # 評估預測指標 (ROC-AUC, Precision, Histogram)
    print("\n[3.5] Evaluating Model Predictions...")
    df_eval_metrics = df_test_combined.dropna(subset=['lstm_prob_smooth', 'True_Target', 'Future_Return', 'signal'])
    
    if not df_eval_metrics.empty:
        from sklearn.metrics import roc_auc_score, precision_score
        import matplotlib.pyplot as plt
        
        y_true = df_eval_metrics['True_Target']
        y_prob = df_eval_metrics['lstm_prob_smooth']
        # 使用 Position == 1.0（實際觸發買入的時刻）作為預測正例，
        # 而非 signal == 1.0（持續性狀態信號），才能正確反映
        # 「每次觸發買入時，隔天確實漲超 0.2% 的比例」
        y_pred = (df_eval_metrics['Position'] == 1.0).astype(int)
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            precision = precision_score(y_true, y_pred, zero_division=0)
            n_buy_signals = y_pred.sum()
            print(f"ROC-AUC Score:  {roc_auc:.4f}")
            print(f"Precision (Buy triggers): {precision:.4f} ({n_buy_signals} buy signals)")
            
            # 繪製機率分佈直方圖
            plt.figure(figsize=(10, 6))
            plt.hist(y_prob, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Distribution of Predicted Probabilities (Test Period)')
            plt.xlabel('Predicted Probability of > 0.2% Gain')
            plt.ylabel('Frequency')
            plt.axvline(x=np.median(y_prob), color='r', linestyle='--', label=f'Median ({np.median(y_prob):.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            hist_path = os.path.join(BASE_DIR, 'models', 'pred_prob_hist.png')
            plt.savefig(hist_path)
            print("Prediction probability histogram is saved.")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not calculate metrics. {e}")
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
        
        position = row.get('Position')
        # Position 可能為 NaN，需先檢查
        if pd.isna(position):
            return action, quantity
        
        # 根據投票結果決定買賣行為：1.0 為買入/回補信號，-1.0 為賣出/放空信號
        if position == 1.0:
            if current_position < 0:
                # 之前有放空，現在有買入信號，所以回補
                action = 'COVER'
                quantity = abs(current_position)
            elif current_capital > current_price:
                # 買多
                action = 'BUY'
                alloc = current_capital * 0.20 
                quantity = int(alloc // current_price)
            
        elif position == -1.0:
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

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
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
    print(f"Sharpe Ratio:           {metrics['Sharpe_Ratio']:.4f}")
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
            
    history = metrics['Trade_History']
    
    try:
        with open("testlog.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print("Trade history saved to testlog.json")
    except Exception as e:
        print(f"Failed to save trade history: {e}")

if __name__ == "__main__":
    main()
