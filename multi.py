import json
import gc
import sys
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

from data.loader import DataLoader
from engine.backtest import BacktestEngine
from models.lstm import LSTMTrader
from config import (
    TICKERS, FEATURES, INITIAL_CAPITAL, SEQUENCE_LENGTH,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

def prepare_multi_asset_data(loader, tickers, start_date, end_date):
    """Fetches data for multiple assets and merges market indices."""
    try:
        market_df = loader.fetch_market_index(start_date, end_date)
    except Exception as e:
        print(f"Error fetching market index: {e}")
        return {}, pd.DataFrame()
        
    if market_df.empty:
        return {}, pd.DataFrame()

    all_data = {}
    combined_data_for_training = []

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
            combined_data_for_training.append(df)
            
        except Exception as e:
            continue
        
    if not combined_data_for_training:
        return {}, pd.DataFrame()
        
    combined_df = pd.concat(combined_data_for_training, ignore_index=True)
    combined_df.sort_values(by='Date', inplace=True)
    return all_data, combined_df

def run_for_year(test_year, enable_short=False):
    print(f"\n========================================================")
    print(f"--- AI Quant Trading System: Simulation for Year {test_year} ---")
    print(f"========================================================")
    
    loader = DataLoader()
    tickers = TICKERS
    
    features = FEATURES
    
    train_start = f"{test_year-2}-01-01"
    train_end = f"{test_year-1}-12-31"
    
    test_buffer_start = f"{test_year-1}-06-01"
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"
    
    print(f"\n--- Fetching Training Data ({train_start} to {train_end}) ---")
    _, df_train_combined = prepare_multi_asset_data(loader, tickers, train_start, train_end)
    if df_train_combined.empty:
        print(f"No training data for {test_year}.")
        return
        
    print(f"\n--- Fetching Testing Data ({test_buffer_start} to {test_end}) ---")
    test_data_dict, _ = prepare_multi_asset_data(loader, tickers, test_buffer_start, test_end)
    
    print("\n--- Initializing and Training LSTM Model ---")
    trader = LSTMTrader(sequence_length=SEQUENCE_LENGTH, features=features) 
    trader.train(df_train_combined, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
    
    del df_train_combined
    gc.collect()
    
    all_signals = []
    print("\n--- Generating Signals ---")
    for ticker in tickers:
        if ticker not in test_data_dict:
            continue
        df_test = test_data_dict[ticker]
        df_signals_full = trader.generate_signals(df_test)
        
        df_signals_full['Position'] = df_signals_full['Position'].shift(1)
        df_signals_full['confidence'] = df_signals_full['confidence'].shift(1)
        
        df_signals = df_signals_full[df_signals_full['Date'] >= test_start].copy()
        if not df_signals.empty:
            df_signals['Ticker'] = ticker
            all_signals.append(df_signals)
            
    if not all_signals:
        print(f"No signals generated for {test_year}.")
        return
        
    master_timeline = pd.concat(all_signals, ignore_index=True)
    master_timeline.sort_values(by='Date', inplace=True)
    
    unique_dates = master_timeline['Date'].unique()
    price_matrix = pd.DataFrame(index=unique_dates)
    for ticker in tickers:
        if ticker in test_data_dict:
            df_t = test_data_dict[ticker]
            df_t_test = df_t[df_t['Date'] >= test_start].set_index('Date')['Close']
            price_matrix[ticker] = df_t_test
            
    price_matrix.ffill(inplace=True)
    
    print(f"\n--- Starting Shared-Capital Backtest for {test_year} ---")
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    
    def ensemble_strategy(row, current_position, current_capital, current_price):
        action = None
        quantity = 0
        
        position = row.get('Position')
        # Position 可能為 NaN（序列初始階段），需先檢查
        if pd.isna(position):
            return action, quantity
        
        if position == 1.0:
            if current_position < 0:
                action = 'COVER'
                quantity = abs(current_position)
            elif current_capital > current_price:
                action = 'BUY'
                alloc = current_capital * 0.20 
                quantity = int(alloc // current_price)
            
        elif position == -1.0:
            if current_position > 0:
                action = 'SELL'
                quantity = current_position
            elif enable_short and current_capital > current_price:
                action = 'SHORT'
                alloc = current_capital * 0.20
                quantity = int(alloc // current_price)
            
        return action, quantity

    metrics = engine.run_daily_batch(master_timeline, ensemble_strategy, price_matrix)
    
    print(f"\n--- Summary for {test_year} ---")
    print(f"Total Return:    {metrics['Total_Return_Pct']:.2f}%")
    print(f"Final Capital:   ${metrics['Total_Final_Value']:,.2f}")
    print(f"Max Drawdown:    {metrics['Max_Drawdown_Pct']:.2f}%")
    print(f"Win Rate:        {metrics['Win_Rate_Pct']:.2f}%")
    print(f"Total Trades:    {metrics['Total_Trades']}")
    print(f"Total Fees Paid: ${metrics['Total_Fees_Paid']:,.2f}")
    
    # 儲存每年的交易紀錄
    history = metrics['Trade_History']
    log_filename = f"multilog_{test_year}.json"
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"Trade history for {test_year} saved to {log_filename}")
    except Exception as e:
        print(f"Failed to save trade history: {e}")

def main():
    enable_short_input = input("Enable Short Selling? (T/F) [Default: F]: ").strip().upper()
    enable_short = True if enable_short_input == 'T' else False
    print(f"Short Selling is {'Enabled' if enable_short else 'Disabled'} for all years.")
    
    years = [2022, 2023, 2024, 2025]
    for year in years:
        run_for_year(year, enable_short=enable_short)

if __name__ == "__main__":
    main()
