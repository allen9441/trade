import sys
import json
from datetime import datetime
from data.loader import DataLoader

def format_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD.")
        sys.exit(1)

def main():
    end_date_input = input("Enter the last testing date (End Date) for valuation [YYYYMMDD]: ")
    val_date = format_date(end_date_input)
    print(f"\n--- Valuing open positions as of {val_date} ---")

    try:
        with open("testlog.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: testlog.json not found.")
        sys.exit(1)

    cur = 0 
    positions = {}
    
    for log in data:
        ticker = log['Ticker']
        action = log['Action']
        price = log['Price']
        qty = log['Quantity']
        fee = log['Fee']
        
        trans = price * qty
        
        if action == 'BUY':
            cost = trans + fee
            cur -= cost
            positions[ticker] = positions.get(ticker, 0) + qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: -{cost:.2f}")
            
        elif action == 'SELL':
            revenue = trans - fee
            cur += revenue
            positions[ticker] = positions.get(ticker, 0) - qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: +{revenue:.2f}")
            
        elif action == 'SHORT':
            # 放空得到保證金外的現金
            revenue = trans - fee
            cur += revenue
            positions[ticker] = positions.get(ticker, 0) - qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: +{revenue:.2f}")
            
        elif action == 'COVER':
            # 回補需要支付買入總額 + 手續費 + 利息
            cost = trans + fee
            cur -= cost
            positions[ticker] = positions.get(ticker, 0) + qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: -{cost:.2f}")

    # 計算未平倉部位的當前價值
    base = 0
    loader = DataLoader()
    
    print(f"\n--- Open Positions (Valued at Close Price on {val_date}) ---")
    
    # 篩出仍有持倉的股票
    open_tickers = {t: q for t, q in positions.items() if q != 0}
    
    if not open_tickers:
        print("No open positions.")
    else:
        for ticker, qty in open_tickers.items():
            # 獲取該股票到 val_date 附近的資料，確保能取到最近收盤價
            start_fetch = (datetime.strptime(val_date, "%Y-%m-%d") - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            try:
                df = loader.fetch_data(ticker, start_fetch, val_date)
                if not df.empty:
                    # 取最後一天的收盤價
                    close_price = df.iloc[-1]['Close']
                    val = qty * close_price
                    base += val
                    print(f"{ticker}: {qty} shares @ {close_price:.2f} = {val:.2f}")
                else:
                    print(f"{ticker}: {qty} shares | WARNING: No price data found ending {val_date}")
            except Exception as e:
                print(f"{ticker}: Failed to fetch data - {e}")
        
    tot = cur + base
    
    print("\n--- Summary ---")
    print(f"Net Cash Flow (Realized): {cur:.2f}")
    print(f"Open Positions Value:     {base:.2f}")
    print(f"Total Net Profit:         {tot:.2f}")

if __name__ == "__main__":
    import pandas as pd
    main()
