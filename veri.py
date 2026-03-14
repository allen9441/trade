import sys
import json
import pandas as pd
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
        with open("test_log.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: log not found.")
        sys.exit(1)

    cur = 0 
    positions = {}
    frozen_margin = {}  # 記錄放空時凍結的保證金
    entry_prices = {}   # 記錄進場價格
    entry_dates = {}    # 記錄進場日期
    
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
            prev_qty = positions.get(ticker, 0)
            if prev_qty > 0 and ticker in entry_prices:
                # 加倉時計算加權平均成本
                old_value = prev_qty * entry_prices[ticker]
                entry_prices[ticker] = (old_value + trans) / (prev_qty + qty)
            else:
                entry_prices[ticker] = price
                entry_dates[ticker] = log['Date']
            positions[ticker] = prev_qty + qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: -{cost:.2f}")
            
        elif action == 'SELL':
            revenue = trans - fee
            cur += revenue
            positions[ticker] = positions.get(ticker, 0) - qty
            if positions.get(ticker, 0) == 0:
                positions.pop(ticker, None)
                entry_prices.pop(ticker, None)
                entry_dates.pop(ticker, None)
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: +{revenue:.2f}")
            
        elif action == 'SHORT':
            # 放空：扣保證金(90%) + 手續費；凍結保證金+賣出所得
            margin = trans * 0.9
            total_cost = margin + fee
            frozen = margin + trans  # 凍結的資金（保證金 + 賣出所得）
            cur -= total_cost
            frozen_margin[ticker] = frozen_margin.get(ticker, 0) + frozen
            prev_qty = abs(positions.get(ticker, 0)) if positions.get(ticker, 0) < 0 else 0
            if prev_qty > 0 and ticker in entry_prices:
                # 加倉時計算加權平均成本
                old_value = prev_qty * entry_prices[ticker]
                entry_prices[ticker] = (old_value + trans) / (prev_qty + qty)
            else:
                entry_prices[ticker] = price
                entry_dates[ticker] = log['Date']
            positions[ticker] = positions.get(ticker, 0) - qty
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Cash: -{total_cost:.2f} (margin+fee, frozen: {frozen:.2f})")
            
        elif action == 'COVER':
            # 回補：釋放凍結資金，支付買回成本 + 手續費（fee 欄位已含利息）
            current_short_qty = abs(positions.get(ticker, 0))
            proportion = qty / current_short_qty if current_short_qty > 0 else 1.0
            released = frozen_margin.get(ticker, 0) * proportion
            cover_cost = trans + fee  # fee 已含利息
            net_return = released - cover_cost
            cur += net_return
            frozen_margin[ticker] = frozen_margin.get(ticker, 0) * (1 - proportion)
            positions[ticker] = positions.get(ticker, 0) + qty
            if positions.get(ticker, 0) == 0:
                positions.pop(ticker, None)
                entry_prices.pop(ticker, None)
                entry_dates.pop(ticker, None)
                frozen_margin.pop(ticker, None)
            print(f"{log['Date']} | {action} {qty} {ticker} @ {price:.2f} | Released: {released:.2f}, Cost: {cover_cost:.2f}, Net: {net_return:+.2f}")

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
                    close_price = df.iloc[-1]['Close']
                    if qty > 0:
                        # 多單：直接用 qty * close_price
                        val = qty * close_price
                        base += val
                        print(f"{ticker}: {qty} shares (LONG) @ {close_price:.2f} = {val:.2f}")
                    else:
                        # 空單：凍結資金 - 買回成本 = 未實現損益
                        frozen = frozen_margin.get(ticker, 0)
                        cover_cost = abs(qty) * close_price
                        val = frozen - cover_cost  # 回補後能拿回的淨值
                        base += val
                        entry_p = entry_prices.get(ticker, 0)
                        print(f"{ticker}: {qty} shares (SHORT, entry ~{entry_p:.2f}) @ {close_price:.2f} | Frozen: {frozen:.2f}, Cover Cost: {cover_cost:.2f}, Net: {val:.2f}")
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
    main()
