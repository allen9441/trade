from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

class Trade:
    def __init__(self, date: pd.Timestamp, ticker: str, action: str, price: float, quantity: int, fee: float, slippage: float):
        self.date = date
        self.ticker = ticker
        self.action = action  # 'BUY' or 'SELL'
        self.price = price
        self.quantity = quantity
        self.fee = fee
        self.slippage = slippage
        self.total_cost = (price * quantity) + fee + slippage if action == 'BUY' else (price * quantity) - fee - slippage

    def __repr__(self):
        return f"Trade({self.date.date()}, {self.action} {self.quantity} {self.ticker} @ {self.price:.2f}, Fee: {self.fee:.2f})"

class FeeCalculator:
    """
    手續費計算器，以國泰證券為例
    買進手續費 = 成交金額 * 0.1425% * 折扣
    賣出手續費 = 成交金額 * 0.1425% * 折扣 + 證交稅(成交金額 * 0.3%)
    融券借券費等假設為年化利率的日均費用 (簡化處理)
    """
    def __init__(self, discount: float = 0.28, short_interest_rate: float = 0.05):
        self.fee_rate = 0.001425
        self.discount = discount
        self.tax_rate = 0.003
        self.short_interest_rate_daily = short_interest_rate / 365.0
        
    def calculate_buy_fee(self, amount: float) -> float:
        fee = amount * self.fee_rate * self.discount
        return max(20.0, fee) # 最低手續費為20元

    def calculate_sell_fee(self, amount: float) -> float:
        fee = amount * self.fee_rate * self.discount
        tax = amount * self.tax_rate
        return max(20.0, fee) + tax

    def calculate_short_interest(self, amount: float, days: int) -> float:
        """計算融券利息"""
        return amount * self.short_interest_rate_daily * days

class BacktestEngine:
    def __init__(self, initial_capital: float = 1000000.0, fee_calculator=None, slippage_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, int] = {} # 正數為做多，負數為放空
        self.history: List[Trade] = []
        self.fee_calculator = fee_calculator or FeeCalculator()
        self.slippage_rate = slippage_rate # 0.1% 滑點
        self.portfolio_value_history: List[Dict[str, float]] = []
        
        # 記錄多空進場資訊以計算未實現損益
        self.entry_dates: Dict[str, pd.Timestamp] = {}
        self.entry_prices: Dict[str, float] = {}

    def execute_trade(self, date: pd.Timestamp, ticker: str, action: str, price: float, quantity: int):
        amount = price * quantity
        slippage_cost = amount * self.slippage_rate
        
        # BUY: 買多, SELL: 賣出多單, SHORT: 放空, COVER: 回補空單
        is_buying = action in ['BUY', 'COVER']
        actual_price = price * (1 + self.slippage_rate) if is_buying else price * (1 - self.slippage_rate)
        actual_amount = actual_price * quantity
        
        if action == 'BUY':
            fee = self.fee_calculator.calculate_buy_fee(actual_amount)
            total_cost = actual_amount + fee
            
            if self.capital >= total_cost:
                self.capital -= total_cost
                # 更新平均成本 (簡單處理：如果有舊部位，更新平均進場價)
                curr_qty = self.positions.get(ticker, 0)
                if curr_qty > 0 and ticker in self.entry_prices:
                    old_value = curr_qty * self.entry_prices[ticker]
                    self.entry_prices[ticker] = (old_value + (actual_price * quantity)) / (curr_qty + quantity)
                else:
                    self.entry_prices[ticker] = actual_price
                    self.entry_dates[ticker] = date
                
                self.positions[ticker] = curr_qty + quantity
                trade = Trade(date, ticker, action, actual_price, quantity, fee, slippage_cost)
                self.history.append(trade)
                return True
            else:
                print(f"{date.date()}: Insufficient capital to buy {quantity} {ticker}.")
                return False
                
        elif action == 'SELL':
            if self.positions.get(ticker, 0) >= quantity:
                fee = self.fee_calculator.calculate_sell_fee(actual_amount)
                total_revenue = actual_amount - fee
                
                self.capital += total_revenue
                self.positions[ticker] -= quantity
                if self.positions[ticker] == 0:
                    del self.positions[ticker]
                    if ticker in self.entry_dates: del self.entry_dates[ticker]
                    if ticker in self.entry_prices: del self.entry_prices[ticker]
                
                trade = Trade(date, ticker, action, actual_price, quantity, fee, slippage_cost)
                self.history.append(trade)
                return True
            else:
                print(f"{date.date()}: Insufficient position to sell {quantity} {ticker}.")
                return False

        elif action == 'SHORT':
            # 放空：計算賣出手續費+稅
            fee = self.fee_calculator.calculate_sell_fee(actual_amount)
            margin_requirement = actual_amount * 0.9 # 假設融券保證金90%
            total_cost = margin_requirement + fee
            
            if self.capital >= total_cost:
                self.capital += actual_amount - fee
                curr_qty = abs(self.positions.get(ticker, 0)) if self.positions.get(ticker, 0) < 0 else 0
                if curr_qty > 0 and ticker in self.entry_prices:
                    old_value = curr_qty * self.entry_prices[ticker]
                    self.entry_prices[ticker] = (old_value + (actual_price * quantity)) / (curr_qty + quantity)
                else:
                    self.entry_prices[ticker] = actual_price
                    self.entry_dates[ticker] = date
                
                self.positions[ticker] = self.positions.get(ticker, 0) - quantity
                trade = Trade(date, ticker, action, actual_price, quantity, fee, slippage_cost)
                self.history.append(trade)
                return True
            else:
                print(f"{date.date()}: Insufficient capital to short {quantity} {ticker}.")
                return False

        elif action == 'COVER':
            current_short_qty = abs(self.positions.get(ticker, 0))
            if current_short_qty >= quantity and self.positions.get(ticker, 0) < 0:
                fee = self.fee_calculator.calculate_buy_fee(actual_amount)
                
                # 計算利息
                entry_date = self.entry_dates.get(ticker, date)
                days_held = max((date - entry_date).days, 1)
                entry_amount = self.entry_prices.get(ticker, actual_price) * quantity
                interest = self.fee_calculator.calculate_short_interest(entry_amount, days_held)
                
                total_cost = actual_amount + fee + interest
                
                if self.capital >= total_cost:
                    self.capital -= total_cost
                    self.positions[ticker] += quantity
                    if self.positions[ticker] == 0:
                        del self.positions[ticker]
                        if ticker in self.entry_dates: del self.entry_dates[ticker]
                        if ticker in self.entry_prices: del self.entry_prices[ticker]
                    
                    trade = Trade(date, ticker, action, actual_price, quantity, fee + interest, slippage_cost)
                    self.history.append(trade)
                    return True
                else:
                    print(f"{date.date()}: Insufficient capital to cover {quantity} {ticker}.")
                    # 強制平倉即使破產 (允許負值)
                    self.capital -= total_cost
                    self.positions[ticker] += quantity
                    if self.positions[ticker] == 0:
                        del self.positions[ticker]
                        if ticker in self.entry_dates: del self.entry_dates[ticker]
                        if ticker in self.entry_prices: del self.entry_prices[ticker]
                    trade = Trade(date, ticker, action, actual_price, quantity, fee + interest, slippage_cost)
                    self.history.append(trade)
                    return True
            else:
                print(f"{date.date()}: No short position to cover {quantity} {ticker}.")
                return False
                
        return False

    def update_portfolio_value(self, date: pd.Timestamp, current_prices: Dict[str, float]):
        total_position_value = sum(self.positions.get(ticker, 0) * price for ticker, price in current_prices.items())
        total_value = self.capital + total_position_value
        self.portfolio_value_history.append({'Date': date, 'Capital': self.capital, 'Positions': total_position_value, 'Total': total_value})

    def run_daily_batch(self, master_timeline: pd.DataFrame, strategy_func, price_matrix: pd.DataFrame):
        """
        以天為單位執行回測
        """
        if master_timeline.empty:
            return self.generate_detailed_metrics()
            
        unique_dates = master_timeline['Date'].unique()
        
        for current_date in unique_dates:
            todays_events = master_timeline[master_timeline['Date'] == current_date]
            
            # 1. 執行所有賣出和平倉（優先於買入，避免同一天內的買賣衝突）
            sells_covers = todays_events[todays_events['Position'] == -1.0]
            for _, row in sells_covers.iterrows():
                ticker = row['Ticker']
                exec_price = row['Open'] if 'Open' in row else row['Close']
                current_pos = self.positions.get(ticker, 0)
                
                action, quantity = strategy_func(row, current_pos, self.capital, exec_price)
                if action in ['SELL', 'SHORT'] and quantity > 0:
                    self.execute_trade(current_date, ticker, action, exec_price, quantity)

            # 2. 執行所有買入和回補
            buys_covers = todays_events[todays_events['Position'] == 1.0]
            if 'confidence' in buys_covers.columns:
                buys_covers = buys_covers.sort_values(by='confidence', ascending=False)
            elif 'lstm_prob_smooth' in buys_covers.columns:
                buys_covers = buys_covers.sort_values(by='lstm_prob_smooth', ascending=False)
                
            for _, row in buys_covers.iterrows():
                ticker = row['Ticker']
                exec_price = row['Open'] if 'Open' in row else row['Close']
                current_pos = self.positions.get(ticker, 0)
                
                action, quantity = strategy_func(row, current_pos, self.capital, exec_price)
                if action in ['BUY', 'COVER'] and quantity > 0:
                    self.execute_trade(current_date, ticker, action, exec_price, quantity)
                    
            # 3. 更新投資組合價值
            if current_date in price_matrix.index:
                current_prices = price_matrix.loc[current_date].to_dict()
                self.update_portfolio_value(current_date, current_prices)
                
        return self.generate_detailed_metrics()

    def generate_detailed_metrics(self):
        df_history = pd.DataFrame(self.portfolio_value_history)
        if df_history.empty:
            final_value = self.initial_capital
        else:
            final_value = df_history.iloc[-1]['Total']
            
        net_profit = final_value - self.initial_capital
        total_return_pct = (net_profit / self.initial_capital) * 100
        total_fees = sum(t.fee for t in self.history)
        
        ticker_metrics = {}
        trade_history = []
        
        for t in self.history:
            trade_history.append({
                'Date': t.date.strftime('%Y-%m-%d'),
                'Ticker': t.ticker,
                'Action': t.action,
                'Price': t.price,
                'Quantity': t.quantity,
                'Fee': t.fee,
                'Slippage': t.slippage
            })
            
            if t.ticker not in ticker_metrics:
                ticker_metrics[t.ticker] = {'Trades': 0, 'Net_Profit': 0.0, 'Fees_Paid': 0.0}
                
            ticker_metrics[t.ticker]['Trades'] += 1
            ticker_metrics[t.ticker]['Fees_Paid'] += t.fee
            
            # 計算已實現損益
            if t.action in ['BUY', 'COVER']:
                ticker_metrics[t.ticker]['Net_Profit'] -= t.total_cost
            elif t.action in ['SELL', 'SHORT']:
                ticker_metrics[t.ticker]['Net_Profit'] += t.total_cost
                
        last_prices = {}
        for t in self.history:
            last_prices[t.ticker] = t.price
            
        # 計算未實現損益
        for ticker, qty in self.positions.items():
            if qty != 0 and ticker in last_prices:
                current_price = last_prices[ticker]
                if qty > 0:
                    # 多單留倉：加上股票現值 (等同平倉取回現金)
                    ticker_metrics[ticker]['Net_Profit'] += (qty * current_price)
                else:
                    # 空單留倉：qty 為負數。放空時已加回保證金外現金，
                    # 結算需扣除買回股票的成本。qty * current_price 即為負的扣除額。
                    ticker_metrics[ticker]['Net_Profit'] += (qty * current_price)
                
        # 計算最大回撤
        max_drawdown_pct = 0.0
        if not df_history.empty:
            df_history['Cumulative_Max'] = df_history['Total'].cummax()
            df_history['Drawdown'] = (df_history['Total'] - df_history['Cumulative_Max']) / df_history['Cumulative_Max']
            max_drawdown_pct = abs(df_history['Drawdown'].min()) * 100
            
        winning_trades = 0
        total_closed_trades = 0
        # Win Rate 計算
        for ticker, m in ticker_metrics.items():
            if m['Trades'] > 0:
                total_closed_trades += 1
                if m['Net_Profit'] > 0:
                    winning_trades += 1
                    
        win_rate_pct = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
                
        return {
            'Total_Initial_Value': self.initial_capital,
            'Total_Final_Value': final_value,
            'Total_Net_Profit': net_profit,
            'Total_Return_Pct': total_return_pct,
            'Total_Trades': len(self.history),
            'Total_Fees_Paid': total_fees,
            'Max_Drawdown_Pct': max_drawdown_pct,
            'Win_Rate_Pct': win_rate_pct,
            'Ticker_Metrics': ticker_metrics,
            'Trade_History': trade_history
        }

    def get_performance_summary(self):
        df_history = pd.DataFrame(self.portfolio_value_history)
        if df_history.empty:
            return {
                "Initial Capital": self.initial_capital,
                "Final Capital": self.initial_capital,
                "Total Return (%)": 0,
                "Sharpe Ratio": 0,
                "Max Drawdown (%)": 0,
                "Total Trades": 0
            }
            
        initial_value = self.initial_capital
        final_value = df_history.iloc[-1]['Total']
        total_return = (final_value - initial_value) / initial_value
        
        # 計算每日報酬率
        df_history['Daily_Return'] = df_history['Total'].pct_change()
        
        # 計算 Sharpe Ratio (假設無風險利率為 0.01% 日均)
        risk_free_rate = 0.01 / 252
        mean_return = df_history['Daily_Return'].mean()
        std_return = df_history['Daily_Return'].std()
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return != 0 else 0
        
        # Max Drawdown
        df_history['Cumulative_Max'] = df_history['Total'].cummax()
        df_history['Drawdown'] = (df_history['Total'] - df_history['Cumulative_Max']) / df_history['Cumulative_Max']
        max_drawdown = df_history['Drawdown'].min()
        
        return {
            "Initial Capital": initial_value,
            "Final Capital": final_value,
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown * 100,
            "Total Trades": len(self.history)
        }
