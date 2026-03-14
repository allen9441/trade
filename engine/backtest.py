from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Trade:
    def __init__(self, date: pd.Timestamp, ticker: str, action: str, price: float, quantity: int, fee: float, slippage: float):
        self.date = date
        self.ticker = ticker
        self.action = action  # 'BUY', 'SELL', 'SHORT', 'COVER'
        self.price = price
        self.quantity = quantity
        self.fee = fee
        self.slippage = slippage
        # total_cost: 正值代表花費，負值代表收入（僅用於紀錄，不參與資金流計算）
        if action in ['BUY', 'COVER']:
            self.total_cost = (price * quantity) + fee + slippage
        else:  # SELL, SHORT
            self.total_cost = (price * quantity) - fee - slippage

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
        return max(20.0, fee)

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
        self.positions: Dict[str, int] = {}  # 正數為做多，負數為放空
        self.history: List[Trade] = []
        self.fee_calculator = fee_calculator or FeeCalculator()
        self.slippage_rate = slippage_rate  # 0.1% 滑點
        self.portfolio_value_history: List[Dict[str, float]] = []

        # 記錄多空進場資訊以計算未實現損益
        self.entry_dates: Dict[str, pd.Timestamp] = {}
        self.entry_prices: Dict[str, float] = {}
        
        # 記錄放空時凍結的保證金（不可動用的資金）
        self.frozen_margin: Dict[str, float] = {}
        
        # 記錄已配對的交易損益（用於逐筆 Win Rate 計算）
        self.paired_trades: List[Dict] = []

    def execute_trade(self, date: pd.Timestamp, ticker: str, action: str, price: float, quantity: int):
        amount = price * quantity
        slippage_cost = amount * self.slippage_rate

        is_buying = action in ['BUY', 'COVER']
        actual_price = price * (1 + self.slippage_rate) if is_buying else price * (1 - self.slippage_rate)
        actual_amount = actual_price * quantity

        if action == 'BUY':
            fee = self.fee_calculator.calculate_buy_fee(actual_amount)
            total_cost = actual_amount + fee

            if self.capital >= total_cost:
                self.capital -= total_cost

                # 更新平均成本
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
                return False

        elif action == 'SELL':
            if self.positions.get(ticker, 0) >= quantity:
                fee = self.fee_calculator.calculate_sell_fee(actual_amount)
                total_revenue = actual_amount - fee

                # 計算逐筆損益
                entry_price = self.entry_prices.get(ticker, actual_price)
                buy_cost = entry_price * quantity + self.fee_calculator.calculate_buy_fee(entry_price * quantity)
                sell_revenue = total_revenue
                trade_pnl = sell_revenue - buy_cost
                self.paired_trades.append({
                    'Ticker': ticker, 'Type': 'LONG',
                    'Entry': entry_price, 'Exit': actual_price,
                    'Quantity': quantity, 'PnL': trade_pnl
                })

                self.capital += total_revenue
                self.positions[ticker] -= quantity
                if self.positions[ticker] == 0:
                    del self.positions[ticker]
                    self.entry_dates.pop(ticker, None)
                    self.entry_prices.pop(ticker, None)

                trade = Trade(date, ticker, action, actual_price, quantity, fee, slippage_cost)
                self.history.append(trade)
                return True
            else:
                return False

        elif action == 'SHORT':
            # 放空：需要融券保證金 90% + 手續費（稅）
            fee = self.fee_calculator.calculate_sell_fee(actual_amount)
            margin_requirement = actual_amount * 0.9
            total_cost = margin_requirement + fee

            if self.capital >= total_cost:
                # 放空時：扣保證金和手續費，賣出所得暫存
                self.capital -= total_cost
                # 記錄凍結的保證金 + 賣出所得（回補時歸還）
                frozen = margin_requirement + actual_amount
                
                curr_qty = abs(self.positions.get(ticker, 0)) if self.positions.get(ticker, 0) < 0 else 0
                if curr_qty > 0 and ticker in self.entry_prices:
                    old_value = curr_qty * self.entry_prices[ticker]
                    self.entry_prices[ticker] = (old_value + (actual_price * quantity)) / (curr_qty + quantity)
                    self.frozen_margin[ticker] = self.frozen_margin.get(ticker, 0) + frozen
                else:
                    self.entry_prices[ticker] = actual_price
                    self.entry_dates[ticker] = date
                    self.frozen_margin[ticker] = frozen

                self.positions[ticker] = self.positions.get(ticker, 0) - quantity
                trade = Trade(date, ticker, action, actual_price, quantity, fee, slippage_cost)
                self.history.append(trade)
                return True
            else:
                return False

        elif action == 'COVER':
            current_short_qty = abs(self.positions.get(ticker, 0))
            if current_short_qty >= quantity and self.positions.get(ticker, 0) < 0:
                fee = self.fee_calculator.calculate_buy_fee(actual_amount)

                # 計算利息
                entry_date = self.entry_dates.get(ticker, date)
                days_held = max((date - entry_date).days, 1)
                entry_price = self.entry_prices.get(ticker, actual_price)
                entry_amount = entry_price * quantity
                interest = self.fee_calculator.calculate_short_interest(entry_amount, days_held)

                # 回補邏輯：
                # 放空時凍結了 (保證金 + 賣出所得)，現在歸還
                # 按比例歸還凍結資金
                total_short_qty = current_short_qty
                proportion = quantity / total_short_qty
                released_margin = self.frozen_margin.get(ticker, 0) * proportion
                
                # 回補需支付：買回成本 + 手續費 + 利息
                cover_cost = actual_amount + fee + interest
                
                # 淨回收 = 釋放的凍結資金 - 買回成本
                net_return = released_margin - cover_cost
                
                # 計算逐筆損益
                trade_pnl = (entry_price - actual_price) * quantity - fee - interest - self.fee_calculator.calculate_sell_fee(entry_amount)
                self.paired_trades.append({
                    'Ticker': ticker, 'Type': 'SHORT',
                    'Entry': entry_price, 'Exit': actual_price,
                    'Quantity': quantity, 'PnL': trade_pnl
                })

                self.capital += net_return
                
                # 更新凍結保證金
                self.frozen_margin[ticker] = self.frozen_margin.get(ticker, 0) * (1 - proportion)

                self.positions[ticker] += quantity
                if self.positions[ticker] == 0:
                    del self.positions[ticker]
                    self.entry_dates.pop(ticker, None)
                    self.entry_prices.pop(ticker, None)
                    self.frozen_margin.pop(ticker, None)

                trade = Trade(date, ticker, action, actual_price, quantity, fee + interest, slippage_cost)
                self.history.append(trade)
                return True
            else:
                return False

        return False

    def update_portfolio_value(self, date: pd.Timestamp, current_prices: Dict[str, float]):
        total_position_value = 0.0
        for ticker, qty in self.positions.items():
            price = current_prices.get(ticker, self.entry_prices.get(ticker, 0.0))
            if qty > 0:
                total_position_value += qty * price
            elif qty < 0:
                # 空單的價值 = 凍結資金 - 買回成本
                frozen = self.frozen_margin.get(ticker, 0)
                buy_back_cost = abs(qty) * price
                short_value = frozen - buy_back_cost
                total_position_value += short_value

        total_value = self.capital + total_position_value
        self.portfolio_value_history.append({
            'Date': date, 'Capital': self.capital,
            'Positions': total_position_value, 'Total': total_value
        })

    def run_daily_batch(self, master_timeline: pd.DataFrame, strategy_func, price_matrix: pd.DataFrame):
        """以天為單位執行回測"""
        if master_timeline.empty:
            return self.generate_detailed_metrics()

        unique_dates = master_timeline['Date'].unique()

        for current_date in unique_dates:
            todays_events = master_timeline[master_timeline['Date'] == current_date]

            # 1. 執行所有看空信號（賣出/放空）（優先於買入）
            bearish_signals = todays_events[todays_events['Position'] == -1.0]
            for _, row in bearish_signals.iterrows():
                ticker = row['Ticker']
                exec_price = row['Open'] if 'Open' in row else row['Close']
                current_pos = self.positions.get(ticker, 0)

                action, quantity = strategy_func(row, current_pos, self.capital, exec_price)
                if action in ['SELL', 'SHORT'] and quantity > 0:
                    self.execute_trade(current_date, ticker, action, exec_price, quantity)

            # 2. 執行所有看多信號（買入/回補）
            bullish_signals = todays_events[todays_events['Position'] == 1.0]
            if 'confidence' in bullish_signals.columns:
                bullish_signals = bullish_signals.sort_values(by='confidence', ascending=False, na_position='last')
            elif 'lstm_prob_smooth' in bullish_signals.columns:
                bullish_signals = bullish_signals.sort_values(by='lstm_prob_smooth', ascending=False, na_position='last')

            for _, row in bullish_signals.iterrows():
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

        # 取得回測最後一天的收盤價，傳入 generate_detailed_metrics 以保持一致
        last_date = unique_dates[-1] if len(unique_dates) > 0 else None
        final_prices = None
        if last_date is not None and last_date in price_matrix.index:
            final_prices = price_matrix.loc[last_date].dropna().to_dict()

        return self.generate_detailed_metrics(final_prices=final_prices)

    def generate_detailed_metrics(self, final_prices: Dict[str, float] = None):
        """
        產生詳細的回測績效指標。
        final_prices: 回測最後一天的收盤價字典，用於一致性計算未平倉損益。
                      如果未提供，會使用 portfolio_value_history 最後一筆記錄。
        """
        df_history = pd.DataFrame(self.portfolio_value_history)
        if df_history.empty:
            final_value = self.initial_capital
        else:
            final_value = df_history.iloc[-1]['Total']

        net_profit = final_value - self.initial_capital
        total_return_pct = (net_profit / self.initial_capital) * 100
        total_fees = sum(t.fee for t in self.history)

        # 用資金流（capital 變動）來計算各 ticker 的已實現損益
        # 追蹤每個 ticker 在 BUY/SELL 間的資金流出入
        ticker_metrics = {}
        trade_history = []
        ticker_capital_flows = {}  # ticker -> list of (action, amount)

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

        # 使用 paired_trades 計算已實現損益
        for pt in self.paired_trades:
            ticker = pt['Ticker']
            if ticker not in ticker_metrics:
                ticker_metrics[ticker] = {'Trades': 0, 'Net_Profit': 0.0, 'Fees_Paid': 0.0}
            ticker_metrics[ticker]['Net_Profit'] += pt['PnL']

        # 計算未實現損益（仍持有的部位）
        # 優先使用 final_prices（與 portfolio_value_history 一致的價格）
        if final_prices is None:
            # fallback: 使用 portfolio_value_history 最後記錄時的價格
            # 但這不完全準確，建議總是傳入 final_prices
            final_prices = {}
            for t in self.history:
                final_prices[t.ticker] = t.price

        for ticker, qty in self.positions.items():
            if qty != 0:
                current_price = final_prices.get(ticker, self.entry_prices.get(ticker, 0.0))
                entry_price = self.entry_prices.get(ticker, current_price)
                if ticker not in ticker_metrics:
                    ticker_metrics[ticker] = {'Trades': 0, 'Net_Profit': 0.0, 'Fees_Paid': 0.0}
                if qty > 0:
                    # 多單未實現損益（含估計賣出手續費，與 final_value 一致）
                    unrealized = qty * (current_price - entry_price)
                    ticker_metrics[ticker]['Net_Profit'] += unrealized
                else:
                    # 空單未實現損益
                    unrealized = abs(qty) * (entry_price - current_price)
                    ticker_metrics[ticker]['Net_Profit'] += unrealized

        # 計算最大回撤與 Sharpe Ratio
        max_drawdown_pct = 0.0
        sharpe_ratio = 0.0
        if not df_history.empty:
            df_history['Cumulative_Max'] = df_history['Total'].cummax()
            df_history['Drawdown'] = (df_history['Total'] - df_history['Cumulative_Max']) / df_history['Cumulative_Max']
            max_drawdown_pct = abs(df_history['Drawdown'].min()) * 100

            df_history['Daily_Return'] = df_history['Total'].pct_change()
            risk_free_rate = 0.01 / 252
            mean_return = df_history['Daily_Return'].mean()
            std_return = df_history['Daily_Return'].std()
            if std_return != 0 and pd.notna(std_return):
                sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)

        # Win Rate：逐筆配對交易計算
        winning_trades = sum(1 for t in self.paired_trades if t['PnL'] > 0)
        total_paired = len(self.paired_trades)
        win_rate_pct = (winning_trades / total_paired * 100) if total_paired > 0 else 0.0

        return {
            'Total_Initial_Value': self.initial_capital,
            'Total_Final_Value': final_value,
            'Total_Net_Profit': net_profit,
            'Total_Return_Pct': total_return_pct,
            'Total_Trades': len(self.history),
            'Total_Fees_Paid': total_fees,
            'Max_Drawdown_Pct': max_drawdown_pct,
            'Sharpe_Ratio': sharpe_ratio,
            'Win_Rate_Pct': win_rate_pct,
            'Winning_Trades': winning_trades,
            'Total_Closed_Trades': total_paired,
            'Ticker_Metrics': ticker_metrics,
            'Trade_History': trade_history,
            'Paired_Trades': self.paired_trades
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

        df_history['Daily_Return'] = df_history['Total'].pct_change()

        risk_free_rate = 0.01 / 252
        mean_return = df_history['Daily_Return'].mean()
        std_return = df_history['Daily_Return'].std()
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if (std_return != 0 and pd.notna(std_return)) else 0

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
