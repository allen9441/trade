import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional

class DataLoader:
    def __init__(self):
        pass

    def fetch_data(self, ticker: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """
        擷取指定股票的歷史數據，並進行基本清理。
        
        Args:
            ticker (str): 股票標的（例如 "2330.TW"）。
            start_date (str): 起始日期，格式為 "YYYY-MM-DD"。
            end_date (str, optional): 結束日期，格式為 "YYYY-MM-DD"。預設為 None（今天）。
            interval (str): 數據間隔（例如 "1d", "1h"）。預設為 "1d"。
            
        Returns:
            pd.DataFrame: 包含日期、開盤價、收盤價、最高價、最低價、成交量等基本數據的 DataFrame。
        """
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            if df.empty:
                print(f"Warning: No data found for {ticker}.")
                return df
            
            # 確保 'Date' 是一列而非索引
            df.reset_index(inplace=True)
            
            # 如果下載的數據有多層列索引，則將其展平
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # 只保留必要的列，並重命名以避免與市場指數列衝突
            df.dropna(inplace=True)
            
            print(f"Successfully fetched {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算並添加技術指標到 DataFrame 中。
        """
        if df.empty:
            return df
            
        # 基本特徵
        df['Body_Size'] = df['Close'] - df['Open']
        df['Daily_Range'] = df['High'] - df['Low']
            
        # 移動平均線 (SMA)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_60'] = df['Close'].rolling(window=60).mean()
        
        # 相對強弱指標 (RSI) — 使用 Wilder's EMA
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 布林格帶 (Bollinger Bands)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        # 布林帶寬度
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # 能量潮 (On-Balance Volume, OBV)
        # 如果今天收盤價高於昨天，則 OBV 增加今天的成交量；如果收盤價低於昨天，則 OBV 減少今天的成交量；如果收盤價與昨天相同，則 OBV 不變。
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # 處理無限值和缺失值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df

    def fetch_market_index(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """擷取市場指數的歷史數據，並計算相關特徵。"""
        print(f"Fetching Market Indices from {start_date} to {end_date}...")
        twii = pd.DataFrame()
        sp500 = pd.DataFrame()
        try:
            # TWII
            twii = yf.download("^TWII", start=start_date, end=end_date, progress=False)
            if not twii.empty:
                twii.reset_index(inplace=True)
                if isinstance(twii.columns, pd.MultiIndex):
                    twii.columns = twii.columns.get_level_values(0)
                twii = twii[['Date', 'Close', 'Volume']].rename(columns={'Close': 'TWII_Close', 'Volume': 'TWII_Volume'})
                twii['TWII_Return'] = twii['TWII_Close'].pct_change()

            # S&P 500
            sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
            if not sp500.empty:
                sp500.reset_index(inplace=True)
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = sp500.columns.get_level_values(0)
                sp500 = sp500[['Date', 'Close']].rename(columns={'Close': 'SP500_Close'})
                sp500['SP500_Return'] = sp500['SP500_Close'].pct_change()
            
            # merge — 以台股交易日為基準，left join S&P 500
            # S&P 500 與台股交易日不完全同步（時區、假日不同），
            # 使用 ffill + bfill 確保首尾都不會因缺值被 dropna 丟棄。
            if not twii.empty and not sp500.empty:
                market_df = pd.merge(twii, sp500, on='Date', how='left')
                sp500_cols = ['SP500_Close', 'SP500_Return']
                market_df[sp500_cols] = market_df[sp500_cols].ffill().bfill()
            elif not twii.empty:
                market_df = twii
            elif not sp500.empty:
                market_df = sp500
            else:
                print("Warning: Both TWII and S&P 500 data are empty.")
                return pd.DataFrame()

            market_df.dropna(inplace=True)
            return market_df
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()
