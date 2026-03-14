import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TICKERS = [
    "0050.TW", "0056.TW", "00878.TW", "00929.TW",
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW",
    "2882.TW", "2891.TW", "2886.TW", "1301.TW", "1303.TW", "2002.TW",
    "2884.TW", "2892.TW", "1216.TW", "2303.TW", "3711.TW", "2885.TW",
    "2880.TW", "3231.TW", "3045.TW", "2883.TW", "5871.TW", "2887.TW",
    "2395.TW", "2412.TW", "2890.TW", "5880.TW", "1101.TW", "2357.TW",
    "2301.TW", "2912.TW", "1326.TW", "2603.TW", "2207.TW", "1304.TW",
    "2324.TW", "6669.TW", "3034.TW", "4938.TW", "3037.TW", "2345.TW",
    "2356.TW", "1590.TW", "5876.TW", "4904.TW", "2379.TW",
]

FEATURES = [
    'Open', 'Close', 'High', 'Low',
    'Body_Size', 'Daily_Range', 'Volume',
    'SMA_5', 'SMA_20', 'SMA_60', 'RSI',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'OBV',
    'Vol_Change', 'TWII_Return', 'TWII_Volume', 'SP500_Return',
]

INITIAL_CAPITAL = 1000000.0
SEQUENCE_LENGTH = 20
NUM_ENSEMBLE_MODELS = 10
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.0005
