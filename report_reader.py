import json
import os
from datetime import datetime

def main():
    signal_file = "/home/node/.openclaw/workspace/trade_cron/latest_signals.json"
    
    # 檢查檔案是否存在
    if not os.path.exists(signal_file):
        print("今日尚未產出任何訊號報表 (檔案不存在)。")
        return
        
    try:
        with open(signal_file, 'r', encoding='utf-8') as f:
            signals = json.load(f)
    except json.JSONDecodeError:
        print("報表格式錯誤，無法讀取。")
        return
        
    # 如果裡面是空的，代表今天模型沒有任何動作
    if not signals:
        print(f"[{datetime.now().strftime('%Y-%m-%d')}] 盤後 AI 量化交易結算：\n今日無任何買賣動作 (空倉/續抱)。")
        return
        
    # 有資料的話就印出整理好的報表
    print(f"[{datetime.now().strftime('%Y-%m-%d')}] 盤後 AI 量化交易報牌 (針對明日開盤)：")
    print("----------------------------------------")
    
    buys = [s for s in signals if s['Signal'] == 'BUY']
    sells = [s for s in signals if s['Signal'] == 'SELL']
    
    if buys:
        print("\n📈 【明日開盤 - 買進建議】")
        for s in buys:
            print(f" - {s['Ticker']} (信心度: {s['Confidence']*100:.1f}%) | 今日收盤: {s['Close']}")
            
    if sells:
        print("\n📉 【明日開盤 - 賣出建議】")
        for s in sells:
            print(f" - {s['Ticker']} (信心度: {(1-s['Confidence'])*100:.1f}%) | 今日收盤: {s['Close']}")
            
    print("----------------------------------------")
    print("請於明日開盤 (09:00) 參考上述訊號掛單。")

if __name__ == "__main__":
    main()
