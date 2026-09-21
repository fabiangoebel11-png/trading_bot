import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import os

def get_data(ticker1, ticker2, start_date, end_date):
    # 1. Wir bauen uns einen Dateinamen für den Cache
    file_path = f"data/{ticker1}_{ticker2}_{start_date}_{end_date}.csv"
    
    # 2. Wenn die Datei schon existiert, laden wir sie lokal (extrem schnell)
    if os.path.exists(file_path):
        print(f"⚡ Lade Daten in Millisekunden aus lokalem Cache ({file_path})...")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    # 3. Wenn nicht, laden wir sie aus dem Internet und speichern sie
    print(f"🌐 Lade Daten aus dem Internet für {ticker1} und {ticker2}...")
    df1 = yf.download(ticker1, start=start_date, end=end_date)['Close']
    df2 = yf.download(ticker2, start=start_date, end=end_date)['Close']
    
    df = pd.concat([df1, df2], axis=1)
    df.columns = [ticker1, ticker2]
    df = df.dropna()
    
    # In den data/ Ordner speichern
    df.to_csv(file_path)
    return df

def check_cointegration(df, t1, t2):
    score, p_value, _ = coint(df[t1], df[t2])
    print(f"Cointegration p-Wert: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Die Zeitreihen sind kointegriert! (Optimal für Pairs Trading)")
    else:
        print("❌ Keine signifikante Kointegration. (Zu riskant)")

    model = sm.OLS(df[t1], df[t2])
    results = model.fit()
    
    # FIX: .iloc[0] statt [0] zwingt Pandas, die Position zu nehmen
    hedge_ratio = results.params.iloc[0]
    print(f"Hedge Ratio: {hedge_ratio:.4f}")
    
    spread = df[t1] - hedge_ratio * df[t2]
    return spread

# --- Ausführung ---
if __name__ == "__main__":
    asset1 = "BTC-USD"
    asset2 = "ETH-USD"
    
    data = get_data(asset1, asset2, "2024-01-01", "2024-09-01")
    
    print("-" * 30)
    spread = check_cointegration(data, asset1, asset2)
    
    # Den Spread visuell prüfen
    plt.figure(figsize=(10, 4))
    plt.plot(spread)
    plt.axhline(spread.mean(), color='red', linestyle='--')
    plt.title(f"Spread zwischen {asset1} und {asset2}")
    plt.show()