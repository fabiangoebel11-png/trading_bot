import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import itertools
import os
import warnings

warnings.filterwarnings("ignore")

def get_basket_data(tickers, start_date, end_date):
    file_path = f"data/basket_{start_date}_{end_date}.csv"
    
    if os.path.exists(file_path):
        print(f"⚡ Lade Basket-Daten aus Cache ({file_path})...")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    print(f"🌐 Lade Basket-Daten aus dem Internet...")
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # FIX 1: Zuerst Spalten löschen, die komplett leer sind (Dead Coins)
    df = df.dropna(axis=1, how='all')
    
    # FIX 2: Dann erst Lücken füllen und fehlerhafte Rest-Zeilen löschen
    df = df.ffill().dropna() 
    
    df.to_csv(file_path)
    return df

def find_cointegrated_pairs(df):
    n = df.shape[1]
    keys = df.columns
    pairs = []
    
    # Schutz vor leerer Matrix
    if n < 2:
        print("❌ Zu wenig gültige Daten heruntergeladen.")
        return pairs

    print(f"🔄 Berechne Kointegration für {n*(n-1)//2} mögliche Paare...")
    
    for i, j in itertools.combinations(range(n), 2):
        asset1 = keys[i]
        asset2 = keys[j]
        
        score, p_value, _ = coint(df[asset1], df[asset2])
        
        if p_value < 0.05:
            pairs.append((asset1, asset2, p_value))
            
    pairs.sort(key=lambda x: x[2])
    return pairs

if __name__ == "__main__":
    # Aktualisierter Korb mit neuen Tickern
    basket = [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD", 
        "LINK-USD", "DOT-USD", "POL-USD", "NEAR-USD", "FET-USD",
        "INJ-USD", "RENDER-USD", "OP-USD", "ARB-USD"
    ]
    
    data = get_basket_data(basket, "2024-01-01", "2024-09-01")
    
    best_pairs = find_cointegrated_pairs(data)
    
    print("\n🏆 Top Kointegrierte Paare (p < 0.05):")
    print("-" * 40)
    for p in best_pairs:
        print(f"{p[0]:<10} / {p[1]:<10} -> p-Wert: {p[2]:.4f}")
        
    if not best_pairs:
        print("Keine stark kointegrierten Paare gefunden.")