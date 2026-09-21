import ccxt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import itertools
import warnings

warnings.filterwarnings("ignore")

def fetch_ccxt_data(symbols, timeframe='1h', limit=2000):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    data = {}
    print(f"🌐 Lade erweiterte Intraday-Daten ({timeframe}, {limit} Kerzen) via CCXT...")
    
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            closes = [x[4] for x in ohlcv]
            timestamps = [pd.to_datetime(x[0], unit='ms') for x in ohlcv]
            
            data[symbol] = pd.Series(closes, index=timestamps)
            print(f"  - {symbol} erfolgreich geladen.")
        except Exception as e:
            print(f"  - Fehler bei {symbol}: {e}")
            
    df = pd.DataFrame(data)
    return df.dropna()

def find_cointegrated_pairs(df):
    n = df.shape[1]
    keys = df.columns
    pairs = []
    
    if n < 2:
        return pairs

    print(f"\n🔄 Analysiere {n*(n-1)//2} Paare auf erweiterter Stundenbasis...")
    
    for i, j in itertools.combinations(range(n), 2):
        asset1 = keys[i]
        asset2 = keys[j]
        
        score, p_value, _ = coint(df[asset1], df[asset2])
        
        # Aufgeweichter, realistischerer Filter: p < 0.05
        if p_value < 0.05:
            pairs.append((asset1, asset2, p_value))
            
    pairs.sort(key=lambda x: x[2])
    return pairs

if __name__ == "__main__":
    # Fokussierter Korb (Hauptsächlich Layer-1 und DeFi)
    symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", 
        "LINK/USDT", "DOT/USDT", "NEAR/USDT", "FET/USDT", 
        "ARB/USDT", "OP/USDT", "UNI/USDT", "AAVE/USDT"
    ]
    
    df_data = fetch_ccxt_data(symbols, timeframe='1h', limit=2000)
    
    best_pairs = find_cointegrated_pairs(df_data)
    
    print("\n🏆 Top Kointegrierte Intraday-Paare (p < 0.05):")
    print("-" * 50)
    if best_pairs:
        for p in best_pairs:
            print(f"{p[0]:<12} / {p[1]:<12} -> p-Wert: {p[2]:.5f}")
    else:
        print("Keine Paare mit p < 0.05 gefunden.")