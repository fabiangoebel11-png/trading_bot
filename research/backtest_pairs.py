import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

def fetch_historical_data(symbol, timeframe='1h', target_candles=26280):
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    print(f"📥 Lade {target_candles} Kerzen ({timeframe}) für {symbol} via Pagination...")
    
    all_ohlcv = []
    # Startzeitpunkt berechnen (ca. 3 Jahre in der Vergangenheit in Millisekunden)
    since = exchange.milliseconds() - (target_candles * 3600 * 1000)
    
    while len(all_ohlcv) < target_candles:
        try:
            limit = min(1000, target_candles - len(all_ohlcv))
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            
            if not ohlcv:
                break
                
            since = ohlcv[-1][0] + 1  
            all_ohlcv.extend(ohlcv)
            
            print(f"  - {symbol}: {len(all_ohlcv)}/{target_candles} Kerzen geladen...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  - Fehler beim Laden von {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['close']

def fetch_pair_data(symbol1, symbol2, target_candles=26280):
    s1 = fetch_historical_data(symbol1, timeframe='1h', target_candles=target_candles)
    s2 = fetch_historical_data(symbol2, timeframe='1h', target_candles=target_candles)
    
    # In einen gemeinsamen DataFrame zusammenführen und Lücken bereinigen
    df = pd.DataFrame({symbol1: s1, symbol2: s2}).dropna()
    print(f"✅ Synchronisiert: {len(df)} gemeinsame Kerzen für {symbol1} und {symbol2}.")
    return df

def run_leveraged_backtest(df, asset1, asset2, window=48, entry_z=2.2, stop_loss_z=3.5, leverage=2.0, fee=0.0005):
    print(f"🔄 Starte gehebelten Backtest ({leverage}x Hebel) mit Trend-Filter & Stop-Loss über {len(df)} Kerzen...")
    
    P1 = df[asset1]
    P2 = df[asset2]
    
    # 1. Hedge Ratio via OLS Regression berechnen
    model = sm.OLS(P1, P2)
    results = model.fit()
    hedge_ratio = results.params.iloc[0]
    
    # 2. Spread & Z-Score berechnen
    df['Spread'] = P1 - hedge_ratio * P2
    rolling_mean = df['Spread'].rolling(window=window).mean()
    rolling_std = df['Spread'].rolling(window=window).std()
    df['Z_Score'] = (df['Spread'] - rolling_mean) / rolling_std
    
    # 3. Trend-Filter (Regime-Erkennung)
    btc_sma = P1.rolling(window=100).mean()
    df['Market_Trend'] = (P1 - btc_sma).abs() / btc_sma
    trend_threshold = 0.08
    
    df = df.dropna()
    P1 = df[asset1]
    P2 = df[asset2]
    
    positions = []
    current_pos = 0
    
    for i in range(len(df)):
        z = df['Z_Score'].iloc[i]
        is_trending = df['Market_Trend'].iloc[i] > trend_threshold
        
        if abs(z) > stop_loss_z:
            current_pos = 0
        elif current_pos != 0:
            if (current_pos == 1 and z >= 0) or (current_pos == -1 and z <= 0):
                current_pos = 0
        else:
            if is_trending:
                current_pos = 0
            else:
                if z < -entry_z:
                    current_pos = 1
                elif z > entry_z:
                    current_pos = -1
                    
        positions.append(current_pos)
        
    df['Position'] = positions
    # 1-Stunden Latenz-Puffer
    df['Execution_Position'] = df['Position'].shift(1).fillna(0)
    
    # 4. Gehebelte PnL & Kostenberechnung
    dP1 = P1.diff()
    dP2 = P2.diff()
    
    df['Strategy_PnL'] = df['Execution_Position'] * (dP1 - hedge_ratio * dP2) * leverage
    capital_base = P1 + hedge_ratio * P2
    
    df['Strategy_Return'] = df['Strategy_PnL'] / capital_base
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    
    df['Trade'] = df['Execution_Position'].diff().abs().fillna(0)
    df['Strategy_Return'] = df['Strategy_Return'] - (df['Trade'] * fee * leverage)
    
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Market'] = (1 + P1.pct_change()).cumprod()
    
    return df

if __name__ == "__main__":
    pair = ("BTC/USDT", "ETH/USDT")
    
    # 3 Jahre historische Daten laden (~26.280 Stundenkerzen)
    data = fetch_pair_data(pair[0], pair[1], target_candles=26280)
    
    # Gehebelter Backtest mit 2x Hebel
    results = run_leveraged_backtest(data, pair[0], pair[1], leverage=2.0)
    
    strategy_return = (results['Cumulative_Strategy'].iloc[-1] - 1) * 100
    market_return = (results['Cumulative_Market'].iloc[-1] - 1) * 100
    trades_count = results['Trade'].sum()
    
    print("\n📊 Detaillierter Performance-Vergleich (3 Jahre, 2x Hebel):")
    print("=" * 65)
    print(f"🔹 Strategie-Gesamtrendite (Netto 2x):  {strategy_return:+.2f}%")
    print(f"🔹 Buy & Hold Gesamtrendite ({pair[0]}):    {market_return:+.2f}%")
    print(f"⚖️ Differenz / Outperformance:         {strategy_return - market_return:+.2f}%")
    print(f"🔢 Anzahl durchgeführter Trades:       {int(trades_count)}")
    print("=" * 65)
    
    plt.figure(figsize=(10, 5))
    plt.plot(results.index, results['Cumulative_Strategy'], label='Pairs Trading (3 Jahre 2x Netto)', color='blue')
    plt.plot(results.index, results['Cumulative_Market'], label=f'Buy & Hold ({pair[0]})', color='gray', alpha=0.5)
    plt.title(f"3-Jahres Backtest (2x Hebel): {pair[0]} vs {pair[1]}")
    plt.legend()
    plt.grid(True)
    plt.show()