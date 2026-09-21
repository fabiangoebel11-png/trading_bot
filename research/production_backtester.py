import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import warnings

warnings.filterwarnings("ignore")

def fetch_resampled_10m_data(symbol, target_candles=30000):
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    print(f"📥 Lade 5m-Basisdaten für {symbol} und resample auf 10m...")
    
    # Da Binance kein 10m unterstützt, laden wir 5m (unterstützt) und fassen je 2 Kerzen zusammen
    target_5m_limit = target_candles * 2
    all_ohlcv = []
    since = exchange.milliseconds() - (target_5m_limit * 5 * 60 * 1000)
    
    while len(all_ohlcv) < target_5m_limit:
        try:
            limit = min(1000, target_5m_limit - len(all_ohlcv))
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=limit)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            print(f"  - {symbol}: {len(all_ohlcv)}/{target_5m_limit} 5m-Kerzen...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  - Fehler bei {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Resample von 5m auf 10m
    df_10m = df.resample('10min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_10m['close'].tail(target_candles)

def run_production_backtest(symbol1, symbol2, target_candles=25000, leverage=2.0):
    s1 = fetch_resampled_10m_data(symbol1, target_candles)
    s2 = fetch_resampled_10m_data(symbol2, target_candles)
    df = pd.DataFrame({symbol1: s1, symbol2: s2}).dropna()
    print(f"✅ Synchronisiert: {len(df)} gemeinsame 10-Minuten-Kerzen.")
    
    P1 = df[symbol1]
    P2 = df[symbol2]
    
    # 1. Rollendes Hedge-Ratio (144 Kerzen = 24 Stunden bei 10m Taktung)
    window_ols = 144
    hedge_ratios = []
    
    print("🔄 Berechne rollendes Hedge-Ratio...")
    for i in range(len(df)):
        if i < window_ols:
            hedge_ratios.append(np.nan)
        else:
            sub_p1 = P1.iloc[i-window_ols:i]
            sub_p2 = P2.iloc[i-window_ols:i]
            model = sm.OLS(sub_p1, sub_p2).fit()
            hedge_ratios.append(model.params.iloc[0])
            
    df['Hedge_Ratio'] = hedge_ratios
    df = df.dropna()
    
    P1 = df[symbol1]
    P2 = df[symbol2]
    hr = df['Hedge_Ratio']
    
    # 2. Spread & Z-Score mit rollendem Mean/Std (72 Kerzen = 12h)
    z_window = 72
    df['Spread'] = P1 - hr * P2
    rolling_mean = df['Spread'].rolling(window=z_window).mean()
    rolling_std = df['Spread'].rolling(window=z_window).std()
    df['Z_Score'] = (df['Spread'] - rolling_mean) / rolling_std
    
    # 3. Trend-Filter & Strenge Schwellenwerte
    btc_sma = P1.rolling(window=360).mean() # Trend-Filter
    df['Market_Trend'] = (P1 - btc_sma).abs() / btc_sma
    trend_threshold = 0.05
    
    df = df.dropna()
    
    entry_z = 3.0    
    stop_loss_z = 4.5 
    maker_fee = 0.0002 
    
    positions = []
    current_pos = 0
    
    print("🔄 Führe Event-Loop aus...")
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
    df['Execution_Position'] = df['Position'].shift(1).fillna(0)
    
    # 4. PnL & Maker-Kosten mit Hebel
    dP1 = P1.diff()
    dP2 = P2.diff()
    
    df['Strategy_PnL'] = df['Execution_Position'] * (dP1 - hr * dP2) * leverage
    capital_base = P1 + hr * P2
    
    df['Strategy_Return'] = df['Strategy_PnL'] / capital_base
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    
    df['Trade'] = df['Execution_Position'].diff().abs().fillna(0)
    df['Strategy_Return'] = df['Strategy_Return'] - (df['Trade'] * maker_fee * leverage)
    
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Market'] = (1 + P1.pct_change()).cumprod()
    
    return df

if __name__ == "__main__":
    results = run_production_backtest("BTC/USDT", "ETH/USDT", target_candles=20000, leverage=2.0)
    
    strat_ret = (results['Cumulative_Strategy'].iloc[-1] - 1) * 100
    mkt_ret = (results['Cumulative_Market'].iloc[-1] - 1) * 100
    trades = results['Trade'].sum()
    
    print("\n📊 Ergebnis des Produktions-Modells (10m Resampled, Maker, 3.0 Z-Score):")
    print("=" * 65)
    print(f"🔹 Strategie-Rendite (Netto): {strat_ret:+.2f}%")
    print(f"🔹 Buy & Hold (BTC):          {mkt_ret:+.2f}%")
    print(f"🔢 Anzahl Trades:             {int(trades)}")
    print("=" * 65)