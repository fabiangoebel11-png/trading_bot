import os
import ccxt
import time
from dotenv import load_dotenv

load_dotenv()

# Setze hier temporär deine TESTNET Keys ein
API_KEY = "DEIN_TESTNET_API_KEY"
API_SECRET = "DEIN_TESTNET_API_SECRET"

def run_bybit_test():
    print("1. Initialisiere Bybit Testnet...")
    exchange = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  # Wir wollen Futures
        }
    })
    
    # ZWINGEND für das Testnet!
    exchange.set_sandbox_mode(True)
    
    try:
        print("\n2. Lade Märkte...")
        exchange.load_markets()
        
        # WICHTIG: Bei Bybit Swaps über CCXT heißt das Symbol "BTC/USDT:USDT"
        symbol = "BTC/USDT:USDT" 
        
        print(f"\n3. Prüfe Symbol: {symbol}")
        market = exchange.market(symbol)
        print(f"Maker Fee: {market['maker'] * 100}%, Taker Fee: {market['taker'] * 100}%")
        print(f"Min Order Size: {market['limits']['amount']['min']}")
        
        print("\n4. Hole Balance...")
        balance = exchange.fetch_balance()
        usdt_free = balance.get('USDT', {}).get('free', 0)
        print(f"Verfügbare USDT: {usdt_free}")
        
        print("\n5. Hole Ticker...")
        ticker = exchange.fetch_ticker(symbol)
        print(f"Aktueller {symbol} Preis: {ticker['last']}")
        
        if usdt_free < 10:
            print("\nNicht genug Fake-USDT. Bitte auf testnet.bybit.com Faucets nutzen.")
            return

        print("\n6. Setze Margin Mode & Leverage (Cross 10x)...")
        try:
            exchange.set_margin_mode('cross', symbol)
            exchange.set_leverage(10, symbol)
            print("Margin & Leverage erfolgreich gesetzt!")
        except Exception as e:
            print(f"Info Margin/Lev (oft normal, wenn bereits gesetzt): {e}")

        # Erzeuge eine Limit-Buy-Order weit unter dem aktuellen Preis, 
        # damit sie garantiert ins Orderbuch geht (Maker) und wir sie canceln können.
        buy_price = ticker['last'] * 0.90  
        amount = 0.001  # 0.001 BTC
        
        print(f"\n7. Erstelle Post-Only Limit Buy Order (Maker Test)...")
        print(f"Menge: {amount} zu Preis: {buy_price}")
        
        # postOnly stellt sicher, dass wir Maker sind.
        order = exchange.create_order(
            symbol=symbol, 
            type='limit', 
            side='buy', 
            amount=amount, 
            price=buy_price,
            params={'postOnly': True} 
        )
        print(f"Order erfolgreich platziert! Order-ID: {order['id']}")
        
        print("\n8. Prüfe offene Orders...")
        time.sleep(2) # Kurz warten
        open_orders = exchange.fetch_open_orders(symbol)
        print(f"Gefundene offene Orders: {len(open_orders)}")
        
        print(f"\n9. Breche Order ab...")
        exchange.cancel_order(order['id'], symbol)
        print("Order gecancelt! Test erfolgreich beendet.")

    except Exception as e:
        print(f"\nFEHLER beim Testlauf: {e}")

if __name__ == "__main__":
    run_bybit_test()