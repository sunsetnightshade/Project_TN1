import websocket
import json
import threading
import time
from rich.live import Live
from rich.table import Table

# --- YOUR API KEYS ---
FINNHUB_KEY = "d779c61r01qp6afkob6gd779c61r01qp6afkob70"
TWELVE_KEY = "c520a929a1e54b3dbda68dccc623b50d"

# --- THE TICKERS ---
us_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ADBE", "TXN"]
in_tickers_base = ["INFY", "TCS", "HCLTECH", "TECHM", "WIPRO", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS", 
                   "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE", "AXISBANK", "KOTAKBANK"]

# Twelve Data requires the ":NSE" tag to find the Indian stocks
in_tickers = [f"{t}:NSE" for t in in_tickers_base]

# The Brain's Memory: Stores the prices
live_prices = {t: 0.0 for t in us_tickers + in_tickers}

# ==========================================
# THREAD 1: US WEBSOCKET (Finnhub)
# ==========================================
def on_us_message(ws, msg):
    try:
        data = json.loads(msg)
        if data.get('type') == 'trade':
            for trade in data['data']:
                live_prices[trade['s']] = trade['p']
    except Exception:
        pass

def run_us():
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_KEY}", on_message=on_us_message)
    ws.on_open = lambda w: [w.send(json.dumps({"type": "subscribe", "symbol": t})) for t in us_tickers]
    ws.run_forever()

# ==========================================
# THREAD 2: INDIA WEBSOCKET (Twelve Data)
# ==========================================
def on_in_message(ws, msg):
    try:
        data = json.loads(msg)
        if data.get('event') == 'price':
            symbol = data.get('symbol')
            price = data.get('price')
            if symbol and price:
                live_prices[symbol] = float(price)
    except Exception:
        pass

def run_in():
    ws = websocket.WebSocketApp(f"wss://ws.twelvedata.com/v1/quotes/price?apikey={TWELVE_KEY}", on_message=on_in_message)
    # Tell Twelve Data we want all 20 Indian stocks
    ws.on_open = lambda w: w.send(json.dumps({
        "action": "subscribe", 
        "params": {"symbols": ",".join(in_tickers)}
    }))
    ws.run_forever()

# ==========================================
# UI ENGINE: GLITCH-FREE SCOREBOARD
# ==========================================
def generate_table():
    # This creates the beautiful single table
    table = Table(title="NIGHTSHADE PURE WEBSOCKET TERMINAL", title_style="bold cyan")
    
    table.add_column("Ticker", justify="left", style="white", no_wrap=True)
    table.add_column("Region", justify="center", style="magenta")
    table.add_column("Live Price", justify="right", style="green")

    # Add US Stocks to the top half
    for t in us_tickers:
        p = live_prices[t]
        table.add_row(t, "US", f"${p:.2f}" if p > 0 else "Waiting for trade...")
        
    # Add Indian Stocks to the bottom half
    for t in in_tickers:
        p = live_prices[t]
        clean_name = t.split(":")[0] # Removes the ":NSE" so it looks clean on screen
        table.add_row(clean_name, "IN", f"₹{p:.2f}" if p > 0 else "Waiting for trade...")
        
    return table

if __name__ == "__main__":
    # Start the data pipes
    threading.Thread(target=run_us, daemon=True).start()
    threading.Thread(target=run_in, daemon=True).start()

    # The 'Live' function handles the screen. No clearing, no glitching!
    with Live(generate_table(), refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(0.25)
            live.update(generate_table())