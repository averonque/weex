import requests
import pandas as pd
from datetime import datetime
import pytz
import feedparser   
import requests
import subprocess
import json 
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat


XAI_API_KEY = os.getenv("XAI_API_KEY")

# --- Config ---
SPOT_BASE = "https://api-spot.weex.com"
SYMBOL = "BTCUSDT_SPBL"   # spot perpetual symbol
PERIOD = "1m"             # candle interval
LIMIT = 1000              # number of candles to fetch
TZ = pytz.timezone("America/New_York")  # Jewcator uses NY time
app = FastAPI(title="Grok Trading Decision API", version="1.0.0")

# --- Compute session stats ---
def compute_session(df, start_hour, end_hour, name):
    mask = (df["dt"].dt.hour >= start_hour) & (df["dt"].dt.hour < end_hour)
    session = df.loc[mask]
    if session.empty:
        return None
    return {
        "name": name,
        "start": session["dt"].iloc[0].isoformat(),
        "end": session["dt"].iloc[-1].isoformat(),
        "high": float(session["high"].max()),
        "low": float(session["low"].min()),
        "open": float(session["open"].iloc[0]),
        "close": float(session["close"].iloc[-1])
    }
    
    

def get_depth(symbol="BTCUSDT_SPBL", limit=20):

    # Define the curl command as a string (exactly how you'd type it in cmd)
    cmd = 'curl https://api-spot.weex.com/api/v2/market/depth?symbol=BTCUSDT_SPBL&limit=20'


    # Run the command and capture output
    result = subprocess.run(  ["cmd", "/c", cmd] , capture_output=True, text=True)

    # Print status and output
   
    tmp = result.stdout
  
    data = json.loads(tmp)
    tmp = data["data"]
  #  print(tmp)
    
    return tmp



def get_candles(symbol=SYMBOL, period=PERIOD, limit=LIMIT):
    """
    Fetch OHLCV candles from WEEX and convert numeric columns to float.
    """
    url = f"{SPOT_BASE}/api/v2/market/candles"
    params = {"symbol": symbol, "period": period, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume","turnover"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ)

    # Convert numeric columns to float
    for col in ["open","high","low","close","volume","turnover"]:
        df[col] = df[col].astype(float)

    return df

def compute_features(df, depth):
    """
    Compute ret5, volatility, order book imbalance, and spread.
    """
    # ret5: 5-bar return
    ret5 = (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6]

    # volatility: rolling std of returns
    volatility = df["close"].pct_change().rolling(10).std().iloc[-1]

    # order book imbalance
    bids = depth["bids"][:10]
    asks = depth["asks"][:10]
    bSum = sum(float(p)*float(q) for p,q in bids)
    aSum = sum(float(p)*float(q) for p,q in asks)
    obImbalance = (bSum - aSum) / (bSum + aSum + 1e-9)

    # spread
    spread = float(asks[0][0]) - float(bids[0][0])

    return ret5, volatility, obImbalance, spread

def check_macro_block():
    """
    Parse ForexFactory RSS feed and check for high-impact events within ±30 minutes.
    """
    feed = feedparser.parse("https://www.forexfactory.com/calendar/rss")
    now = datetime.utcnow()
    for entry in feed.entries:
        desc = entry.get("description","")
        if "Impact: High" in desc:
            pubDate = entry.get("published","")
            try:
                event_time = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
                if abs((event_time - now).total_seconds()) <= 30*60:
                    return True
            except Exception:
                continue
    return False



def get_features(symbol=SYMBOL):
    """
    Aggregate all features into a dict.
    """
    df = get_candles(symbol)
   
    depth = get_depth(symbol)
    ret5, volatility, obImbalance, spread = compute_features(df, depth)
    macro_block = check_macro_block()
    return {
       
        "ret5": ret5,
        "volatility": volatility,
        "obImbalance": obImbalance,
        "spread": spread,
        "macro_block": macro_block
    }





# --- Main ---
if __name__ == "__main__":
    print("GOOOD")


@app.get("/health") 
def health():
    df = get_candles()

    sessions = []
    sessions.append(compute_session(df, 2, 5, "LDN"))   # London
    sessions.append(compute_session(df, 7, 12, "HUNT")) # Hunt
    sessions.append(compute_session(df, 20, 24, "ASIA"))# Asia
    sessions.append(compute_session(df, 9, 11, "NY"))   # New York

    sessions = [s for s in sessions if s]
    features = get_features()

    result = {"symbol": SYMBOL, "sessions": sessions,  "ret5": features["ret5"],
        "volatility": features["volatility"],
        "obImbalance": features["obImbalance"],
        "spread": features["spread"],
        "macro_block": features["macro_block"] }
    print(result)
 


    XAI_API_URL = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {YOUR_XAI_API_KEY}"}

    prompt = f"""

    Features:
    {result}
    """

    payload = {
        "model": "grok-4-latest",   # example model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    resp = requests.post(XAI_API_URL, headers=headers, json=payload)
    print(resp.json())


    XAI_API_URL = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "   You are a trading signal analyst. Given the following market features, decide buy/sell/hold.Explain reasoning step by step in JSON."
            },
            {   
                "role": "user",
                "content": prompt
            }
        ],
        "model": "grok-4.1-fast",
        "stream": False,
        "temperature": 0
    }

    resp = requests.post(XAI_API_URL, headers=headers, json=payload)

    print("Status code:", resp.status_code)
    print("Response:", resp.text)

    return {"status": "ok"}