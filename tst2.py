import os
import math
import json
import requests
import pandas as pd
import feedparser
from datetime import datetime, timezone, timedelta
import pytz
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
import subprocess
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import math
from datetime import time as dtime
import time
import hmac
import hashlib
import base64
from bs4 import BeautifulSoup
import pytz
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio

# ------------------ Config ------------------
SPOT_BASE = "https://api-spot.weex.com"
CONTRACT_BASE = "https://api-contract.weex.com"
SYMBOL = os.getenv("WEEX_SYMBOL", "BTCUSDT_SPBL")
DAILY_PERIOD = os.getenv("DAILY_PERIOD", "1d")
INTRA_PERIOD = os.getenv("INTRA_PERIOD", "5m")
CANDLE_LIMIT_DAILY = int(os.getenv("CANDLE_LIMIT_DAILY", "200"))
CANDLE_LIMIT_INTRA = int(os.getenv("CANDLE_LIMIT_INTRA", "1500"))
TZ = pytz.timezone(os.getenv("SESSION_TZ", "America/New_York"))  # Jewcator uses NY time

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_API_KEY = os.getenv("XAI_API_KEY") or os.getenv("YOUR_XAI_API_KEY")  # allow both env var names
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")

api_key = "weex_f67d25e3b8c4d7639e7deb7c558016bb"
secret_key = "29056e6c4da2ea623bdfbf6fb223a48f7d192622e31803e6e64c5ceee3bc2611"
access_passphrase = "weex652694794"
INTERVAL_SECONDS = int(os.getenv("ANALYZE_INTERVAL_SECONDS", "60"))  # run every 60s


app = FastAPI(title="Dynamic Range + WEEX + xAI", version="1.0.0")

runner_lock = asyncio.Lock()

# --- Session windows in NY time ---
NY_TZ = TZ  # uses your configured America/New_York

HUNT_START = dtime(7, 0)
HUNT_END   = dtime(11, 45)

HODL_START = dtime(11, 45)
HODL_END   = dtime(19, 0)

ASIA_START = dtime(20, 0)
ASIA_END   = dtime(23, 59, 59)  # treat up to midnight as Asia ref

LONDON_START = dtime(2, 0)
LONDON_END   = dtime(5, 0)



@dataclass
class LiquidityZone:
    timeframe: str   # "HTF" or "LTF"
    type: str        # "BigStop" or "Stop"
    side: str        # "above" or "below"
    level: float
    priority: int    # lower = higher priority
    note: str

def detect_htf_stops(daily_df) -> List[LiquidityZone]:
    zones = []
    if daily_df.empty:
        return zones
    last = daily_df.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    zones.append(LiquidityZone("HTF","BigStop","above",high,1,"Daily high"))
    zones.append(LiquidityZone("HTF","BigStop","below",low,1,"Daily low"))
    return zones

def detect_ltf_stops(intra_df) -> List[LiquidityZone]:
    zones = []
    if intra_df.empty:
        return zones
    # simple equal highs/lows detection
    highs = intra_df["high"].tail(20)
    lows = intra_df["low"].tail(20)
    eq_high = float(highs.max())
    eq_low = float(lows.min())
    zones.append(LiquidityZone("LTF","Stop","above",eq_high,3,"Intraday equal high"))
    zones.append(LiquidityZone("LTF","Stop","below",eq_low,3,"Intraday equal low"))
    return zones

def rank_liquidity_zones(htf: List[LiquidityZone], ltf: List[LiquidityZone]) -> List[LiquidityZone]:
    zones = htf + ltf
    zones.sort(key=lambda z: z.priority)
    return zones



def time_based_opens(now: pd.Timestamp) -> Dict[str,float]:
    ny = now.tz_convert("America/New_York")
    # True Daily Open = NY midnight
    tdo = ny.replace(hour=0,minute=0,second=0,microsecond=0)
    return {
        "true_daily_open": tdo.timestamp(),
        # weekly/monthly can be added similarly
    }


def bias_from_opens(price: float, opens: Dict[str,float]) -> str:
    tdo = opens.get("true_daily_open")
    if tdo and price < tdo:
        return "bullish"
    elif tdo and price > tdo:
        return "bearish"
    return "neutral"


@app.get("/liquidity/context")
def liquidity_context():
    try:
        daily = weex_get_candles(SYMBOL, DAILY_PERIOD, CANDLE_LIMIT_DAILY)
        intra = weex_get_candles(SYMBOL, INTRA_PERIOD, CANDLE_LIMIT_INTRA)

        htf_zones = detect_htf_stops(daily)
        ltf_zones = detect_ltf_stops(intra)
        zones = rank_liquidity_zones(htf_zones, ltf_zones)

        last_close = float(intra["close"].iloc[-1]) if len(intra) else None
        opens = time_based_opens(intra.index[-1]) if len(intra) else {}
        bias = bias_from_opens(last_close, opens) if last_close else None

        payload = {
            "symbol": SYMBOL,
            "bias": bias,
            "zones": [z.__dict__ for z in zones],
            "opens": opens,
        }
        return sanitize_json(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"liquidity_context error: {e}")


def in_window(ts: pd.Timestamp, start: dtime, end: dtime) -> bool:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC").tz_convert(NY_TZ)
    h = ts.timetz()
    return (h >= start) and (h < end)

def session_mask(df: pd.DataFrame, start: dtime, end: dtime) -> pd.Series:
    idx = df.index.tz_convert(NY_TZ)
    return (idx.time >= start) & (idx.time < end)

def sanitize_json(obj):
    import math
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None

def percenter(top: float, bottom: float, close: float):
    if top is None or bottom is None or top == bottom:
        return None
    return round(((close - bottom) / (top - bottom)) * 100, 1)




@dataclass
class AsianRange:
    start: str
    end: str
    high: float
    low: float
    mid: float

def build_asian_reference(intra_df: pd.DataFrame) -> Optional[AsianRange]:
    mask = session_mask(intra_df, ASIA_START, ASIA_END)
    asia = intra_df.loc[mask]
    if asia.empty:
        return None
    high = float(asia["high"].max())
    low = float(asia["low"].min())
    mid = (high + low) / 2.0
    return AsianRange(
        start=asia.index[0].isoformat(),
        end=asia.index[-1].isoformat(),
        high=high, low=low, mid=mid
    )



def recent_liquidity_sweeps(intra_df: pd.DataFrame, lookback: int = 60) -> Dict[str, bool]:
    """
    Heuristics:
    - sweep_highs: a new local high followed by immediate failure (close back below prior high)
    - sweep_lows: a new local low followed by immediate failure (close back above prior low)
    """
    if len(intra_df) < lookback + 5:
        return {"sweep_highs": False, "sweep_lows": False}
    window = intra_df.iloc[-lookback:]
    prior_high = window["high"].iloc[:-1].max()
    prior_low = window["low"].iloc[:-1].min()
    last = window.iloc[-1]
    sweep_highs = (last["high"] > prior_high) and (last["close"] < prior_high)
    sweep_lows = (last["low"] < prior_low) and (last["close"] > prior_low)
    return {"sweep_highs": bool(sweep_highs), "sweep_lows": bool(sweep_lows)}

def old_levels_raided(daily_df: pd.DataFrame, intra_df: pd.DataFrame, lookback_days: int = 7) -> Dict[str, bool]:
    """
    Use recent daily extremes as 'old highs/lows' and check if intraday has pierced them.
    """
    dd = daily_df.tail(lookback_days)
    if dd.empty:
        return {"old_high_raided": False, "old_low_raided": False}
    old_high = float(dd["high"].max())
    old_low = float(dd["low"].min())
    last_intra = intra_df.iloc[-1]
    return {
        "old_high_raided": bool(last_intra["high"] > old_high),
        "old_low_raided": bool(last_intra["low"] < old_low),
    }

def htf_exhaustion_2h(intra_df: pd.DataFrame, direction: str, bars_2h: int = 3, step_minutes: int = 5) -> bool:
    """
    Aggregate intraday bars into ~2H candles by grouping. Check 3+ consecutive candles
    printing against intended direction.
    - Intended 'long' => look for 3+ red 2H candles (exhaustion down)
    - Intended 'short' => look for 3+ green 2H candles (exhaustion up)
    """
    if intra_df.empty:
        return False
    # Build 2H groups via fixed-size aggregation (approximation)
    group_size = max(1, int((120) / step_minutes))
    closes = intra_df["close"].tail(group_size * bars_2h * 2)
    opens = intra_df["open"].tail(group_size * bars_2h * 2)
    # Create synthetic 2H candles
    chunks = []
    for i in range(0, len(closes), group_size):
        sl = slice(i, i + group_size)
        if i + group_size <= len(closes):
            c_open = float(opens.iloc[sl][0])
            c_close = float(closes.iloc[sl][-1])
            chunks.append((c_open, c_close))
    if len(chunks) < bars_2h:
        return False
    last_n = chunks[-bars_2h:]
    colors = ["red" if (c[1] < c[0]) else "green" for c in last_n]
    if direction == "long":
        return colors.count("red") >= bars_2h
    else:
        return colors.count("green") >= bars_2h




@dataclass
class HuntCandidate:
    time: str
    side: str
    entry: float
    reason: str
    asia_ref: Dict[str, float]
    targets: Dict[str, float]  # t1 (old levels/cluster), t2 (EQ or next pool)

def price_location_vs_daily(close: float, low40: Optional[float], high40: Optional[float], eq40: Optional[float]) -> Optional[str]:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [low40, high40, eq40]):
        return None
    if close > eq40 and close < high40:
        return "premium"
    if close < eq40 and close > low40:
        return "discount"
    if close >= high40:
        return "above_premium_edge"
    if close <= low40:
        return "below_discount_edge"
    return None

def build_hunt_entries(daily_df: pd.DataFrame, intra_df: pd.DataFrame, eq40: Optional[float], high40: Optional[float], low40: Optional[float], asia_ref: Optional[AsianRange]) -> List[HuntCandidate]:
    """
    Hunt window 07:00–11:45 NY: require confluence:
    - price at premium/discount area
    - recent sweeps occurred
    - old levels raided
    - HTF exhaustion 2H aligns
    Skip entries if within Asia window (we don't enter in Asia), but allow Asia range as reference.
    """
    if intra_df.empty:
        return []
    mask = session_mask(intra_df, HUNT_START, HUNT_END)
    hunt_df = intra_df.loc[mask]
    if hunt_df.empty:
        return []

    sweeps = recent_liquidity_sweeps(intra_df)
    raids = old_levels_raided(daily_df, intra_df)
    entries: List[HuntCandidate] = []

    for ts, row in hunt_df.iterrows():
        close = float(row["close"])
        loc = price_location_vs_daily(close, low40, high40, eq40)
        if loc not in ("premium", "discount", "above_premium_edge", "below_discount_edge"):
            continue

        # Direction intent: mean-reversion at edges or with sweeps
        intent = None
        reason_parts = []

        if loc in ("premium", "above_premium_edge"):
            # prefer short when in/above premium
            intent = "short"
            reason_parts.append(f"location={loc}")
            if sweeps["sweep_highs"]:
                reason_parts.append("recent_sweep_highs")
        if loc in ("discount", "below_discount_edge"):
            # prefer long when in/below discount
            intent = "long"
            reason_parts.append(f"location={loc}")
            if sweeps["sweep_lows"]:
                reason_parts.append("recent_sweep_lows")

        # Old levels raided adds confluence
        if intent == "short" and raids["old_high_raided"]:
            reason_parts.append("old_high_raided")
        if intent == "long" and raids["old_low_raided"]:
            reason_parts.append("old_low_raided")

        # HTF 2H exhaustion against intended direction
        if intent and htf_exhaustion_2h(intra_df, direction=intent, bars_2h=3, step_minutes=5):
            reason_parts.append("htf_2h_exhaustion")

        if intent and reason_parts:
            # Targets: use Asia ref pools and EQ
            asia_targets = {}
            if asia_ref:
                asia_targets = {"asia_high": asia_ref.high, "asia_low": asia_ref.low, "asia_mid": asia_ref.mid}

            t2 = eq40 if eq40 is not None and not math.isnan(eq40) else None  # EQ as secondary
            # t1: nearest old level depending on side
            t1 = None
            if intent == "short" and asia_ref:
                # aim toward asia_mid/low
                t1 = asia_ref.mid
            elif intent == "long" and asia_ref:
                t1 = asia_ref.mid

            entries.append(HuntCandidate(
                time=ts.isoformat(),
                side=intent,
                entry=close,
                reason="|".join(reason_parts),
                asia_ref=asia_targets,
                targets={"t1": t1, "t2": t2}
            ))

    return entries

@dataclass
class HodlPlan:
    start: str
    end: str
    rules: List[str]

def build_hodl_plan(intra_df: pd.DataFrame) -> Optional[HodlPlan]:
    mask = session_mask(intra_df, HODL_START, HODL_END)
    hodl_df = intra_df.loc[mask]
    if hodl_df.empty:
        return None
    return HodlPlan(
        start=hodl_df.index[0].isoformat(),
        end=hodl_df.index[-1].isoformat(),
        rules=[
            "If partial targets met during Hunt -> move SL to BE at HODL start",
            "Hold remaining toward liquidity pools or HTF targets",
            "If still unmet by HODL end -> trail or leave 10% at BE until 20:00 cutoff",
            "No new entries in HODL"
        ]
    )

@app.get("/rush_hours/context")
def rush_hours_context():
    try:
        # Data
        daily = weex_get_candles(SYMBOL, DAILY_PERIOD, CANDLE_LIMIT_DAILY)
        intra = weex_get_candles(SYMBOL, INTRA_PERIOD, CANDLE_LIMIT_INTRA)

        # Daily range context (40-day primary)
        # Compute eq40/high40/low40 like in DynamicRangeModel
        lows40 = daily["low"].rolling(40).min().shift(1)
        highs40 = daily["high"].rolling(40).max().shift(1)
        eq40_series = (lows40 + highs40) / 2.0

        eq40 = float(eq40_series.dropna().iloc[-1]) if eq40_series.notna().any() else None
        high40 = float(highs40.dropna().iloc[-1]) if highs40.notna().any() else None
        low40 = float(lows40.dropna().iloc[-1]) if lows40.notna().any() else None

        last_close = float(intra["close"].iloc[-1]) if len(intra) else None
        percent40 = percenter(high40, low40, last_close) if last_close is not None else None

        # Asian reference range
        asia_ref = build_asian_reference(intra)

        # Hunt candidates (7:00–11:45 NY), skip entries during Asia; Asia used only as reference
        hunt_entries = build_hunt_entries(daily, intra, eq40, high40, low40, asia_ref)

        # HODL plan (11:45–19:00 NY)
        hodl = build_hodl_plan(intra)

        payload = {
            "symbol": SYMBOL,
            "daily_range_40": {"low40": low40, "high40": high40, "eq40": eq40, "percent40": percent40},
            "asian_reference": asia_ref.__dict__ if asia_ref else None,
            "hunt_candidates": [e.__dict__ for e in hunt_entries],
            "hodl_plan": hodl.__dict__ if hodl else None,
            "notes": [
                "Skip new entries in Asia; use Asian range as reference frame.",
                "Hunt focuses on NY AM rush hours; more confluence near open increases probability.",
            ],
        }
        return sanitize_json(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rush_hours_context error: {e}")








# ------------------ WEEX client ------------------

def weex_get_candles(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """
    WEEX spot candles: GET /api/v2/market/candles?symbol=...&period=...&limit=...
    Assumes response data as arrays: [ts, open, high, low, close, volume, turnover]
    """
    url = f"{SPOT_BASE}/api/v2/market/candles"
    params = {"symbol": symbol, "period": period, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=502, detail="WEEX candles response empty or invalid")

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    # Convert to floats
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize time to tz-aware NY time
    df["dt"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ)
    df = df.sort_values("dt").reset_index(drop=True)
    df.set_index("dt", inplace=True)
    return df[["open", "high", "low", "close", "volume", "turnover"]]

def weex_get_depth(symbol: str, limit: int = 20) -> Dict[str, Any]:
    """
    WEEX spot depth: GET /api/v2/market/depth?symbol=...&limit=...
    Returns dict with 'bids' and 'asks' arrays.
    """
    cmd = 'curl https://api-spot.weex.com/api/v2/market/depth?symbol=BTCUSDT_SPBL&limit=20'


    # Run the command and capture output
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    # Print status and output
   
    tmp = result.stdout
  
    data = json.loads(tmp)
    data = data["data"]
    if not data or "bids" not in data or "asks" not in data:
        raise HTTPException(status_code=502, detail="WEEX depth response invalid")
    return data

# ------------------ Feature computation ------------------

def compute_session(df: pd.DataFrame, start_hour: int, end_hour: int, name: str):
    mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
    session = df.loc[mask]
    if session.empty:
        return None
    return {
        "name": name,
        "start": session.index[0].isoformat(),
        "end": session.index[-1].isoformat(),
        "high": float(session["high"].max()),
        "low": float(session["low"].min()),
        "open": float(session["open"].iloc[0]),
        "close": float(session["close"].iloc[-1]),
    }

def compute_features(df: pd.DataFrame, depth: Dict[str, Any]):
    # ret5: last close vs close 5 bars ago
    if len(df) < 6:
        ret5 = float("nan")
    else:
        ret5 = (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6]

    # volatility: rolling std of returns (10 bars)
    returns = df["close"].pct_change()
    vol = float(returns.rolling(10).std().iloc[-1])

    # order book imbalance
    bids = depth.get("bids", [])[:10]
    asks = depth.get("asks", [])[:10]
    try:
        bSum = sum(float(p) * float(q) for p, q in bids)
        aSum = sum(float(p) * float(q) for p, q in asks)
        obImbalance = (bSum - aSum) / (bSum + aSum + 1e-9)
        spread = float(asks[0][0]) - float(bids[0][0])
    except Exception:
        obImbalance = float("nan")
        spread = float("nan")

    return ret5, vol, obImbalance, spread

def check_macro_block() -> bool:
    """
    Check high-impact events within ±30 minutes from ForexFactory RSS.
    """
    try:
        feed = feedparser.parse("https://www.forexfactory.com/calendar/rss")
        now = datetime.now(timezone.utc)
        for entry in feed.entries:
            desc = entry.get("description", "")
            if "Impact: High" in desc:
                pubDate = entry.get("published", "")
                try:
                    event_time = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %z")
                    event_time = event_time.astimezone(timezone.utc)
                    if abs((event_time - now).total_seconds()) <= 30 * 60:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False

# ------------------ Dynamic Range model ------------------

def percenter(top: float, bottom: float, close: float) -> float:
    return round(((close - bottom) / (top - bottom)) * 100, 1) if (top is not None and bottom is not None and top != bottom) else float("nan")

@dataclass
class RangeWindow:
    low: float
    high: float
    eq: float

class DynamicRangeModel:
    def __init__(self, daily_df: pd.DataFrame, intraday_df: pd.DataFrame):
        self.daily = daily_df.copy()
        self.intra = intraday_df.copy()
        self.daily_ranges: Dict[pd.Timestamp, Dict[int, RangeWindow]] = {}

    def compute_daily_ranges(self):
        for w in [20, 40, 60]:
            lows = self.daily["low"].rolling(w).min().shift(1)
            highs = self.daily["high"].rolling(w).max().shift(1)
            eq = (lows + highs) / 2.0
            self.daily[f"low{w}"] = lows
            self.daily[f"high{w}"] = highs
            self.daily[f"eq{w}"] = eq

        for ts, row in self.daily.iterrows():
            self.daily_ranges[ts] = {
                20: RangeWindow(row.get("low20"), row.get("high20"), row.get("eq20")),
                40: RangeWindow(row.get("low40"), row.get("high40"), row.get("eq40")),
                60: RangeWindow(row.get("low60"), row.get("high60"), row.get("eq60")),
            }

    def nearest_daily_ts(self, intra_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = self.daily.index
        prior = idx[idx <= intra_ts]
        return prior.max() if len(prior) else None

    def htf_trigger(self, close: float, rw: RangeWindow) -> Optional[str]:
        if rw is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [rw.low, rw.high, rw.eq]):
            return None
        if close > rw.high:
            return "short"  # mean reversion back inside premium edge
        if close < rw.low:
            return "long"   # mean reversion back inside discount edge
        return None

    def ltf_bias(self, close: float, rw: RangeWindow) -> Optional[str]:
        if rw is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [rw.low, rw.high, rw.eq]):
            return None
        if close > rw.eq and close < rw.high:
            return "short-bias"
        if close < rw.eq and close > rw.low:
            return "long-bias"
        if abs(close - rw.eq) <= 1e-8:
            return "neutral-eq"
        return None

    @dataclass
    class CountdownState:
        direction: str        # "long" or "short"
        edge: float           # low40 for long, high40 for short
        start_ts: pd.Timestamp
        fired: bool = False
        half_size_fired: bool = False

    def simulate_htf(self, use_window: int = 40, sl_pct: float = 0.006) -> List[Dict[str, Any]]:
        self.compute_daily_ranges()
        positions: List[Dict[str, Any]] = []
        countdown: Optional[DynamicRangeModel.CountdownState] = None

        def range_size(rw: RangeWindow) -> Optional[float]:
            if rw is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [rw.low, rw.high]):
                return None
            return rw.high - rw.low

        for ts, row in self.intra.iterrows():
            daily_ts = self.nearest_daily_ts(ts)
            if daily_ts is None:
                continue
            rw = self.daily_ranges.get(daily_ts, {}).get(use_window)
            if rw is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [rw.low, rw.high, rw.eq]):
                continue

            close = float(row["close"])

            if countdown is None:
                intent = self.htf_trigger(close, rw)
                if intent == "short":
                    countdown = DynamicRangeModel.CountdownState(direction="short", edge=rw.high, start_ts=ts)
                elif intent == "long":
                    countdown = DynamicRangeModel.CountdownState(direction="long", edge=rw.low, start_ts=ts)

            if countdown:
                elapsed = (ts - countdown.start_ts).total_seconds() / 60.0
                returned_inside = (close <= rw.high) if countdown.direction == "short" else (close >= rw.low)

                if not countdown.fired and elapsed >= 5.0:
                    if returned_inside:
                        entry = close
                        rng = range_size(rw)
                        if not rng or rng <= 0:
                            countdown = None
                            continue
                        t1 = entry + (0.20 * rng) * (-1 if countdown.direction == "short" else 1)
                        t2 = rw.eq
                        sl = entry * (1 + sl_pct) if countdown.direction == "short" else entry * (1 - sl_pct)
                        positions.append({
                            "time": ts.isoformat(),
                            "side": countdown.direction,
                            "entry": entry,
                            "sl": sl,
                            "t1": t1,
                            "t2": t2,
                            "size": 1.0,
                            "note": "HTF full size entry after 5m return inside"
                        })
                        countdown.fired = True

                if countdown.fired is False and not countdown.half_size_fired and elapsed > 5.0 and elapsed <= 17.0 and returned_inside:
                    entry = close
                    rng = range_size(rw)
                    if not rng or rng <= 0:
                        countdown = None
                        continue
                    t1 = entry + (0.20 * rng) * (-1 if countdown.direction == "short" else 1)
                    t2 = rw.eq
                    sl = entry * (1 + sl_pct) if countdown.direction == "short" else entry * (1 - sl_pct)
                    positions.append({
                        "time": ts.isoformat(),
                        "side": countdown.direction,
                        "entry": entry,
                        "sl": sl,
                        "t1": t1,
                        "t2": t2,
                        "size": 0.5,
                        "note": "HTF half size entry within next 12m"
                    })
                    countdown.half_size_fired = True

                if elapsed > 17.0 and not (countdown.fired or countdown.half_size_fired):
                    countdown = None

                if countdown and (countdown.fired and countdown.half_size_fired):
                    countdown = None

        return positions

# ------------------ FastAPI endpoints ------------------

@app.get("/health")
def health():
    try:
        intra = weex_get_candles(SYMBOL, INTRA_PERIOD, 200)
        return {"status": "ok", "latest": intra.index[-1].isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"health error: {e}")

@app.get("/features")
def features():
    try:
        df = weex_get_candles(SYMBOL, INTRA_PERIOD, 1000)
        depth = weex_get_depth(SYMBOL, 20)
        ret5, volatility, obImbalance, spread = compute_features(df, depth)

        sessions = [
            compute_session(df, 2, 5, "LDN"),
            compute_session(df, 7, 12, "HUNT"),
            compute_session(df, 20, 24, "ASIA"),
            compute_session(df, 9, 11, "NY"),
        ]
        sessions = [s for s in sessions if s]

        return {
            "symbol": SYMBOL,
            "ret5": ret5,
            "volatility": volatility,
            "obImbalance": obImbalance,
            "spread": spread,
            "macro_block": check_macro_block(),
            "sessions": sessions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"features error: {e}")



def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
  message = timestamp + method.upper() + request_path + query_string
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def get_usdt_balance() -> float:
    """
    Fetch account assets and return available USDT balance.
    """
    path = "/capi/v2/account/assets"
    ts = str(int(time.time() * 1000))

    timestamp = str(int(time.time() * 1000))

    signature = generate_signature_get(secret_key, timestamp,"GET", path, "")
 
    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": access_passphrase,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }
    url = f"{CONTRACT_BASE}{path}"
    resp = requests.get(url, headers=headers, timeout=10)
    
    resp.raise_for_status()
    data = resp.json()

    for item in data:  # because data is already a list
        if item.get("coinName") == "USDT":
            return float(item["available"])

    return 0.0




def fetch_red_folder_events():
    """
    Scrape ForexFactory calendar for high-impact USD events.
    Returns a list of datetime objects in NY timezone.
    """
    url = "https://www.forexfactory.com/calendar"
    #resp = requests.get(url, timeout=15)
    resp = requests.get( "https://www.forexfactory.com/calendar", headers={"User-Agent": "Mozilla/5.0"}, timeout=15 )
    #print(resp.text)
   # resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ny_tz = pytz.timezone("America/New_York")
    events = []

    # Example parsing logic (depends on actual HTML structure)
    for row in soup.select("tr.calendar__row"):
        currency = row.select_one(".calendar__currency").get_text(strip=True)
        impact = row.select_one(".calendar__impact").get("title", "")
        time_str = row.select_one(".calendar__time").get_text(strip=True)

        if currency == "USD" and "High" in impact:
            # Parse time string into datetime
            try:
                event_time = datetime.strptime(time_str, "%I:%M%p").time()
                today = datetime.now(ny_tz).date()
                event_dt = ny_tz.localize(datetime.combine(today, event_time))
                events.append(event_dt)
            except Exception:
                continue
    return events

def is_ny_hunt_session() -> bool:
    """
    Returns True if current time is within NY Hunt session (8:30–11:30 AM New York time).
    """
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz).time()

    start = dtime(8, 30)
    end = dtime(11, 30)

    return start <= now_ny <= end


def generate_signature(secret_key, timestamp, method, request_path, query_string, body):
  message = timestamp + method.upper() + request_path + query_string + str(body)
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_post(api_key, secret_key, access_passphrase, method, request_path, query_string, body):
  timestamp = str(int(time.time() * 1000))
  body = json.dumps(body)
  signature = generate_signature(secret_key, timestamp, method, request_path, query_string, body)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "POST":
    response = requests.post(url + request_path, headers=headers, data=body)
  return response


def weex_get_ticker(symbol: str) -> Dict[str, Any]:
    # Market ticker endpoint (adjust path if your env differs)
    path = f"/api/v2/market/ticker?symbol={symbol}"
    url = f"{SPOT_BASE}{path}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data)


def normalize_size(raw_size: float, step: float = 0.0001) -> float:
    # Floor to nearest step
    return math.floor(raw_size / step) * step






def placeOrder(symbol, decision):
    print(decision)
    side = decision["decision"]
    decision["amount"] = 10
    amount_usdt = float(decision["amount"])
    balance = get_usdt_balance()
  #  print(str(balance)+","+str(amount_usdt))

    # Risk cap: max 2% intraday, 3% pivotal
    max_risk_pct = 0.02 if decision.get("setup") == "intraday" else 0.03
    max_amount = balance * max_risk_pct
    safe_amount = min(amount_usdt, max_amount)

    ticker = weex_get_ticker(symbol)
    print(ticker)
    last_price = float(ticker["lastPrice"])
    size = round(safe_amount / last_price, 6)
    safe_size = normalize_size(size, 0.0001)  # → 0.0001

    body = {
        "symbol": "cmt_btcusdt",
        "client_oid": str(int(time.time()*1000)),
        "size": str(safe_size),
        "type": "1" if side == "buy" else "2",
        "order_type": "1",  # market
        "match_price": "0",
        "price": last_price
    }
    print(body)
    result = send_request_post(api_key, secret_key, access_passphrase,
                             "POST", "/capi/v2/order/placeOrder", "", body).json()
    print(result)
    return result


def is_red_folder_window(events=None) -> bool:
    """
    Returns True if current NY time is within ±30 minutes of any Red Folder event.
    """
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)

    if events is None:
        events = fetch_red_folder_events()

    for event in events:
        if abs((now_ny - event).total_seconds()) <= 30 * 60:
            return True
    return False


@app.get("/analyze")
def analyze_and_trade():
    #print(XAI_API_KEY)
    if not XAI_API_KEY:

        raise HTTPException(status_code=500, detail="XAI_API_KEY not set")

#try:
    # --- Data ---
    daily = weex_get_candles(SYMBOL, DAILY_PERIOD, CANDLE_LIMIT_DAILY)
    intra = weex_get_candles(SYMBOL, INTRA_PERIOD, CANDLE_LIMIT_INTRA)
    depth = weex_get_depth(SYMBOL, 20)

    ret5, volatility, obImbalance, spread = compute_features(intra, depth)

    # --- Dynamic range model ---
    drm = DynamicRangeModel(daily_df=daily, intraday_df=intra)
    trades = drm.simulate_htf(use_window=40, sl_pct=0.006)

    # --- Latest EQ and range percent ---
    eq40 = float(daily["eq40"].dropna().iloc[-1]) if "eq40" in daily and daily["eq40"].notna().any() else None
    high40 = float(daily["high40"].dropna().iloc[-1]) if "high40" in daily and daily["high40"].notna().any() else None
    low40 = float(daily["low40"].dropna().iloc[-1]) if "low40" in daily and daily["low40"].notna().any() else None
    last_close = float(intra["close"].iloc[-1]) if len(intra) else None
    percent40 = percenter(high40, low40, last_close) if last_close is not None else None

    # --- Build context ---
    payload_context = {
        "symbol": SYMBOL,
        "features": {
            "ret5": ret5,
            "volatility": volatility,
            "obImbalance": obImbalance,
            "spread": spread,
            "macro_block": check_macro_block(),
        },
        "dynamic_range": {
            "eq40": eq40,
            "high40": high40,
            "low40": low40,
            "percent40": percent40,
        },
        "htf_candidates": trades[-5:] if trades else [],
    }
    daily = weex_get_candles(SYMBOL, DAILY_PERIOD, CANDLE_LIMIT_DAILY)
    intra = weex_get_candles(SYMBOL, INTRA_PERIOD, CANDLE_LIMIT_INTRA)

    htf_zones = detect_htf_stops(daily)
    ltf_zones = detect_ltf_stops(intra)
    zones = rank_liquidity_zones(htf_zones, ltf_zones)

    last_close = float(intra["close"].iloc[-1]) if len(intra) else None
    opens = time_based_opens(intra.index[-1]) if len(intra) else {}
    bias = bias_from_opens(last_close, opens) if last_close else None

    


    payload_context["liquidity"] = {
    "bias": bias,
    "zones": [z.__dict__ for z in zones],
    "opens": opens,
}

    payload_context["account"] = {
        "usdt_balance": get_usdt_balance()
        }

    user_prompt = ( "You are a trading signal analyst. Return strict JSON only.\n" "Schema: {decision: 'buy'|'sell'|'hold', confidence: 0..1, rationale: string, amount: float}.\n" "Rules:\n" "- decision must be 'buy', 'sell', or 'hold'.\n" "- amount is the USDT notional to trade, based on account.usdt_balance and risk logic.\n" "- If macro_block is true, prefer 'hold'.\n" "- Only trade during Hunt session (NY AM window).\n" "- Align with HTF bias: below True Daily Open = long only; above True Daily Open = short only.\n" "- Use liquidity zones: BigStops > Stops, HTF > LTF.\n" "- Confidence is a float between 0 and 1.\n" "- rationale must explain why the decision was made (HTF/LTF sweep, bias, liquidity, session).\n\n" "Context:\n" f"{json.dumps(payload_context, ensure_ascii=False)}" )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}",
    }
    body = {
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": "Return only JSON. No prose."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }

    resp = requests.post(XAI_API_URL, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    start_marker = '"content":"'
    start_idx = resp.text.find(start_marker)
    if start_idx == -1:
        raise ValueError("content field not found")

    # Move past the marker
    start_idx += len(start_marker)

    # Find the closing quote for content (before ,"refusal")
    end_idx = resp.text.find('"refusal"', start_idx)
    if end_idx == -1:
        raise ValueError("end of content not found")

    # Step back to the last quote before "refusal"
    end_idx = resp.text.rfind('"', start_idx, end_idx)

    content = resp.text[start_idx:end_idx]

    content =  content.replace('\\"', '"')
    decision = json.loads(content)
    placeOrder(SYMBOL, decision)
    if not is_ny_hunt_session() or is_red_folder_window():
        print("Framework filter: HOLD — outside Hunt session or Red Folder window")
    else:
        placeOrder(SYMBOL, decision)

    return {"data":decision}


async def interval_runner():
    while True:
        try:
            async with runner_lock:
                resp = analyze_and_trade()
             #   logger.info(f"interval_run: {resp}")
        except Exception as e:
            print(e)
           # logger.exception("Interval run failed")
        await asyncio.sleep(INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    # Optionally pre-populate RED_FOLDER_EVENTS here
    # RED_FOLDER_EVENTS.append(pytz.timezone("America/New_York").localize(datetime(2026, 1, 7, 8, 30)))
    asyncio.create_task(interval_runner())


#except HTTPException:
 #   raise
#except Exception as e:
 #   raise HTTPException(status_code=500, detail=f"analyze error: {e}")
