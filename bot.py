# bot.py
import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

FASTAPI_BASE = os.getenv("FASTAPI_BASE", "http://localhost:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_SHARED_SECRET = os.getenv("BOT_SHARED_SECRET")
ALLOWED_CHAT_ID = int(os.getenv("ALLOWED_CHAT_ID", "0"))  # optional: restrict usage

def api_get(path: str):
    return requests.get(f"{FASTAPI_BASE}{path}", headers={}, timeout=20)

def api_post(path: str, json_body: dict):
    return requests.post(f"{FASTAPI_BASE}{path}", json=json_body, headers={}, timeout=20)

def authorized(update: Update) -> bool:
    return (ALLOWED_CHAT_ID == 0) or (update.effective_chat and update.effective_chat.id == ALLOWED_CHAT_ID)

async def send_text(update: Update, text: str):
    await update.message.reply_text(text[:4000])  # keep within Telegram limits

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update):
       # return await send_text(update, "Access denied.")
    await send_text(update, "Bot ready. Use /status, /analyze, /risk, /max, /auto, /hunt, /trade, /kill")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    try:
        r = api_get("/status")
        await send_text(update, r.text if r.ok else f"Status error: {r.text}")
    except Exception as e:
        await send_text(update, f"Status exception: {e}")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ##if not authorized(update): return await send_text(update, "Access denied.")
    try:
        r = api_get("/analyze")
        await send_text(update, r.text if r.ok else f"Analyze error: {r.text}")
    except Exception as e:
        await send_text(update, f"Analyze exception: {e}")

async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    if len(context.args) != 1:
        return await send_text(update, "Usage: /risk 0.5 (percent of equity)")
    try:
        val = float(context.args[0])
        r = api_post("/config/set", {"risk_percent": val})
        await send_text(update, r.text if r.ok else f"Risk error: {r.text}")
    except Exception as e:
        await send_text(update, f"Risk exception: {e}")

async def maxsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    if len(context.args) != 1:
        return await send_text(update, "Usage: /max 100 (USDT)")
    try:
        val = float(context.args[0])
        r = api_post("/config/set", {"max_trade_usdt": val})
        await send_text(update, r.text if r.ok else f"Max error: {r.text}")
    except Exception as e:
        await send_text(update, f"Max exception: {e}")

async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    if len(context.args) != 1 or context.args[0] not in ("on", "off"):
        return await send_text(update, "Usage: /auto on|off")
    val = context.args[0] == "on"
    r = api_post("/config/set", {"auto_trade_enabled": val})
    await send_text(update, r.text if r.ok else f"Auto error: {r.text}")

async def hunt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    if len(context.args) != 1 or context.args[0] not in ("on", "off"):
        return await send_text(update, "Usage: /hunt on|off")
    val = context.args[0] == "on"
    r = api_post("/config/set", {"hunt_enabled": val})
    await send_text(update, r.text if r.ok else f"Hunt error: {r.text}")

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    # Usage: /trade buy 50  OR  /trade sell 0.002 btc (quantity mode optional)
    if len(context.args) < 2:
        return await send_text(update, "Usage: /trade buy 50 (USDT notional) or /trade sell 0.002 qty")
    side = context.args[0].lower()
    second = context.args[1]
    payload = {"side": side, "orderType": "market"}
    # default: notional in USDT; optional third arg "qty" to send quantity
    if len(context.args) == 3 and context.args[2].lower() == "qty":
        payload["quantity"] = second
    else:
        payload["notional_usdt"] = second
    try:
        r = api_post("/trade/place", payload)
        await send_text(update, r.text if r.ok else f"Trade error: {r.text}")
    except Exception as e:
        await send_text(update, f"Trade exception: {e}")

async def kill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #if not authorized(update): return await send_text(update, "Access denied.")
    try:
        r = api_post("/trade/kill", {})
        await send_text(update, r.text if r.ok else f"Kill error: {r.text}")
    except Exception as e:
        await send_text(update, f"Kill exception: {e}")

def main():
    if not BOT_TOKEN or not FASTAPI_BASE or not BOT_SHARED_SECRET:
        raise RuntimeError("Missing BOT_TOKEN, FASTAPI_BASE, or BOT_SHARED_SECRET")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("risk", risk))
    app.add_handler(CommandHandler("max", maxsize))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("hunt", hunt))
    app.add_handler(CommandHandler("trade", trade))
    app.add_handler(CommandHandler("kill", kill))
    app.run_polling()

if __name__ == "__main__":
    main()
