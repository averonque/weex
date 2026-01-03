import time, hmac, hashlib, base64, requests, json

API_KEY = "weex_f67d25e3b8c4d7639e7deb7c558016bb"
API_SECRET = "29056e6c4da2ea623bdfbf6fb223a48f7d192622e31803e6e64c5ceee3bc2611"
PASSPHRASE = "weex652694794"

url = "https://api-spot.weex.com/api/v2/trade/orders"
timestamp = str(int(time.time() * 1000))  # ms timestamp

body = {
    "symbol": "BTCUSDT_SPBL",
    "side": "buy",
    "orderType": "market",
    "force": "normal",
    "quantity": "10",
    "clientOrderId": "5234234234"
}
body_str = json.dumps(body, separators=(',', ':'))

# Signature: HMAC SHA256 of (timestamp + method + requestPath + body)
message = timestamp + "POST" + "/api/v2/trade/orders" + body_str
#signature = hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
signature = base64.b64encode( hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).digest() ).decode()
headers = {
    "ACCESS-KEY": API_KEY,
    "ACCESS-SIGN": signature,
    "ACCESS-PASSPHRASE": PASSPHRASE,
    "ACCESS-TIMESTAMP": timestamp,
    "Content-Type": "application/json",
}   

resp = requests.post(url, headers=headers, data=body_str)
print( resp.status_code)
