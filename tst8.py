import time
import hmac
import hashlib
import base64
import requests
import json

api_key = "weex_f67d25e3b8c4d7639e7deb7c558016bb"
secret_key = "29056e6c4da2ea623bdfbf6fb223a48f7d192622e31803e6e64c5ceee3bc2611"
access_passphrase = "weex652694794"

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

def placeOrder():
    request_path = "/capi/v2/order/placeOrder"
    body = {
        "symbol": "cmt_btcusdt",
        "client_oid": "test",
        "size": "0.0001",
        "type": "1",
        "order_type": "0",
        "match_price": "0",
        "price": "1000.0"}
    query_string = ""
    response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    placeOrder()