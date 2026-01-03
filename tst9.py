import time
import hmac
import hashlib
import base64
import requests

api_key = "weex_f67d25e3b8c4d7639e7deb7c558016bb"
secret_key = "29056e6c4da2ea623bdfbf6fb223a48f7d192622e31803e6e64c5ceee3bc2611"
access_passphrase = "weex652694794"


def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
  message = timestamp + method.upper() + request_path + query_string
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_get(api_key, secret_key, access_passphrase, method, request_path, query_string):
  timestamp = str(int(time.time() * 1000))
  signature = generate_signature_get(secret_key, timestamp, method, request_path, query_string)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }

  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string, headers=headers)
  return response

def fills():
    request_path = "/capi/v2/order/fills"
    query_string = "?symbol=cmt_btcusdt&orderId=702171352861246368"
    response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)
if __name__ == '__main__':
    fills()