import requests

def send_request_get( method, request_path, query_string):
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string)
  return response

def contracts():
    request_path = "/capi/v2/market/contracts"
    query_string = "?symbol=cmt_btcusdt"
    response = send_request_get( "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    contracts()