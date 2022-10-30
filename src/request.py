import pymysql
import requests
import pandas as pd


url_fin_maa = "https://financialmodelingprep.com/api/v4/mergers-acquisitions-rss-feed?page=0&apikey="
# Company Enterprise Value
url_fin_kpi = "https://financialmodelingprep.com/api/v3/enterprise-values/AAPL?apikey="
url_fin_symbol = "https://financialmodelingprep.com/api/v3/available-traded/list?apikey="

# SYMBOL | Name ...
def api_request_symbol() -> list:  # Returns list of dictionaries received from API
    try:
        response = requests.get(url_fin_symbol)
        if response.status_code == 200:
            print("Connection to symbol-API successful")
            return response.json()
    except Exception as e:
        print("maa-API Error")


symbol_data = api_request_symbol()
df_symbol = pd.DataFrame(symbol_data)
# df_symbol.head()
# df_symbol.info()


def api_request_maa() -> list:  # Returns list of dictionaries received from API
    try:
        response = requests.get(url_fin_maa)
        if response.status_code == 200:
            print("Connection to maa-API successful")
            return response.json()
    except Exception as e:
        print("maa-API Error")


def api_request_kpi() -> list:  # Returns list of dictionaries received from API
    try:
        response = requests.get(url_fin_kpi)
        if response.status_code == 200:
            print("Connection to kpi-API successful")
            return response.json()
    except Exception as e:
        print("kpi-API Error")


maa_secData = api_request_maa()
# kpi_data = api_request_kpi()

keys_maa = [
    "companyName",
    "targetedCompanyName",
    "transactionDate",
]  # Keys of interest for dictionary data
keys_kpi = ""
