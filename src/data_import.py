"""
IMPORT KAGGLE MAA DATASET
"""

import datetime
from stringprep import map_table_b2
import pandas as pd
from table import *
import yfinance as yf

columns_maa = {
    "parent_comp": "varchar(255)",
    "acquired_comp": "varchar(255)",
    "bsector": "varchar(255)",
    "country ": "varchar(255)",
    "acquisition_date": "DATE",
}


df1 = pd.read_csv("/data/Kaggle_MAA.csv")
df2 = df1.drop(["Acquisition Price", "Category", "Derived Products"], axis=1)
df_clean = df2
month = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
df_clean = df_clean.replace(
    {"Acquisition Month": month}
)  # DATA FRAME WITHOUT UNECESARY COLUMNS
df_clean = df_clean[
    df_clean["Acquisition Month"] != "-"
]  # DATA FRAME WITHOUT ROWS WHERE AC-MONTH = "-"

years = df_clean.loc[:, "Acquisition Year"]
months = df_clean.loc[:, "Acquisition Month"]  # SERIES OF MONTHS AS INTEGERS
dates = list(
    map(lambda x, y: datetime.datetime(int(x), y, 15), years, months)
)  # LIST OF DATAFRAME DATES

# GET START DATES 30 DAYS PRIOR AND END DATES 60 DAYS AFTER MAA DATE
start_dates = []
end_dates = []
for date in dates:
    start = date - datetime.timedelta(days=30)
    end = date + datetime.timedelta(days=120)
    start_dates.append(start)
    end_dates.append(end)

# APPEND COLUMNS WITH START AND END DATES TO DATAFRAME
df_clean = df_clean.assign(start_dates=start_dates, end_dates=end_dates)
df_clean = df_clean.reset_index()

# Insert_db(columns_maa, list(row)
def insert_data_df() -> None:
    for index, row in df_clean.iterrows():
        a = list(row)
        values = [a[2], a[5], a[6], a[7], datetime.datetime(int(a[3]), int(a[4]), 15)]
        maa_table = Table("maa_tech", columns_maa)
        maa_table.create_table()
        maa_table.insert_db(columns_maa, values)
        print("SUCK")


tickers = {
    "microsoft": "MSFT",
    "google": "GOOGL",
    "ibm": "IBM",
    "hp": "HP",
    "apple": "AAPL",
    "amazon": "AMZN",
    "facebook": "META",
    "twitter": "TWTR",
    "ebay": "EBAY",
    "adobe": "ADBE",
    "citrix": "CTXS",
    "redhat": "RHT",
    "blackberry": "BB",
    "disney": "DIS",
}


"""
CREATE TABLE WITH TICKER. START_DATA, END_DATE, DATA_POINTS
"""
columns = {"ticker": "varchar(255)", "start_date": "date", "end_date": "date"}
# for i in range (0,151):
#   columns.update({"%id"%i:"float(24)"})
columns_stock_data = {
    "ticker": "varchar(255)",
    "start_date": "date",
    "end_date": "date",
}
for i in range(151):
    columns_stock_data["%id" % i] = "float(24)"
#
# stock_table.drop_table()
stock_table = Table("stock_table", columns_stock_data)
stock_table.create_table()

# insert_stock_data()
# stock_table.drop_table()

# create_stock_data_table()
# Table.check_tables()
# stock_table.show_table_entries()


company_names, start_dates, end_dates = (
    list(df_clean.loc[:, "Parent Company"]),
    list(df_clean.loc[:, "start_dates"]),
    list(df_clean.loc[:, "end_dates"]),
)


"""
INSERT DATA INTO TABLE
"""
# RETRIEVE SERIES OF STOCK DATA
def get_stock_data(
    company_name: str, start: datetime.datetime, end: datetime.datetime
) -> list[float]:
    ticker = tickers.get(company_name.lower())
    data = yf.download(ticker, start=start, end=end, interval="1d").iloc[:, 1]
    # FILL IN MISSING DATES WITH NULL VALUE
    idx = pd.date_range(start, end)
    data.index = pd.DatetimeIndex(data.index)
    data = data.reindex(idx, fill_value="NULL")
    return data  # MAYBE CHANGE TO LIST FOR CONTINUITY


# list(get_stock_data(company_names[2], start_dates[2], end_dates[2]))

columns_stock_data = {
    "ticker": "varchar(255)",
    "start_date": "date",
    "end_date": "date",
}
for i in range(151):
    columns_stock_data["%id" % i] = "float(24)"
# print(columns_stock_data)
def insert_stock_data() -> None:
    for i in range(len(df_clean)):
        fin_data = list(get_stock_data(company_names[i], start_dates[i], end_dates[i]))
        values = [company_names[i], start_dates[i], end_dates[i], *fin_data]
        stock_table.insert_db(columns_stock_data, values)


# insert_stock_data()
# STOCK DATA THAT IS MISSING FOR START AND END DATE IS AUTOMATICALLY DISMISSED

