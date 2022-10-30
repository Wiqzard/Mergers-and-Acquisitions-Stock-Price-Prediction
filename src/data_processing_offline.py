from csv import excel_tab
import pandas as pd
import datetime
import time
import yfinance as yf
from typing import Tuple, List


def nonsene() -> Tuple[pd.DataFrame, List, List, List]:
    df = pd.read_csv("data/Kaggle_MAA.csv")
    df = df.drop(
        [
            "Acquisition Price",
            "Country",
            "Acquired Company",
            "Business",
            "Category",
            "Derived Products",
        ],
        axis=1,
    )
    df = df.set_index("ID")
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
    df = df.replace({"Acquisition Month": month})
    df = df[df["Acquisition Month"] != "-"]

    # DATE STUFF
    years = df.loc[:, "Acquisition Year"]
    months = df.loc[:, "Acquisition Month"]  # SERIES OF MONTHS AS INTEGERS
    dates = list(map(lambda x, y: datetime.datetime(int(x), y, 15), years, months))

    # GET START DATES 30 DAYS PRIOR AND END DATES 60 DAYS AFTER MAA DATE
    start_dates = []
    end_dates = []
    for date in dates:
        start = date - datetime.timedelta(days=100)
        end = date + datetime.timedelta(days=100)
        start_dates.append(start)
        end_dates.append(end)
    # APPEND COLUMNS WITH START AND END DATES TO DATAFRAME
    df = df.assign(start_dates=start_dates, end_dates=end_dates)
    df = df.drop(["Acquisition Month", "Acquisition Year"], axis=1)
    company_names, start_dates, end_dates = (
        list(df.loc[:, "Parent Company"]),
        list(df.loc[:, "start_dates"]),
        list(df.loc[:, "end_dates"]),
    )
    return df, company_names, start_dates, end_dates


# @title NO DATABASE GET STOCK DATA


def get_stock_data(
    company_name: str, start: datetime.datetime, end: datetime.datetime, tickers: dict
) -> pd.DataFrame:
    ticker = tickers.get(company_name.lower())
    return yf.download(ticker, start=start, end=end, interval="1d").iloc[:, 3]


def get_stock_data2(
    company_name: str, start: datetime.datetime, end: datetime.datetime, tickers: dict
) -> pd.DataFrame:
    ticker = tickers.get(company_name.lower())
    try:
        data = yf.download(ticker, start=start, end=end, interval="1d").iloc[
            :, 3
        ]  # Close Price
    except Exception:
        print("Smth went wrong!")
    # FILL IN MISSING DATES WITH NULL VALUE
    idx = pd.date_range(start, end)
    data.index = pd.DatetimeIndex(data.index)
    data = data.reindex(idx, fill_value=None)

    data_frame = data.to_frame().reset_index()
    data_frame = data_frame.iloc[:-1]

    data_frame = data_frame.dropna(axis=1, how="all")
    while data_frame.isnull().values.any():
        data_frame = data_frame.fillna(method="ffill")
        data_frame = data_frame.fillna(method="bfill")

    data_frame = data_frame.rename({"index": "Date", "Close": "Price"}, axis="columns")
    data_frame["Date"] = pd.to_datetime(data_frame["Date"], format="%y-%m-%d")
    return data_frame


def insert_stock_data(df_inp) -> None:  # sourcery skip: do-not-use-bare-except
    data = []
    df, company_names, start_dates, end_dates = nonsene()
    for i in range(5):  # len(df_inp)
        try:
            yfaa = get_stock_data(
                company_names[i], start_dates[i], end_dates[i]
            )  # .index.tz_localize(None)#convert(None)
            data.append(yfaa.values)
            time.sleep(0)
            df = pd.concat([pd.Series(x) for x in data], axis=1)
        except Exception:
            continue
    return df


def insert_stock_data_vertical(df_inp) -> pd.DataFrame:
    data = pd.DataFrame(columns=["Date", "Price"])
    lengths = []
    df, company_names, start_dates, end_dates = nonsene()
    for i in range(1300):  # len(df_inp)):#len(df_inp)
        yfaa = get_stock_data2(
            company_names[i], start_dates[i], end_dates[i]
        )  # .index.tz_localize(None)#convert(None)
        data = pd.concat([data, yfaa], ignore_index=True, axis=0)
        lengths.append(yfaa.shape[0])

    return data, lengths


# data_complete = insert_stock_data(df)#.head()


csv_address = "data/data.csv"

data_comp = pd.read_csv(csv_address, sep="\t")
print(data_comp.isnull().values.any())
