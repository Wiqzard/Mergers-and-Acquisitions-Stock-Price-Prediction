import pymysql
from joblib import dump
import torch
from torch import Tensor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset


from table import *
from config import config

"""
FETCH DATA FROM SQL DATABASE
"""

# IF TICKER is NONE -> ALL TICKERS
def fetch_data(table_num: int, company: str) -> pd.DataFrame:
    try:
        db = pymysql.connect(
            host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
            user="admin",
            password="123456789",
        )

    except Exception as e:
        print("Can not connect to database")

    columns = {"ticker": "varchar(255)", "start_date": "date", "end_date": "date"}
    columns_price = "0d"

    if table_num == 1 and company is None:
        table_name = "stock_table"
        for i in range(1, 150):
            columns_price += f", {i}d"
        sql = "SELECT DISTINCT {columns} FROM {table_name}".format(
            columns=columns_price, table_name=table_name
        )
    elif table_num == 2:
        table_name = "stock_table2"
        for i in range(1, 450):
            columns_price += f", {i}d"
        if company is None:
            sql = "SELECT DISTINCT {columns} FROM {table_name}".format(
                columns=columns_price, table_name=table_name
            )
        else:
            sql = """SELECT DISTINCT {columns} FROM {table_name}
                WHERE ticker = '{company}';""".format(
                columns=columns_price, table_name=table_name, company=company
            )

    cursor = db.cursor()
    cursor.execute("USE {db_name}".format(db_name=Table.db_name))

    sql_query = pd.read_sql_query(sql, db)
    return pd.DataFrame(sql_query)


def processed_data(table_num: int, company: str) -> Tensor:
    df = fetch_data(table_num, company)
    # FILL ZERO VALUES WITH VALUE BEFORE AND AFTER
    df_t = df.transpose()  # [150, TOTAL_SEQUENCES]
    df_t = df_t.replace(to_replace=0, method="ffill")
    df_t = df_t.replace(to_replace=0, method="bfill")
    # REMOVE RESIDUAL SEQUENCES CONTAINING 0'S
    # DELETE RESIDIUAL DATA
    df_t = df_t.loc[:, (df_t != 0).any(axis=0)]  # TOTAL SEQUENCES: 1396 -> 1299
    return torch.tensor(
        df_t.values
    )  # CONVERT TO TENSOR | [SEQUECE_LENGTH, TOTAL_SEQUENCES]


"""
CREATE CUSTOM DATASET
"""


class StockPriceDataset(Dataset):
    def __init__(
        self, dataset: Tensor, device
    ) -> None:  # , training_length:int, forecast_window:int

        # self.T = training_length #var
        # self.S = forecast_window #120
        self.data = dataset
        # self.transform = MinMaxScaler(feature_range=(0, 1))
        self.transform = StandardScaler()
        self.device = device

    def __len__(self) -> int:

        return self.data.size()[1]

    def __getitem__(self, idx: int) -> Tensor:

        _input = self.data[:, idx]  # THE idx'TH  SEQUENCE
        scaler = self.transform
        scaler.fit(_input.unsqueeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, "save/scalar_item.joblib")
        _input = (
            torch.tensor(scaler.transform(_input.unsqueeze(-1))).squeeze(-1).to(device)
        )
        return _input


"""
PLAIN INTO TRAIN AND TEST DATA
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_test_ratio = 0.8
# [FULL_SEQUENCE, NUM_SEQUENCES]
# 30PRI 120 AFTER
total_data_30_120 = processed_data(1, None)
full_stock_data_30_120 = StockPriceDataset(total_data_30_120, device)
train_size = int(train_test_ratio * len(full_stock_data_30_120))
test_size = len(full_stock_data_30_120) - train_size
train_dataset_30_120, test_dataset_30_120 = torch.utils.data.random_split(
    full_stock_data_30_120, [train_size, test_size]
)

# 30 PRI 30 AFTER

total_data_30_30 = total_data_30_120[:60, :]

full_stock_data_30_30 = StockPriceDataset(total_data_30_30, device)
train_size = int(train_test_ratio * len(full_stock_data_30_30))
test_size = len(full_stock_data_30_30) - train_size
train_dataset_30_30, test_dataset_30_30 = torch.utils.data.random_split(
    full_stock_data_30_30, [train_size, test_size]
)


# 300PRI 150 AFTER
total_data_300_150 = processed_data(2, None)
full_stock_data_300_150 = StockPriceDataset(total_data_300_150, device)
train_size2 = int(train_test_ratio * len(full_stock_data_300_150))
test_size2 = len(full_stock_data_300_150) - train_size2
train_dataset_300_150, test_dataset_300_150 = torch.utils.data.random_split(
    full_stock_data_300_150, [train_size2, test_size2]
)

# 100 PRI 100 AFTER
stock_data_100_100 = StockPriceDataset(total_data_300_150[200:400, :], device)
train_size2 = int(train_test_ratio * len(stock_data_100_100))
test_size2 = len(stock_data_100_100) - train_size2
train_dataset_100_100, test_dataset_100_100 = torch.utils.data.random_split(
    stock_data_100_100, [train_size2, test_size2]
)

# 150 PRI 150 AFTER
stock_data_150_150 = StockPriceDataset(total_data_300_150[150:450, :], device)
train_dataset_150_150, test_dataset_150_150 = torch.utils.data.random_split(
    stock_data_150_150, [train_size2, test_size2]
)

# GOOGLE ONLY  300PRI 150 AFTER
total_data_google = processed_data(2, "Google")
google_dataset_300_150 = StockPriceDataset(total_data_google, device)
train_size_google = int(train_test_ratio * len(google_dataset_300_150))
test_size_google = len(google_dataset_300_150) - train_size_google
(
    train_dataset_google_300_150,
    test_dataset_google_300_150,
) = torch.utils.data.random_split(
    google_dataset_300_150, [train_size_google, test_size_google]
)
# GOOGLE ONLY 150 PRI 150 AFTER
google_dataset_150_150 = StockPriceDataset(total_data_google[200:400, :], device)
(
    train_dataset_google_150_150,
    test_dataset_google_150_150,
) = torch.utils.data.random_split(
    google_dataset_150_150, [train_size_google, test_size_google]
)
# SMALL MODEL BUILDING DATASETS (25 SAMPLE SEQUENCES)
model_building_dataset_30_120 = StockPriceDataset(total_data_30_120[:, :25], device)
model_building_dataset_300_150 = StockPriceDataset(total_data_300_150[:, :25], device)

# MODEL BUILDING DATA
output_window = 100
input_window = 100


def create_sin_sequences(length):
    time1 = np.arange(0, 400, 0.1)
    amplitude = (
        np.sin(time1)
        + np.sin(time1 * 0.05)
        + np.sin(time1 * 0.12) * np.random.normal(-0.2, 0.2, len(time1))
    )
    samples = 2800
    train_data = amplitude[:samples]
    test_data = amplitude[samples:]

    train_sequences = []
    test_sequences = []
    l_train = len(train_data)
    l_test = len(test_data)
    for i in range(l_train - length):
        train_seq = train_data[i : i + length]
        train_sequences.append(train_seq)

    for i in range(l_test - length):
        test_seq = test_data[i : i + length]
        test_sequences.append(test_seq)

    return (
        torch.tensor(train_sequences).permute(1, 0),
        torch.tensor(test_sequences).permute(1, 0),
    )


total_data_sin = create_sin_sequences(200)
train_dataset_sin = StockPriceDataset(total_data_sin[0], device)
test_dataset_sin = StockPriceDataset(total_data_sin[1], device)

total_data_sin_400 = create_sin_sequences(400)
train_dataset_sin_400 = StockPriceDataset(total_data_sin_400[0], device)
test_dataset_sin_400 = StockPriceDataset(total_data_sin_400[1], device)


# @title INITIATE ALL DATALOADER
"""
ALL DATASETS
train_dataset_30_120, test_dataset_30_120
train_dataset_300_150, test_dataset_300_150
train_dataset_100_100, test_dataset_100_100
train_dataset_150_150, test_dataset_150_150
train_dataset_google_300_150, test_dataset_google_300_150
train_dataset_google_150_150, test_dataset_google_150_150
model_building_dataset_30_120, model_building_dataset_300_150
"""

batch_size = config["training"]["batch_size"]
shuffle = config["training"]["shuffle"]
drop_last = config["training"]["drop_last"]
num_workers = config["training"]["num_workers"]

train_dataloader_30_120 = DataLoader(
    train_dataset_30_120,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_30_120 = DataLoader(
    test_dataset_30_120,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_30_30 = DataLoader(
    train_dataset_30_30,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_30_30 = DataLoader(
    test_dataset_30_30,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)


train_dataloader_300_150 = DataLoader(
    train_dataset_300_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_300_150 = DataLoader(
    test_dataset_300_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_100_100 = DataLoader(
    train_dataset_100_100,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_100_100 = DataLoader(
    test_dataset_100_100,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_150_150 = DataLoader(
    train_dataset_150_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_150_150 = DataLoader(
    test_dataset_150_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_google_300_150 = DataLoader(
    train_dataset_google_300_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_google_300_150 = DataLoader(
    test_dataset_google_300_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_google_150_150 = DataLoader(
    train_dataset_google_150_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_google_150_150 = DataLoader(
    test_dataset_google_150_150,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

model_building_dataloader_30_120 = DataLoader(
    model_building_dataset_30_120,
    batch_size=4,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
model_building_dataloader_300_150 = DataLoader(
    model_building_dataset_300_150,
    batch_size=4,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

single_dataloader = DataLoader(
    model_building_dataset_300_150,
    batch_size=1,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_sin = DataLoader(
    train_dataset_sin,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_sin = DataLoader(
    test_dataset_sin,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

train_dataloader_sin_400 = DataLoader(
    train_dataset_sin_400,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)
test_dataloader_sin_400 = DataLoader(
    test_dataset_sin_400,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
)

