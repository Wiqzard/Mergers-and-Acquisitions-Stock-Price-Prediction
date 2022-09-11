import torch
import numpy as np
import matplotlib as plt
import torch.optim as optim
import shutil

from model import Transformer
from training_helpers import *
from training import *
from data_processing import *
from data_processing import *
from inference import *


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_sin_data():
    time1 = np.arange(0, 400, 0.1)
    amplitude = (
        np.sin(time1)
        + np.sin(time1 * 0.05)
        + np.sin(time1 * 0.12) * np.random.normal(-0.2, 0.2, len(time1))
    )
    fig, ax = plt.subplots()
    ax.plot(time1, amplitude)
    ax.grid()
    plt.show()


def clean_directory():
    if os.path.exists("save"):
        shutil.rmtree("save")
    os.mkdir("save")


model = Transformer(
    input_size=1,
    batch_first=False,
    d_model=512,
    num_encoder_layers=4,
    num_decoder_layers=2,
    num_heads=8,
    dropout_encoder=0.1,
    dropout_decoder=0.1,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
criterion = torch.nn.MSELoss()
path = "/save/"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_directory()
    """run_training(
        model=model,
        training_mode=2,
        EPOCH=10,
        input_length=30,
        forecast_window=100,
        label_length=1,
        step=1,
        window_size=100,
        batch_size=32,
        train_dataloader=train_dataloader_30_120,
        test_dataloader=test_dataloader_30_120,
        save_model=False,
        path_to_save_model=path,
    )

    plot_inference(
        model=model,
        inference_mode=3,
        input_length=30,
        forecast_window=100,
        idx=None,
        label_len=1,
        maa_index=30,
        step=1,
        window_size=100,
        train_dataset=train_dataset_30_120,
        test_dataset=test_dataset_30_120,
        train=False,
    )"""

    plot_sin_data()


if __name__ == "__main__":
    main()
