import random
import torch
from torch.utils.data.dataset import Subset
from joblib import load
import matplotlib.pyplot as plt
import numpy as np

from main import config
from model import *

device = config["model"]["device"]
# @title INFERENCE2222
def inference2(
    model: Transformer,
    inference_mode: int,
    idx: int,
    input_length: int,
    forecast_window: int,
    label_len: int,
    step: int,
    window_size: int,
    maa_index: int,
    train_datasetz,
    test_datasetz,
    train=True,
) -> Tensor:
    model.eval()
    assert maa_index - input_length >= 0, "Input length to big!"
    if train and idx < len(train_datasetz):
        print("train")
        _input = train_datasetz[idx]
    elif not train and idx < len(test_datasetz):
        print("test")
        _input = test_datasetz[idx]

    with torch.no_grad():
        if inference_mode == 1:
            """
            WHOLE PREDICTION IN ONE FORWARD PASS
        """
            source_begin = maa_index - input_length
            source_end = source_begin + input_length
            target_begin = source_end - label_len
            target_end = target_begin + label_len + forecast_window
            source = (
                _input[source_begin:source_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            target = (
                _input[target_begin:target_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            placeholder = torch.zeros(forecast_window, 1, 1).float().to(device)
            inp_target = torch.cat((source[label_len:, :, :], placeholder), dim=0)
            prediction = model(source, inp_target)
            data = _input[source_begin:target_end]
            prediction = prediction[-forecast_window:, 0, 0]
            print(prediction)

        elif inference_mode == 2:
            """
            MAKES SENSE FOR STEP not 1
            ENCODER INPUT MOVES ALONG PREDICTION, LABEL CHANGES ALSO
            DECODER INPUTS: LABEL -> LABEL'+1st OUTPUT -> LABEL''+1st+2nd OUTPUT....
            DECODER OUTPUTS: DECODER INPUTS SHIFTED BY STEP
        """
            source_begin = maa_index - input_length
            source_end = source_begin + input_length
            target_in_begin = source_end - label_len
            target_in_end = source_end
            source = (
                _input[source_begin:source_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            target_in = (
                _input[target_in_begin:target_in_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            data = _input[: source_end + forecast_window]
            for _ in range(0, forecast_window, step):
                output = model(source, target_in)
                source = torch.cat((source[step:, :, :], output[-step:, :, :]), dim=0)
                target_in = torch.cat((target_in[:, :, :], output[-step:, :, :]), dim=0)
            prediction = []
            prediction = output[-forecast_window:, 0, 0]
            print(prediction)

        elif inference_mode == 3:
            """
            ENCODER INPUT STAYS SAME
            DECODER INPUTS: LABEL -> LABEL+1st OUTPUT -> LABEL+1st+2nd OUTPUT....
            DECODER OUTPUTS: DECODER INPUTS SHIFTED BY STEP
        """
            source_begin = maa_index - input_length
            source_end = source_begin + input_length
            target_in_begin = source_end - label_len
            target_in_end = source_end
            source = (
                _input[source_begin:source_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            target_in = (
                _input[target_in_begin:target_in_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            data = _input[: source_end + forecast_window]
            for _ in range(0, forecast_window, step):
                output = model(source, target_in)
                target_in = torch.cat((target_in[:, :, :], output[-step:, :, :]), dim=0)
            prediction = torch.cat(
                (
                    _input[source_end].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    output[-forecast_window + 1 :, :, :],
                ),
                dim=0,
            )
            prediction = prediction.view(-1)
            print(prediction)

        elif inference_mode == 4:
            """
            END IS NOT WHOLE OUTPUT BUT EVERY SINGLE STEP APPENDED
            ENCODER INPUT MOVES ALONG PREDICTION, LABEL CHANGES ALSO
            DECODER INPUTS: LABEL -> LABEL'+1st OUTPUT -> LABEL''+1st+2nd OUTPUT....
            DECODER OUTPUTS: DECODER INPUTS SHIFTED BY STEP
        """
            source_begin = maa_index - input_length
            source_end = source_begin + input_length
            target_in_begin = source_end - label_len
            target_in_end = source_end
            source = (
                _input[source_begin:source_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            target_in = (
                _input[target_in_begin:target_in_end]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
                .to(device)
            )
            data = _input[: source_end + forecast_window]
            prediction = []
            for _ in range(0, forecast_window, step):
                output = model(source, target_in)
                source = torch.cat((source[step:, :, :], output[-step:, :, :]), dim=0)
                target_in = torch.cat((target_in[:, :, :], output[-step:, :, :]), dim=0)
                prediction.append(output[-step:, :, :].item())
            prediction = torch.FloatTensor(prediction)
            print(prediction)

        return prediction.to("cpu"), data.to("cpu")


def plot_inference(
    model: Transformer,
    inference_mode: int,
    input_length: int,
    forecast_window: int,
    idx: int,
    label_len: int,
    step: int,
    window_size: int,
    maa_index: int,
    train_dataset: Subset,
    test_dataset: Subset,
    train: bool,
) -> None:

    if idx is None:
        if train:
            idx = random.randint(0, len(train_dataset))
        else:
            idx = random.randint(0, len(test_dataset))

    num_data_points = input_length + forecast_window
    data_indx = list(range(maa_index + forecast_window))

    fig = plt.figure(figsize=(20, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))

    scaler = load("scalar_item.joblib")
    if train:
        prediction, data_price_train = inference2(
            model,
            inference_mode,
            idx,
            input_length,
            forecast_window,
            label_len,
            step,
            window_size,
            maa_index,
            train_dataset,
            test_dataset,
            train,
        )
        # prediction = torch.cat((data_price_train,prediction[1:]),0)

        data_price_train_re = torch.from_numpy(
            scaler.inverse_transform(data_price_train.unsqueeze(-1)).squeeze(-1)
        )

        to_plot_data_y_train_pred = np.zeros(maa_index + forecast_window)
        to_plot_data_y_train_pred[
            maa_index : maa_index + forecast_window
        ] = scaler.inverse_transform(prediction.unsqueeze(-1)).squeeze(-1)
        to_plot_data_y_train_pred = np.where(
            to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred
        )

        plt.plot(
            data_indx,
            data_price_train_re,
            label="Actual prices (train)",
            color=config["plots"]["color_actual"],
        )
        plt.plot(
            data_indx,
            to_plot_data_y_train_pred,
            label="Predicted prices (train)",
            color=config["plots"]["color_pred_train"],
        )

    else:
        prediction, data_price_val = inference2(
            model,
            inference_mode,
            idx,
            input_length,
            forecast_window,
            label_len,
            step,
            window_size,
            maa_index,
            train_dataset,
            test_dataset,
            train,
        )

        data_price_val_re = torch.from_numpy(
            scaler.inverse_transform(data_price_val.unsqueeze(-1)).squeeze(-1)
        )

        to_plot_data_y_val_pred = np.zeros(maa_index + forecast_window)
        to_plot_data_y_val_pred[
            maa_index : maa_index + forecast_window
        ] = scaler.inverse_transform(prediction.unsqueeze(-1)).squeeze(-1)
        to_plot_data_y_val_pred = np.where(
            to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred
        )

        plt.plot(
            data_indx,
            data_price_val_re,
            label="Actual prices (val)",
            color=config["plots"]["color_actual"],
        )
        plt.plot(
            data_indx,
            to_plot_data_y_val_pred,
            label="Predicted prices (val)",
            color=config["plots"]["color_pred_val"],
        )

    plt.title("Compare predicted prices to actual prices")
    xticks = [
        data_indx[i]
        if (
            (
                i % config["plots"]["xticks_interval"] == 0
                and (num_data_points - i) > config["plots"]["xticks_interval"]
            )
            or i == num_data_points - 1
        )
        else None
        for i in range(num_data_points)
    ]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation="vertical")
    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.legend()
    plt.show()

