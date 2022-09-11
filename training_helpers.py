from torch import Tensor
from model import *
import matplotlib.pyplot as plt
import os


def get_batch(source: Tensor, i_source: int, i_target: int, sequence_length: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size], seq_len
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len , batch_size]
    """
    seq_len = min(sequence_length, len(source) - 1 - i_source)
    data = source[i_source : seq_len + i_source, :]
    target = source[i_target : seq_len + i_target, :]
    return data, target


def log_loss(
    epoch: int,
    train_loss: float,
    loss_val: float,
    training_mode: int,
    input_length: int,
    forecast_window: int,
    label_len: int,
    step: int,
    window_size: int,
    learning_rate: float,
    batch_size: int,
    path_to_save_loss: str,
    model: Transformer,
) -> None:
    num_encoder_layers = model.num_encoder_layers
    num_decoder_layers = model.num_decoder_layers
    dim_model = model.d_model
    file_name = "loss_log.txt"
    path_to_file = path_to_save_loss + file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        if epoch == 0:
            f.write(
                """training_mode={} - input_length={} - forecast_window={}
          label_len={} - step={} - window_size={} 
          batch_size={} - lr={:.3f} -
          num_encoder_layers={} -num_decoder_layers={} - dim_model={} \n""".format(
                    training_mode,
                    input_length,
                    forecast_window,
                    label_len,
                    step,
                    window_size,
                    batch_size,
                    learning_rate,
                    num_encoder_layers,
                    num_decoder_layers,
                    dim_model,
                )
            )

        f.write(
            f"EPOCH {epoch + 1}:   Training loss: {train_loss}    Validation loss {loss_val} \n"
        )

        f.close()


def EMA(values: list, alpha=0.1) -> list:
    ema_values = [values[0]]
    ema_values.extend(
        alpha * item + (1 - alpha) * ema_values[idx]
        for idx, item in enumerate(values[1:])
    )
    return ema_values


def plot_loss(train_loss_list: list, val_loss_list: list):
    EMA_train_loss = EMA(train_loss_list)
    EMA_val_loss = EMA(val_loss_list)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(EMA_train_loss, label="EMA train loss")
    plt.plot(val_loss_list, label="val loss")
    plt.plot(EMA_val_loss, label="EMA val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()
