import torch
import time
from torch.utils.data.dataset import Subset

from model import *
from main import model, criterion, optimizer, scheduler
from config import config
from training_helpers import *

device = config["model"]["device"]


def training_mode1(input, index, input_length, label_len, forecast_window):
    source_begin = index
    source_end = source_begin + input_length
    target_begin = source_end - label_len
    target_end = target_begin + label_len + forecast_window

    source = input[source_begin:source_end, :].unsqueeze(-1).float().to(device)
    target = input[target_begin:target_end].unsqueeze(-1).float().to(device)
    placeholder = torch.zeros(forecast_window, batch_size, 1).float().to(device)
    inp_target = torch.cat((source[label_len:, :, :], placeholder), dim=0)

    prediction = model(source, inp_target)  # , src_mask, tgt_mask)

    prediction = prediction[-forecast_window:, :, :]
    target = target[-forecast_window:, :, :]
    return criterion(prediction, target)


def training_mode2(input, index, input_length, label_length, window_size, step):
    assert window_size >= label_length, "Label length to big!"
    source_begin = index
    source_end = source_begin + input_length
    target_in_begin = source_end - label_length
    target_in_end = target_in_begin + window_size
    target_out_begin = target_in_begin + step
    target_out_end = target_in_end + step
    source = input[source_begin:source_end, :].unsqueeze(-1).float().to(device)
    target_in = input[target_in_begin:target_in_end].unsqueeze(-1).float().to(device)
    target = input[target_out_begin:target_out_end].unsqueeze(-1).float().to(device)
    prediction = model(source, target_in)
    calculate_loss_over_all_values = False
    return (
        criterion(prediction, target)
        if calculate_loss_over_all_values
        else criterion(
            prediction[-(window_size - label_length) :, :, :],
            target[-(window_size - label_length) :, :, :],
        )
    )


def training_mode3(input, index, input_length, label_length, window_size, step):
    source_begin = 0
    source_end = source_begin + input_length
    target_in_begin = source_end - label_length + index
    target_in_end = target_in_begin + window_size
    target_out_begin = target_in_begin + step
    target_out_end = target_in_end + step

    source = input[source_begin:source_end, :].unsqueeze(-1).float().to(device)
    target_in = input[target_in_begin:target_in_end].unsqueeze(-1).float().to(device)
    target = input[target_out_begin:target_out_end].unsqueeze(-1).float().to(device)

    prediction = model(source, target_in)
    return criterion(prediction, target)


# @title RUN EPOCH
def run_epoch(
    model: Transformer,
    dataloader,
    epoch: int,
    input_length: int,
    forecast_window: int,
    label_length: int,
    training_mode: int,
    window_size=None,
    step=None,
    is_training=False,
):
    start_time = time.time()
    epoch_loss = 0
    if is_training:
        model.train()
        for idx, _input in enumerate(dataloader):
            """
            SHAPE OF _input : [BATCH, SEQUENCE_LENGTH]
            NEEDED FOR MODEL: [SEQUENCE_LENGTH, BATCH, FEATURE=1]
            MODES FROM:
            https://arxiv.org/pdf/2204.11115.pdf
            https://arxiv.org/pdf/2205.01138.pdf
            https://arxiv.org/pdf/2012.07436.pdf
            https://arxiv.org/pdf/2001.08317.pdf
            """

            idx += 1
            _input = _input.permute(1, 0)
            batch_size = _input.size(1)
            if training_mode == 1:
                """
                WHOLE PREDICTION IN ONE FORWARD PASS, WITH PLACEHOLDER
                INPUT ENCODER: [INPUT_LENGTH, BATCH_SIZE, 1]
                INPUT DECODER: [LAST FIVE INPUT_LENGTH+FORECSAT_WINDOW-LABEL_LEN*0, BATCH_SIZE,1] 
                OUTPUT DECODER: [FORECAST_WINDOW, BATCH_SIZE, 1]
                """

                for index in range(len(_input) - input_length - forecast_window + 1):
                    optimizer.zero_grad()
                    loss = training_mode1(
                        input=_input,
                        index=index,
                        input_length=input_length,
                        label_length=label_length,
                        forecast_window=forecast_window,
                    )

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item() / batch_size
            elif training_mode == 2:
                """
                SEMI - AUTOREGRESSIVE (LIKE IN 1 BUT WITH TARGET INPUT INSTEAD OF PLACEHOLDER)
                INPUT ENCODER: INPUT -> INPUT +STEP -> ...
                INPUT DECODER: [LABEL(INPUT[-label_len:..]), TARGET(SHIFTED BY LABEL_LEN)]
                OUTPUT DECODER: TARGET
                """

                for index in range(
                    len(_input) - input_length - window_size - step + label_length + 1
                ):
                    optimizer.zero_grad()
                    loss = training_mode2(
                        input=_input,
                        index=index,
                        input_length=input_length,
                        label_length=label_length,
                        window_size=window_size,
                        step=step,
                    )

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item() / batch_size
            elif training_mode == 3:
                """
                LIKE 2 WITHOUT SLIDING INPUT TO ENCODER (BETTER SUITED IF INPUT,TARGET SPAN WHOLE DATA SEQUENCE)
                INPUT ENCODER: INPUT_LENGTH
                INPUT DECODER: LABEL,TARGET(SHIFTED BY LABEL TO RIGHT)
                OUTPUT DECODER: TARGET
                """

                for index in range(
                    len(_input) - input_length - window_size - step + label_length + 1
                ):
                    optimizer.zero_grad()
                    loss = training_mode3(
                        input=_input,
                        index=index,
                        input_length=input_length,
                        label_length=label_length,
                        window_size=window_size,
                        step=step,
                    )

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item() / batch_size
            else:
                print("Training mode not available!")
            log_interval = 1 if len(dataloader) // 5 == 0 else len(dataloader) // 5
            if idx % log_interval == 0 and idx > 0 and is_training:
                cur_loss = epoch_loss / len(dataloader)
                elapsed = time.time() - start_time
                print(
                    """| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | """.format(
                        epoch + 1,
                        idx,
                        len(dataloader),
                        optimizer.param_groups[0]["lr"],
                        elapsed * 1000,
                        cur_loss,
                    )
                )

                start_time = time.time()
    else:
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for idx, _input in enumerate(dataloader):
                """
                SHAPE OF _input : [BATCH, SEQUENCE_LENGTH]
                NEEDED FOR MODEL: [SEQUENCE_LENGTH, BATCH, FEATURE=1]
                """

                idx += 1
                _input = _input.permute(1, 0)
                batch_size = _input.size(1)
                if training_mode == 1:
                    for index in range(
                        len(_input) - input_length - forecast_window + 1
                    ):
                        loss = training_mode1(
                            input=_input,
                            index=index,
                            input_length=input_length,
                            label_length=label_length,
                            forecast_window=forecast_window,
                        )

                        epoch_loss = +loss.detach().item() / batch_size
                elif training_mode == 2:
                    for index in range(
                        (
                            (
                                (((len(_input) - input_length) - window_size) - step)
                                + label_length
                            )
                            + 1
                        )
                    ):
                        loss = training_mode2(
                            input=_input,
                            index=index,
                            input_length=input_length,
                            label_length=label_length,
                            window_size=window_size,
                            step=step,
                        )

                        epoch_loss = +loss.detach().item() / batch_size
                elif training_mode == 3:
                    for index in range(
                        (
                            (
                                (((len(_input) - input_length) - window_size) - step)
                                + label_length
                            )
                            + 1
                        )
                    ):
                        loss = training_mode3(
                            input=_input,
                            index=index,
                            input_length=input_length,
                            label_length=label_length,
                            window_size=window_size,
                            step=step,
                        )

                        epoch_loss += loss.detach().item() / batch_size
    total_epoch_loss = epoch_loss / len(dataloader)
    return total_epoch_loss, optimizer.param_groups[0]["lr"]


def run_training(
    model: Transformer,
    training_mode: int,
    EPOCH: int,
    input_length: int,
    forecast_window: int,
    label_length: int,
    step: int,
    window_size: int,
    batch_size: int,
    train_dataloader: Subset,
    test_dataloader: Subset,
    save_model: bool,
    path_to_save_model: str,
):
    min_train_loss = float("inf")
    best_model = ""
    loss_train_epochs, loss_val_epochs = [], []
    try:
        split_ratio = 0.8
        for epoch in range(EPOCH):
            loss_train, lr_train = run_epoch(
                model=model,
                dataloader=train_dataloader,
                epoch=epoch,
                input_length=input_length,
                forecast_window=forecast_window,
                label_length=label_length,
                training_mode=training_mode,
                window_size=window_size,
                step=step,
                is_training=True,
            )

            loss_val, lr_val = run_epoch(
                model=model,
                dataloader=test_dataloader,
                epoch=epoch,
                input_length=input_length,
                forecast_window=forecast_window,
                label_length=label_length,
                training_mode=training_mode,
                window_size=window_size,
                step=step,
                is_training=False,
            )

            scheduler.step()
            if loss_train < min_train_loss and save_model:
                torch.save(
                    model.state_dict(),
                    f"{path_to_save_model}best_train_{training_mode}_Epoch{epoch}.pth",
                )

                torch.save(
                    optimizer.state_dict(),
                    f"{path_to_save_model}optimizer_{training_mode}_Epoch: {epoch}.pth",
                )

                min_train_loss = loss_train
                best_model = f"best_train_{epoch}.pth"
            loss_train_epochs.append(loss_train)
            loss_val_epochs.append(loss_val)
            log_loss(
                epoch,
                loss_train,
                loss_val,
                lr_train,
                training_mode,
                input_length,
                forecast_window,
                label_length,
                step,
                window_size,
                batch_size,
                path_to_save_model,
                model,
            )

            print(
                "Epoch[{}/{}] | loss train :{:.6f}, test :{:.6f} | lr:{:.6f}".format(
                    epoch + 1, EPOCH, loss_train, loss_val, lr_train
                )
            )

        plot_loss(loss_train_epochs, loss_val_epochs)
    except KeyboardInterrupt:
        plot_loss(loss_train_epochs, loss_val_epochs)
    return best_model

