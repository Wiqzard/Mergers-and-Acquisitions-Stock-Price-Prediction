import torch

config = {
    "alpha_vantage": {
        "key": "YOUR_API_KEY",
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {"window_size": 20, "train_split_size": 0.80},
    "plots": {
        "xticks_interval": 10,  # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "embed_dim": 512,
        "nheads": 16,
        "nlayers": 1,
        "dropout": 0.1,
        "input_size": 100,
        "batch_first": False,
        "d_model": 256,
        "num_encoder_layers": 4,
        "num_decoder_layers": 2,
        "num_heads": 8,
        "dropout_encoder": 0.2,
        "dropout_decoder": 0.2,
        "dim_feedforward_encoder": 2048,
        "dim_feedforward_decoder": 2048,
        "num_predicted_features": 1,
    },
    "training": {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 0,
    },
}
