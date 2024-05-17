import typing as t
from collections.abc import Iterable

import random
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import wandb

from dataset import TimeSeriesDataset, EncoderDecoderDataset, scale, reverse_weather_shift
from transformer import EncoderOnlyTransformer, EncoderDecoderTransformer
from ceemdan import calculate_ceemdan


wandb.login()
wandb.init(project="transformer-heat-pump-forecasting")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

INPUT_DAYS = 2
ENCODER_DECODER = True
EPOCHS = 200
PATIENCE = 100
DATA_PATH = "data/"
TARGET_COLUMN = "Agg Load"
PREDICTION_COLUMN = "Predict"


def get_previous_days(df, day, days):
    df = df.copy()
    if days == 0:
        start_date = datetime.datetime.strptime(day,"%Y-%m-%d")
        end_date = datetime.datetime.strptime(day,"%Y-%m-%d") + datetime.timedelta(days=1)
    else:
        start_date = datetime.datetime.strptime(day,"%Y-%m-%d") - datetime.timedelta(days=days)
        end_date = datetime.datetime.strptime(day,"%Y-%m-%d")
    mask = (df.index.tz_localize(None) >= start_date) & (df.index.tz_localize(None) < end_date)
    return df[mask]


def get_all_days_in_month(month, year):
    target_date = datetime.date(year, month, 1)
    delta = datetime.timedelta(days=1)
    dates = []
    while target_date.month == month:
        dates.append((target_date).strftime('%Y-%m-%d'))
        target_date += delta
    return dates


def analyze_month(df, year, month, args):
    target_df = df.copy()
    target_df[PREDICTION_COLUMN] = 0
    # get all days in month
    all_days = get_all_days_in_month(month,year)
    # get first day in month
    first_day = all_days[0]
    # get last 365 days for train df
    last_365_df = get_previous_days(df,first_day,365)
    # train model w. last_365
    model, outputs_train = train_model(last_365_df, args)
    # forecast every day in month, insert values in df
    for day in tqdm(all_days):
        last_7d_from_day = get_previous_days(df, day, INPUT_DAYS)
        next_day = (datetime.datetime.strptime(day,"%Y-%m-%d") + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        decoder_features = get_previous_days(df, next_day, 1)
        pred_Inverse = forecast_model(last_7d_from_day, decoder_features, model, outputs_train)
        idx_day = get_previous_days(df,day,0).index
        for idx_count, idx_row in enumerate(idx_day):
            target_df.at[idx_row, PREDICTION_COLUMN] = float(pred_Inverse[0][idx_count])
    target_df = target_df[target_df.index.year == year]
    target_df = target_df[target_df.index.month == month]
    return target_df


def get_dataset(data):
    if ENCODER_DECODER:
        ds = EncoderDecoderDataset(data, INPUT_DAYS * 24, 24, 24, DEVICE)
    else:
        ds = TimeSeriesDataset(data, INPUT_DAYS * 24, 24, 24, DEVICE)
    return ds


def get_model(n_features, args):
    num_layers = args.num_layers
    d_model = args.d_model
    num_heads = args.num_heads
    if ENCODER_DECODER:
        model = EncoderDecoderTransformer(INPUT_DAYS * 24, n_features, 24, num_layers, d_model, num_heads)
    else:
        model = EncoderOnlyTransformer(INPUT_DAYS * 24, n_features, 24, num_layers, d_model, num_heads)
    model = model.to(DEVICE)
    return model


def apply_model(model, batch):
    if ENCODER_DECODER:
        x_enc, x_dec, y = batch
        y_hat = model(x_enc, x_dec)
    else:
        X, y = batch
        y_hat = model(X)
    return y_hat


def evaluate(model, data_loader, criterion):
    loss = 0
    with torch.no_grad():
        for batch in data_loader:
            y_hat = apply_model(model, batch)
            y = batch[-1]
            loss += criterion(y, y_hat)
    return loss / len(data_loader)


def train_model(data, args):
    n_features = len(data.columns)
    model = get_model(n_features, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = LinearLR(optimizer, total_iters=EPOCHS)
    criterion = torch.nn.MSELoss()
    ds = get_dataset(data)
    split_idx = int(0.9 * len(ds))
    train_ds = Subset(ds, list(range(split_idx)))
    val_ds = Subset(ds, list(range(split_idx, len(ds))))
    print(f"train ds size = {len(train_ds)}, val ds size = {len(val_ds)}")
    batch_size = min(args.batch_size, len(train_ds))
    train_data_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_ds, len(val_ds), shuffle=False)
    best_val_loss = np.inf
    epochs_without_improvement = 0
    for epoch in tqdm(list(range(EPOCHS))):
        #print(f"-- epoch {epoch + 1} --")
        epoch_loss = 0
        for batch in train_data_loader:
            optimizer.zero_grad()
            y_hat = apply_model(model, batch)
            y = batch[-1]
            loss = criterion(y, y_hat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        epoch_loss = epoch_loss / len(train_data_loader)
        #print(f"loss = {epoch_loss}")
        val_loss = evaluate(model, val_data_loader, criterion)
        #print(f"val loss = {val_loss}")
        wandb.log({
            "train_loss": epoch_loss,
            "val_loss": val_loss
        })
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_checkpoint = deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement == PATIENCE:
            print("early stopping")
            break
        best_val_loss = min(best_val_loss, val_loss)
        scheduler.step()
    print(f"best val loss = {best_val_loss}")
    model.load_state_dict(best_checkpoint)
    model.eval()
    return model, ds.scaler


def forecast_model(data, decoder_input, model, scaler):
    data = scale(data, scaler)
    data = torch.tensor(data, dtype=torch.float).to(DEVICE)
    if ENCODER_DECODER:
        decoder_input = scale(decoder_input, scaler)
        decoder_input[:, -1] = 0
        decoder_input = torch.tensor(decoder_input, dtype=torch.float).to(DEVICE)
        y_hat = model(data[None, :, :], decoder_input[None, :, :])
    else:
        y_hat = model(data[None, :, :])
    backtransformed = scaler.data_min_[-1] + y_hat * scaler.data_range_[-1]
    return backtransformed


def load_dataframe(dataset):
    if dataset == "AGG":
        dataset_file = "Feature Selection aggregated energy community load.pkl"
    elif dataset == "HP":
        dataset_file = "Feature Selection aggregated heat pump load.pkl"
    elif dataset == "HH":
        dataset_file = "Feature Selection aggregated household load.pkl"
    else:
        raise Exception(f"Unknown dataset {dataset}.")
    df = pd.read_pickle(f"{DATA_PATH}/{dataset_file}")
    return df


def sample_plot(
    xs: t.Union[t.Iterable, t.Iterable[t.Iterable]],
    ys: t.Iterable[t.Iterable],
    keys: t.Optional[t.Iterable] = None,
    title: t.Optional[str] = None,
    xname: t.Optional[str] = None,
):
    """Construct a line series plot.

    Arguments:
        xs (array of arrays, or array): Array of arrays of x values
        ys (array of arrays): Array of y values
        keys (array): Array of labels for the line plots
        title (string): Plot title.
        xname: Title of x-axis

    Returns:
        A plot object, to be passed to wandb.log()

    Example:
        When logging a singular array for xs, all ys are plotted against that xs
        <!--yeadoc-test:plot-line-series-single-->
        ```python
        import wandb

        run = wandb.init()
        xs = [i for i in range(10)]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot1": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
        xs can also contain an array of arrays for having different steps for each metric
        <!--yeadoc-test:plot-line-series-double-->
        ```python
        import wandb

        run = wandb.init()
        xs = [[i for i in range(10)], [2 * i for i in range(10)]]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot2": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
    """
    if not isinstance(xs, Iterable):
        raise TypeError(f"Expected xs to be an array instead got {type(xs)}")

    if not isinstance(ys, Iterable):
        raise TypeError(f"Expected ys to be an array instead got {type(xs)}")

    for y in ys:
        if not isinstance(y, Iterable):
            raise TypeError(
                f"Expected ys to be an array of arrays instead got {type(y)}"
            )

    if not isinstance(xs[0], Iterable) or isinstance(xs[0], (str, bytes)):
        xs = [xs for _ in range(len(ys))]
    assert len(xs) == len(ys), "Number of x-lines and y-lines must match"

    if keys is not None:
        assert len(keys) == len(ys), "Number of keys and y-lines must match"

    data = [
        [x, f"key_{i}" if keys is None else keys[i], y]
        for i, (xx, yy) in enumerate(zip(xs, ys))
        for x, y in zip(xx, yy)
    ]

    table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])

    return wandb.plot_table(
        "sample_plot",
        table,
        {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
        {"title": title, "xname": xname or "x"},
    )


def log_predictions(df):
    n_vals = len(df)
    residuals = df[TARGET_COLUMN] - df[PREDICTION_COLUMN]
    wandb.log({
        "predictions": sample_plot(
            xs=np.arange(n_vals),
            ys=[df[TARGET_COLUMN].values, df[PREDICTION_COLUMN].values, residuals.values],
            keys=["Target", "Prediction", "Residuals"],
            title="Predictions"
        )
    })


def log_monthly_rmse(df):
    months = sorted(np.unique([time_point.month for time_point in df.index]))
    rmse_vals = []
    for m in months:
        month_df = df[df.index.month == m]
        rmse_val = mean_squared_error(month_df[TARGET_COLUMN], month_df[PREDICTION_COLUMN], squared=False)
        rmse_vals.append(rmse_val)
    wandb.log({
        "monthly_rmse": sample_plot(
            xs=months,
            ys=[rmse_vals],
            keys=["RMSE"],
            title="Monthly RMSE"
        )
    })


def main(args):
    if args.seed is None:
        args.seed = random.randint(0, 99999)
    torch.manual_seed(args.seed)

    df = load_dataframe(args.dataset)
    if ENCODER_DECODER:
        df = reverse_weather_shift(df)

    if args.test:
        MONTHS = list(range(1, 13))
        YEAR = 2020
    else:
        MONTHS = list(range(7, 13))
        YEAR = 2019

    if args.ceemdan:
        year_df = pd.DataFrame()

        for month in MONTHS:
            print(f"===== month {month} ====")
            start_date = pd.Timestamp(year=YEAR - 1, month=month, day=1)
            end_date = start_date + pd.DateOffset(months=13)
            df_timerange = df[start_date:end_date]

            # calculate ceemdan
            print("computing CEEMDAN...")
            ceem = calculate_ceemdan(df_timerange.copy(), "Agg Load")
            print("done.")

            res_df = pd.DataFrame()
            final_df_month = pd.DataFrame()

            for col_i, col in enumerate(ceem.columns):
                print(f"column {col_i + 1}/{len(ceem.columns)}")

                col_df = df_timerange.copy()
                col_df[TARGET_COLUMN] = ceem[col].values
                target_df = analyze_month(col_df, YEAR, month, args)

                res_df[col] = target_df[PREDICTION_COLUMN].values

            final_df_month[PREDICTION_COLUMN] = res_df.sum(axis=1).values
            year_df = pd.concat([year_df, final_df_month])
            print(year_df)

        year_df[TARGET_COLUMN] = df[df.index.year == YEAR][TARGET_COLUMN].values[:year_df.shape[0]]
        year_df.index = df.index[df.index.year == YEAR][:year_df.shape[0]]
    else:
        target_dfs = []
        for m in MONTHS:
            print(f"*****month {m}*****")
            target_df = analyze_month(df, YEAR, m, args)
            target_dfs.append(target_df)
            rmse = mean_squared_error(target_df[TARGET_COLUMN], target_df[PREDICTION_COLUMN], squared=False)
            print(f"RMSE = {rmse}")
        year_df = pd.concat(target_dfs)

    targets = year_df[TARGET_COLUMN]
    prediction = year_df[PREDICTION_COLUMN]
    rmse = mean_squared_error(targets, prediction, squared=False)
    print(f"total RMSE = {rmse}")

    wandb.log({"RMSE": rmse})

    log_predictions(year_df)
    log_monthly_rmse(year_df)

    #year_df[[TARGET_COLUMN, PREDICTION_COLUMN]].plot()
    #plt.show()

    if args.test:
        result_file_name = f"Transformer_{args.dataset}{'_CEEMDAN' if args.ceemdan else ''}_seed{args.seed}.pkl"
        year_df.to_pickle(result_file_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=("AGG", "HH", "HP"), default="AGG")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--ceemdan", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)
    args = parser.parse_args()

    wandb.log(vars(args))

    main(args)
