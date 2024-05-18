import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


WEATHER_FEATURES = [
    "WIND", "HUMIDITY", "TEMPERATURE", "PRECIPITATION"
]


def is_weather_feature(feature_name):
    for weather_feature in WEATHER_FEATURES:
        if weather_feature in feature_name:
            return True
    return False


def reverse_weather_shift(df):
    for column in df.columns:
        if is_weather_feature(column):
            print(column)
            df[column][24:] = df[column][:-24]
    return df


def trim_to_first_day(df):
    first_hour = df.index[0].hour
    if first_hour != 0:
        trim_steps = 24 - first_hour
        df = df.iloc[trim_steps:]
    return df


def get_windows(data, input_length, horizon, step):
    n_windows = (data.shape[0] - input_length - horizon) // step + 1
    n_features = data.shape[1]
    data = torch.tensor(data)
    X = torch.zeros((n_windows, input_length, n_features))
    Y = torch.zeros((n_windows, horizon))
    for i in range(n_windows):
        input_start = i * step
        input_end = input_start + input_length
        target_end = input_end + horizon
        X[i, :, :] = data[input_start:input_end]
        Y[i, :] = data[input_end:target_end, -1]
    return X, Y


def scale(df, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        return df, scaler
    else:
        df = scaler.transform(df)
        return df


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length, horizon, step, device):
        data = trim_to_first_day(data)
        data, self.scaler = scale(data)
        self.X, self.Y = get_windows(data, input_length, horizon, step)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_encoder_decoder_windows(data, input_length, horizon, step):
    n_windows = (data.shape[0] - input_length - horizon) // step + 1
    n_features = data.shape[1]
    data = torch.tensor(data)
    X_enc = torch.zeros((n_windows, input_length, n_features))
    X_dec = torch.zeros((n_windows, horizon, n_features))
    Y = torch.zeros((n_windows, horizon))
    for i in range(n_windows):
        input_start = i * step
        input_end = input_start + input_length
        target_end = input_end + horizon
        X_enc[i, :, :] = data[input_start:input_end]
        X_dec[i, :, :] = data[input_end:target_end]
        Y[i, :] = data[input_end:target_end, -1]
    X_dec[:, :, -1] = 0  # remove target values from decoder input
    return X_enc, X_dec, Y


class EncoderDecoderDataset(Dataset):
    def __init__(self, data, input_length, horizon, step, device):
        data = trim_to_first_day(data)
        data, self.scaler = scale(data)
        self.X_enc, self.X_dec, self.Y = get_encoder_decoder_windows(data, input_length, horizon, step)
        self.X_enc = self.X_enc.to(device)
        self.X_dec = self.X_dec.to(device)
        self.Y = self.Y.to(device)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X_enc[idx], self.X_dec[idx], self.Y[idx]
