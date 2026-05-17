import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def preprocess_for_prophet(df):
    prophet_df = df.reset_index().rename(columns={"Datetime": "ds", "PJME_MW": "y"})
    return prophet_df[["ds", "y"]]


def train_prophet(df):
    prophet_df = preprocess_for_prophet(df)
    model = Prophet()
    model.fit(prophet_df)
    return model


def forecast_prophet(model, df, periods):
    freq = pd.infer_freq(df.index) or "H"
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast = forecast.loc[forecast["ds"] > df.index.max(), ["ds", "yhat"]]
    forecast = forecast.rename(columns={"ds": "Datetime", "yhat": "Forecast"})
    forecast["Datetime"] = forecast["Datetime"].astype(str)
    forecast["Forecast"] = forecast["Forecast"].astype(float)
    return forecast


def train_arima(train_series, order=(1, 1, 1)):
    model = SARIMAX(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_arima(model, periods):
    forecast = model.get_forecast(steps=periods).predicted_mean
    forecast = forecast.reset_index()
    forecast.columns = ["Datetime", "Forecast"]
    forecast["Datetime"] = forecast["Datetime"].astype(str)
    forecast["Forecast"] = forecast["Forecast"].astype(float)
    return forecast


def evaluate_prophet(train_df, test_df):
    model = train_prophet(train_df)
    freq = pd.infer_freq(train_df.index) or "H"
    future = model.make_future_dataframe(periods=len(test_df), freq=freq)
    preds = model.predict(future)
    preds = preds.loc[preds["ds"] > train_df.index.max(), ["ds", "yhat"]]
    preds = preds["yhat"].values
    return compute_rmse(test_df["PJME_MW"].values, preds)


def evaluate_arima(train_df, test_df, order=(1, 1, 1)):
    train_series = train_df["PJME_MW"]
    model = train_arima(train_series, order=order)
    preds = model.get_forecast(steps=len(test_df)).predicted_mean
    return compute_rmse(test_df["PJME_MW"].values, preds.values)


def find_best_arima_order(train_df, test_df, candidate_orders=None):
    if candidate_orders is None:
        candidate_orders = [(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (3, 1, 1)]

    best_order = None
    best_rmse = float("inf")

    for order in candidate_orders:
        try:
            rmse = evaluate_arima(train_df, test_df, order=order)
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception:
            continue

    return best_order, best_rmse


def select_best_forecaster(df, periods=24, val_size=0.2, model_choice="auto"):
    df = df.copy()
    val_len = max(1, int(len(df) * val_size))
    train_df = df.iloc[:-val_len]
    test_df = df.iloc[-val_len:]

    if model_choice == "prophet":
        best_model_name = "prophet"
    elif model_choice == "arima":
        best_model_name = "arima"
    else:
        prophet_rmse = evaluate_prophet(train_df, test_df)
        best_order, arima_rmse = find_best_arima_order(train_df, test_df)
        if arima_rmse < prophet_rmse:
            best_model_name = "arima"
        else:
            best_model_name = "prophet"

    if best_model_name == "prophet":
        model = train_prophet(df)
        forecast = forecast_prophet(model, df, periods)
    else:
        if model_choice == "arima" or best_model_name == "arima":
            best_order, _ = find_best_arima_order(train_df, test_df)
            if best_order is None:
                best_order = (1, 1, 1)
            model = train_arima(df["PJME_MW"], order=best_order)
        forecast = forecast_arima(model, periods)

    return best_model_name, forecast
