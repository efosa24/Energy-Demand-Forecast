import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
import logging
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(filepath):
    """Loads dataset from the given filepath."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data successfully loaded from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def preprocess_data(df):
    """Preprocesses data: handling datetime, sorting, deduplication, and interpolation."""
    if df is None:
        logging.error("Dataframe is empty. Skipping preprocessing.")
        return None

    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)

        df.sort_values(by='Datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.rename(columns={'PJME_MW': 'demand_in_MW'}, inplace=True)

        df.drop_duplicates(subset='Datetime', keep='last', inplace=True)

        df.set_index('Datetime', inplace=True)

        # Ensure a continuous datetime index
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df = df.reindex(date_range)
        df['demand_in_MW'].interpolate(method='linear', inplace=True)

        logging.info("Preprocessing complete. Data is now continuous.")
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return None


def train_test_split_time_series(df, test_size=0.2):
    """Splits time series data into train and test sets."""
    train_size = int(len(df) * (1 - test_size))
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    return train, test


def prophet_forecast(train, test):
    """Forecasts using Facebook Prophet."""
    logging.info("Training Prophet model...")
    
    df_train = train.reset_index().rename(columns={"Datetime": "ds", "demand_in_MW": "y"})
    
    model = Prophet()
    model.fit(df_train)

    future = pd.DataFrame(test.index).rename(columns={"Datetime": "ds"})
    forecast = model.predict(future)

    return forecast[['ds', 'yhat']], model


def holt_winters_forecast(train, test):
    """Forecasts using Holt-Winters (ETS) Model."""
    logging.info("Training Holt-Winters model...")

    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=24).fit()
    predictions = model.forecast(len(test))
    
    return predictions, model


def arima_forecast(train, test):
    """Forecasts using ARIMA model."""
    logging.info("Training ARIMA model...")

    model = ARIMA(train, order=(5, 1, 0)).fit()
    predictions = model.forecast(steps=len(test))

    return predictions, model


def random_forest_forecast(train, test):
    """Forecasts using Random Forest Regressor."""
    logging.info("Training Random Forest model...")

    train['hour'] = train.index.hour
    test['hour'] = test.index.hour

    X_train, X_test = train[['hour']], test[['hour']]
    y_train, y_test = train['demand_in_MW'], test['demand_in_MW']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions, model


def evaluate_model(actual, predicted, model_name):
    """Evaluates the forecasting model."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    logging.info(f"{model_name} Performance: MAE={mae}, RMSE={rmse}")
    return {"MAE": mae, "RMSE": rmse}


# === Run the pipeline ===
if __name__ == "__main__":
    FILEPATH = "../Energy.csv"

    df = load_data(FILEPATH)
    df = preprocess_data(df)

    if df is not None:
        train, test = train_test_split_time_series(df)

        # Prophet Model
        prophet_preds, prophet_model = prophet_forecast(train, test)
        evaluate_model(test['demand_in_MW'], prophet_preds['yhat'], "Prophet")

        # Holt-Winters Model
        hw_preds, hw_model = holt_winters_forecast(train['demand_in_MW'], test)
        evaluate_model(test['demand_in_MW'], hw_preds, "Holt-Winters")

        # ARIMA Model
        arima_preds, arima_model = arima_forecast(train['demand_in_MW'], test)
        evaluate_model(test['demand_in_MW'], arima_preds, "ARIMA")

        # Random Forest Model
        rf_preds, rf_model = random_forest_forecast(train.copy(), test.copy())
        evaluate_model(test['demand_in_MW'], rf_preds, "Random Forest")

        logging.info("All models have been trained and evaluated.")
