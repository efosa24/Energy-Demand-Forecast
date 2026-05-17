import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    df,
    datetime_col='Datetime',
    target_col='PJME_MW',
    resample_freq='H',
    fill_method='time',
):
    """Clean raw time series and return a uniformly resampled target series."""
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=[datetime_col, target_col])
    df = df.sort_values(datetime_col)
    df = df.drop_duplicates(subset=[datetime_col])
    df = df.set_index(datetime_col)
    df = df[[target_col]]
    df = df.resample(resample_freq).mean()
    if df[target_col].isna().any():
        df[target_col] = df[target_col].interpolate(method=fill_method, limit_direction='both')
    return df


def add_time_features(df):
    """Add calendar features for time series modeling."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    return df


def add_lag_features(df, target_col='PJME_MW', lags=None):
    """Add lag features for autoregressive modeling."""
    if lags is None:
        lags = [1, 24, 168]
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col='PJME_MW', windows=None):
    """Add rolling aggregate features for trend and seasonality."""
    if windows is None:
        windows = [24, 72, 168]
    df = df.copy()
    for window in windows:
        df[f'roll_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'roll_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
    return df


def build_feature_matrix(df, target_col='PJME_MW'):
    """Run full feature engineering pipeline and drop rows with missing values."""
    df = add_time_features(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    df = df.dropna()
    return df


def train_test_split(df, test_size=0.2):
    """Split time series data chronologically into train and test sets."""
    if not 0 < test_size < 1:
        raise ValueError('test_size must be a fraction between 0 and 1')
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def scale_features(train_df, test_df, features=None):
    """Scale numeric feature columns using StandardScaler."""
    if features is None:
        features = train_df.columns.difference(['PJME_MW'])
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[features] = scaler.fit_transform(train_scaled[features])
    test_scaled[features] = scaler.transform(test_scaled[features])
    return train_scaled, test_scaled, scaler


def save_preprocessed_data(df, path='app/data/processed/energy_processed.csv'):
    """Save the processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    return path


if __name__ == '__main__':
    import sys

    # Ensure the repository root is on sys.path when running this file directly.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from app.preprocessing.data_loader import load_data
    from app.config.settings import DATA_PATH

    raw_df = load_data(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    feature_df = build_feature_matrix(processed_df)
    saved_path = save_preprocessed_data(feature_df)
    print(f'Preprocessed data saved to: {saved_path}')
