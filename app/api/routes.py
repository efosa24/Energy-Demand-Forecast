import os
import sys

# Allow direct execution of this module from the repository root.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import APIRouter

from app.config.settings import DATA_PATH, TEST_SIZE
from app.preprocessing.data_loader import load_data
from app.preprocessing.preprocessing import preprocess_data
from app.forecasting.prophet_forecaster import select_best_forecaster

router = APIRouter()

@router.get("/forecast")
def forecast(periods: int = 24, model: str = "auto"):
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    best_model, forecast_df = select_best_forecaster(df, periods=periods, val_size=TEST_SIZE, model_choice=model)
    return {
        "model": best_model,
        "periods": periods,
        "forecast": forecast_df.to_dict(orient="records"),
    }
