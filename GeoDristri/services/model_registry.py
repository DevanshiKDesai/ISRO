from pathlib import Path
from threading import Lock
from typing import Any, Optional
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def data_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def safe_read_csv(*parts: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(data_path(*parts), **kwargs)
    except Exception:
        return None


def safe_read_excel(*parts: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(data_path(*parts), **kwargs)
    except Exception:
        return None


def safe_joblib_load(*parts: str) -> Any:
    try:
        return joblib.load(data_path(*parts))
    except Exception:
        return None


def safe_json_load(*parts: str, default: Any = None) -> Any:
    try:
        with open(data_path(*parts), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


class Registry:
    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = Lock()
        self._init_empty()

    def _init_empty(self) -> None:
        self.df_pop = None
        self.df_forest = None
        self.df_crop = None
        self.df_drought = None

        self.crop_model = None
        self.crop_encoders = {}
        self.crop_target_encoder = None
        self.crop_feature_cols = []

        self.drought_category_model = None
        self.drought_status_model = None
        self.drought_feature_cols = []
        self.drought_category_encoder = None

        self.weather_event_model = None
        self.weather_intensity_model = None
        self.weather_feature_cols = []
        self.weather_intensity_feature_cols = []
        self.weather_state_encoder = None
        self.weather_season_encoder = None
        self.weather_event_encoder = None
        self.weather_meta = {}
        self.coastal_states = set()
        self.hilly_states = set()
        self.dry_states = set()

        self.forest_alert_model = None
        self.forest_ndvi_model = None
        self.forest_cover_model = None
        self.forest_aqi_model = None
        self.forest_human_model = None
        self.forest_state_encoder = None
        self.forest_feature_cols = []
        self.forest_meta = {}
        self.forest_state_data = {}

        self.urban_pop_model = None
        self.urban_urb_model = None
        self.urban_infra_model = None
        self.urban_upop_model = None
        self.urban_grow_model = None
        self.urban_feature_cols = []
        self.urban_meta = {}
        self.urban_national_latest = {}
        self.urban_city_rates = {}
        self.urban_state_rates = {}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_all()
            self._loaded = True

    def _load_all(self) -> None:
        self.df_pop = safe_read_csv("Population", "india_enriched.csv", encoding="latin1")
        self.df_forest = safe_read_csv("Forest_prediction", "New Forest.csv", encoding="latin1")
        self.df_crop = safe_read_csv("crop_predictor", "enhanced_crop_yield_dataset (1).csv", encoding="latin1")

        self.df_drought = None
        for engine_name in ("openpyxl", "xlrd"):
            self.df_drought = safe_read_excel("drought_prediction", "Drought New.xlsx", engine=engine_name)
            if self.df_drought is not None:
                break
        if self.df_drought is None:
            self.df_drought = safe_read_csv(
                "drought_prediction",
                "Drought New.xlsx",
                encoding="latin1",
                on_bad_lines="skip",
                sep=None,
                engine="python",
            )

        self.crop_model = safe_joblib_load("crop_predictor", "model.joblib")
        self.crop_encoders = safe_joblib_load("crop_predictor", "encoders.joblib") or {}
        self.crop_target_encoder = safe_joblib_load("crop_predictor", "target_encoder.joblib")
        self.crop_feature_cols = safe_joblib_load("crop_predictor", "feature_cols.joblib") or []

        self.drought_category_model = safe_joblib_load("drought_prediction", "category_model.joblib")
        self.drought_status_model = safe_joblib_load("drought_prediction", "status_model.joblib")
        self.drought_feature_cols = safe_joblib_load("drought_prediction", "feature_cols.joblib") or []
        self.drought_category_encoder = safe_joblib_load("drought_prediction", "category_encoder.joblib")

        self.weather_event_model = safe_joblib_load("Weather_Prediction", "event_model.joblib")
        self.weather_intensity_model = safe_joblib_load("Weather_Prediction", "intensity_model.joblib")
        self.weather_feature_cols = safe_joblib_load("Weather_Prediction", "feature_cols.joblib") or []
        self.weather_intensity_feature_cols = safe_joblib_load("Weather_Prediction", "intensity_feature_cols.joblib") or []
        self.weather_state_encoder = safe_joblib_load("Weather_Prediction", "state_encoder.joblib")
        self.weather_season_encoder = safe_joblib_load("Weather_Prediction", "season_encoder.joblib")
        self.weather_event_encoder = safe_joblib_load("Weather_Prediction", "event_encoder.joblib")
        self.weather_meta = safe_json_load("Weather_Prediction", "metadata.json", default={}) or {}
        self.coastal_states = set(self.weather_meta.get("coastal_states", []))
        self.hilly_states = set(self.weather_meta.get("hilly_states", []))
        self.dry_states = set(self.weather_meta.get("dry_states", []))

        self.forest_alert_model = safe_joblib_load("Forest_prediction", "alert_model.joblib")
        self.forest_ndvi_model = safe_joblib_load("Forest_prediction", "ndvi_model.joblib")
        self.forest_cover_model = safe_joblib_load("Forest_prediction", "cover_model.joblib")
        self.forest_aqi_model = safe_joblib_load("Forest_prediction", "aqi_model.joblib")
        self.forest_human_model = safe_joblib_load("Forest_prediction", "human_model.joblib")
        self.forest_state_encoder = safe_joblib_load("Forest_prediction", "state_encoder.joblib")
        self.forest_feature_cols = safe_joblib_load("Forest_prediction", "feature_cols.joblib") or []
        self.forest_meta = safe_json_load("Forest_prediction", "metadata.json", default={}) or {}
        self.forest_state_data = {
            row["State"]: row
            for row in safe_json_load("Forest_prediction", "state_data.json", default=[]) or []
            if isinstance(row, dict) and row.get("State")
        }

        self.urban_pop_model = safe_joblib_load("Population", "pop_model.joblib")
        self.urban_urb_model = safe_joblib_load("Population", "urb_model.joblib")
        self.urban_infra_model = safe_joblib_load("Population", "infra_model.joblib")
        self.urban_upop_model = safe_joblib_load("Population", "upop_model.joblib")
        self.urban_grow_model = safe_joblib_load("Population", "grow_model.joblib")
        self.urban_feature_cols = safe_joblib_load("Population", "feature_cols.joblib") or []
        self.urban_meta = safe_json_load("Population", "metadata.json", default={}) or {}
        self.urban_national_latest = self.urban_meta.get("national_latest", {})
        self.urban_city_rates = self.urban_meta.get("city_urb_rates", {})
        self.urban_state_rates = self.urban_meta.get("state_urb_rates", {})

    @property
    def model_status(self) -> dict[str, bool]:
        self.ensure_loaded()
        return {
            "crop_yield_predictor": self.crop_model is not None,
            "drought_category_model": self.drought_category_model is not None,
            "drought_status_model": self.drought_status_model is not None,
            "weather_event_model": self.weather_event_model is not None,
            "weather_intensity_model": self.weather_intensity_model is not None,
            "forest_alert_model": self.forest_alert_model is not None,
            "forest_ndvi_model": self.forest_ndvi_model is not None,
            "forest_cover_model": self.forest_cover_model is not None,
            "forest_aqi_model": self.forest_aqi_model is not None,
            "forest_human_model": self.forest_human_model is not None,
            "urban_pop_model": self.urban_pop_model is not None,
            "urban_urb_model": self.urban_urb_model is not None,
            "urban_infra_model": self.urban_infra_model is not None,
            "urban_upop_model": self.urban_upop_model is not None,
            "urban_grow_model": self.urban_grow_model is not None,
        }


registry = Registry()
