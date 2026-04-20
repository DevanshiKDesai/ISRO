from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import requests


def geocode(loc: str) -> tuple[Optional[float], Optional[float]]:
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": loc, "count": 1},
            timeout=6,
        )
        data = resp.json()
        if "results" not in data or not data["results"]:
            return None, None
        return data["results"][0]["latitude"], data["results"][0]["longitude"]
    except Exception:
        return None, None


def reverse_geocode_state(lat: float, lon: float, fallback: Optional[str] = None) -> Optional[str]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "geodhrishti-backend/1.0"},
            timeout=10,
        )
        data = resp.json()
        addr = data.get("address", {})
        
        # Priority order for state-like fields
        state = addr.get("state") or addr.get("state_district") or addr.get("region") or addr.get("province")
        
        # If no state-like field, fallback to country for context, or use fallback
        return state or addr.get("country") or fallback
    except Exception:
        return fallback


def is_land_area(lat: float, lon: float) -> bool:
    """Check if coordinates are on land using SoilGrids API."""
    try:
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon,
                "lat": lat,
                "property": ["phh2o"],
                "depth": "0-30cm",
                "value": "mean",
            },
            timeout=5,
        )
        data = resp.json()
        # ISRIC returns empty layers or error for oceanic points
        layers = data.get("properties", {}).get("layers", [])
        return len(layers) > 0
    except Exception:
        # If API fails, fallback to true to not block, 
        # but Nominatim check will still run in service layer
        return True


def live_weather(lat: float, lon: float) -> dict[str, Any]:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "daily": [
                "precipitation_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "windspeed_10m_max",
                "shortwave_radiation_sum",
                "sunshine_duration",
            ],
            "hourly": "relativehumidity_2m",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def ndvi_proxy(lat: float, lon: float) -> Optional[float]:
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=16)).strftime("%Y%m%d")
        resp = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params={
                "parameters": "ALLSKY_SFC_SW_DWN,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": start,
                "end": end,
                "format": "JSON",
            },
            timeout=12,
        )
        payload = resp.json()["properties"]["parameter"]
        srad = [v for v in payload["ALLSKY_SFC_SW_DWN"].values() if v != -999]
        prec = [v for v in payload["PRECTOTCORR"].values() if v != -999]
        return round(
            min(max(float(np.nanmean(prec) if prec else 2) * 0.04 + float(np.nanmean(srad) if srad else 4) * 0.005, 0), 1),
            3,
        )
    except Exception:
        return None


def get_altitude(lat: float, lon: float) -> float:
    try:
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": [{"latitude": lat, "longitude": lon}]},
            timeout=10,
        )
        resp.raise_for_status()
        elevation = resp.json()["results"][0]["elevation"]
        return float(elevation) if elevation is not None else 350.0
    except Exception:
        return 350.0


def get_soil_data(lat: float, lon: float) -> dict[str, float]:
    try:
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon,
                "lat": lat,
                "property": ["phh2o", "soc", "wv0010"],
                "depth": "0-30cm",
                "value": "mean",
            },
            timeout=15,
        )
        resp.raise_for_status()
        layers = resp.json()["properties"]["layers"]
        result: dict[str, float] = {}
        for layer in layers:
            name = layer["name"]
            value = layer["depths"][0]["values"]["mean"]
            if value is None:
                continue
            if name == "phh2o":
                result["pH"] = round(value / 10, 2)
            elif name == "soc":
                result["Organic_Carbon"] = round(value / 10, 2)
            elif name == "wv0010":
                result["Soil_Moisture"] = round(value / 10, 2)
        result.setdefault("pH", 6.8)
        result.setdefault("Organic_Carbon", 0.8)
        result.setdefault("Soil_Moisture", 38.0)
        return result
    except Exception:
        return {"pH": 6.8, "Organic_Carbon": 0.8, "Soil_Moisture": 38.0}


def get_mei_index() -> float:
    try:
        resp = requests.get("https://psl.noaa.gov/enso/mei/data/meiv2.data", timeout=10)
        lines = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                values = [float(v) for v in parts[1:] if v not in {"-999.00", "-999"}]
                if values:
                    return round(values[-1], 2)
    except Exception:
        pass
    return 0.0

