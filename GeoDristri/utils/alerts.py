import math
from datetime import datetime


def current_crop_season(month: int) -> str:
    if month in (11, 12, 1, 2):
        return "Rabi"
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    if month in (3, 4, 5):
        return "Summer"
    return "Kharif"


def current_weather_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def severity_from_intensity(intensity: float) -> str:
    if intensity >= 8:
        return "RED"
    if intensity >= 4:
        return "YELLOW"
    return "GREEN"


def drought_category_label(value: float) -> str:
    if value <= -2.0:
        return "Extremely Dry"
    if value <= -1.5:
        return "Severely Dry"
    if value <= -1.0:
        return "Moderately Dry"
    if value < 1.0:
        return "Near Normal"
    if value < 1.5:
        return "Moderately Wet"
    if value < 2.0:
        return "Severely Wet"
    return "Extremely Wet"


def drought_alert_level(category: str, drought_active: bool) -> str:
    if drought_active and category in {"Extremely Dry", "Severely Dry"}:
        return "RED"
    if drought_active and category == "Moderately Dry":
        return "YELLOW"
    return "GREEN"


def alert_score(ndvi: float | None, temp: float | None, wind: float | None, precip: float | None) -> str:
    score = 0
    if ndvi is not None:
        if ndvi < 0.15:
            score += 3
        elif ndvi < 0.3:
            score += 1
    if temp is not None:
        if temp > 45:
            score += 3
        elif temp > 40:
            score += 2
        elif temp > 37:
            score += 1
    if wind is not None:
        if wind > 80:
            score += 3
        elif wind > 55:
            score += 2
        elif wind > 35:
            score += 1
    if precip is not None:
        if precip > 80:
            score += 2
        elif precip > 50:
            score += 1
    if score >= 6:
        return "RED"
    if score >= 3:
        return "YELLOW"
    return "GREEN"


def estimate_spei(precip_mm: float, avg_temp: float, humidity: float) -> float:
    temp = max(avg_temp, 0)
    heat_index = max((temp / 5) ** 1.514, 0.001)
    alpha = 0.492 + 0.0179 * heat_index - 7.71e-5 * heat_index**2 + 6.75e-7 * heat_index**3
    pet = 16 * ((10 * temp / (heat_index * 12)) ** alpha) if temp > 0 else 0
    pet_adjusted = max(pet, 0) * (1 - humidity / 200)
    water_balance = precip_mm - pet_adjusted
    return round(max(min(water_balance / (pet_adjusted + 50 + 1e-6), 3.5), -3.5), 4)


def encode_spatiotemporal(lat: float, lon: float, month: int) -> dict[str, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    month_rad = math.radians(month * 30)
    return {
        "lat_sin": round(math.sin(lat_rad), 6),
        "lat_cos": round(math.cos(lat_rad), 6),
        "lon_sin": round(math.sin(lon_rad), 6),
        "lon_cos": round(math.cos(lon_rad), 6),
        "month_sin": round(math.sin(month_rad), 6),
        "month_cos": round(math.cos(month_rad), 6),
    }


def current_month() -> int:
    return datetime.now().month

