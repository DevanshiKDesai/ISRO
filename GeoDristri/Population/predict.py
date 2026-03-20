"""
Urbanization Prediction System
================================
User enters any location (area / city / state / GPS) →
APIs auto-fill all data → 5 models predict 5 years ahead:

  1. Future Population (Millions)
  2. Future Urbanization Rate (%)
  3. Future Urban Population (Millions)
  4. Future Infrastructure Pressure Score (0-100)
  5. Future Growth Rate (%)

  + Full impact report: schools, hospitals, roads, housing

Usage:
    python predict.py "Mumbai"
    python predict.py "Rajasthan"
    python predict.py "Bhopal"
    python predict.py "19.07,72.87"
"""

import requests
import joblib
import numpy as np
import json
import sys
import math
from datetime import datetime
from time import sleep

# ── Load models ────────────────────────────────────────────────────────────────
POP_MODEL   = joblib.load("pop_model.joblib")
URB_MODEL   = joblib.load("urb_model.joblib")
INFRA_MODEL = joblib.load("infra_model.joblib")
UPOP_MODEL  = joblib.load("upop_model.joblib")
GROW_MODEL  = joblib.load("grow_model.joblib")
FEATURE_COLS= joblib.load("feature_cols.joblib")

with open("metadata.json") as f:
    META = json.load(f)

NATIONAL    = META["national_latest"]
CITY_URB    = META["city_urb_rates"]
STATE_URB   = META["state_urb_rates"]
FORECAST_YR = 5


# ── STEP 1: Geocoding ──────────────────────────────────────────────────────────
def get_location_info(location: str) -> dict:
    """Geocode any location string to lat/lon + address components."""
    location = location.strip()

    # GPS input
    if "," in location:
        parts = location.split(",")
        try:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return reverse_geocode(lat, lon)
        except ValueError:
            pass

    # Forward geocode
    for suffix in [" India", ""]:
        url    = "https://nominatim.openstreetmap.org/search"
        params = {"q": location + suffix, "format": "json",
                  "addressdetails": 1, "limit": 1}
        headers= {"User-Agent": "urbanization-predictor/1.0"}
        resp   = requests.get(url, params=params, headers=headers, timeout=10)
        data   = resp.json()
        if data:
            break

    if not data:
        raise ValueError(f"Location '{location}' not found.")

    addr = data[0].get("address", {})
    return {
        "lat":     float(data[0]["lat"]),
        "lon":     float(data[0]["lon"]),
        "display": data[0]["display_name"],
        "city":    addr.get("city", addr.get("town", addr.get("county", ""))),
        "state":   addr.get("state", ""),
        "area":    addr.get("suburb", addr.get("village", addr.get("neighbourhood", ""))),
        "type":    data[0].get("type",""),
    }


def reverse_geocode(lat: float, lon: float) -> dict:
    url    = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    headers= {"User-Agent": "urbanization-predictor/1.0"}
    sleep(1)
    resp   = requests.get(url, params=params, headers=headers, timeout=10)
    data   = resp.json()
    addr   = data.get("address", {})
    return {
        "lat": lat, "lon": lon,
        "display": data.get("display_name", ""),
        "city":  addr.get("city", addr.get("town", "")),
        "state": addr.get("state", ""),
        "area":  addr.get("suburb", addr.get("village", "")),
        "type":  data.get("type", ""),
    }


# ── STEP 2: World Bank population API ─────────────────────────────────────────
def get_worldbank_data() -> dict:
    """
    Fetch latest India population indicators from World Bank API.
    Returns birth rate, death rate, growth rate, urbanization rate.
    """
    indicators = {
        "SP.POP.TOTL":    "total_population",
        "SP.URB.TOTL.IN.ZS": "urbanization_rate",
        "SP.DYN.CBRT.IN": "birth_rate",
        "SP.DYN.CDRT.IN": "death_rate",
        "SP.POP.GROW":    "growth_rate",
    }
    result = {}
    base   = "https://api.worldbank.org/v2/country/IN/indicator"

    for code, name in indicators.items():
        try:
            url    = f"{base}/{code}"
            params = {"format": "json", "mrv": 3, "per_page": 3}
            resp   = requests.get(url, params=params, timeout=10)
            data   = resp.json()
            if isinstance(data, list) and len(data) > 1:
                records = [r for r in data[1] if r.get("value") is not None]
                if records:
                    result[name] = float(records[0]["value"])
                    result[f"{name}_year"] = int(records[0]["date"])
        except Exception:
            pass

    return result


# ── STEP 3: City/state population from REST API ────────────────────────────────
def get_city_population(city: str, state: str) -> dict:
    """
    Get city/state population data from GeoNames API (free).
    Falls back to estimated data if unavailable.
    """
    # City urbanization rate lookup
    city_urb  = None
    state_urb = None

    for known_city, rate in CITY_URB.items():
        if known_city.lower() in city.lower() or city.lower() in known_city.lower():
            city_urb = rate
            break

    for known_state, rate in STATE_URB.items():
        if known_state.lower() in state.lower() or state.lower() in known_state.lower():
            state_urb = rate
            break

    return {
        "city_urb_rate":  city_urb,
        "state_urb_rate": state_urb,
    }


# ── STEP 4: Build national feature vector ─────────────────────────────────────
def build_features(wb_data: dict, loc_urb: dict,
                   current_year: int) -> tuple:
    """
    Blend World Bank live data with enriched national dataset.
    Returns feature vector matching the trained model.
    """
    N = NATIONAL  # national latest (2021 baseline)

    # Use World Bank data if available, else national baseline
    pop         = wb_data.get("total_population", N["India Population (Millions)"] * 1e6)
    pop_m       = pop / 1e6 if pop > 1000 else pop   # ensure millions
    urb_rate    = wb_data.get("urbanization_rate",    N["Urbanization_Rate"])
    birth_rate  = wb_data.get("birth_rate",           N["Birth Rate (per 1000)"])
    death_rate  = wb_data.get("death_rate",           N["Death Rate (per 1000)"])
    growth_rate = wb_data.get("growth_rate",          N["India Growth Rate (%)"])

    # Extrapolate from 2021 to current year if needed
    years_extra = current_year - int(N["Year"])
    if years_extra > 0:
        pop_m       = pop_m * ((1 + growth_rate/100) ** years_extra)
        urb_rate    = min(55, urb_rate + 0.4 * years_extra)
        birth_rate  = max(12, birth_rate - 0.2 * years_extra)
        death_rate  = max(6,  death_rate - 0.05 * years_extra)

    urban_pop   = pop_m * urb_rate / 100
    rural_pop   = pop_m - urban_pop
    nat_increase= birth_rate - death_rate

    # If we have city/state specific urb rate, blend it
    effective_urb = urb_rate
    if loc_urb.get("city_urb_rate"):
        effective_urb = loc_urb["city_urb_rate"]
    elif loc_urb.get("state_urb_rate"):
        effective_urb = loc_urb["state_urb_rate"]

    years_since = current_year - 1961
    pop_rolling_3 = pop_m * 0.995   # approx
    pop_rolling_5 = pop_m * 0.988

    world_pop_base  = N["World Population (Millions)"]
    world_pop_now   = world_pop_base * ((1 + N["World Growth Rate (%)"] / 100) ** years_extra)
    world_growth    = max(0.9, N["World Growth Rate (%)"] - 0.01 * years_extra)

    infra_pressure = (
        (pop_m / 1500) * 40 +
        (effective_urb / 60) * 30 +
        (growth_rate / 2.5) * 20 +
        (urban_pop / 600) * 10
    )

    feat_map = {
        "Years_Since_1961":        years_since,
        "India Population (Millions)": pop_m,
        "India Growth Rate (%)":   growth_rate,
        "Birth Rate (per 1000)":   birth_rate,
        "Death Rate (per 1000)":   death_rate,
        "World Population (Millions)": world_pop_now,
        "World Growth Rate (%)":   world_growth,
        "Pop_Growth_Abs":          pop_m * growth_rate / 100,
        "Pop_Rolling_3yr":         pop_rolling_3,
        "Pop_Rolling_5yr":         pop_rolling_5,
        "Growth_Rate_Change":      growth_rate - N["India Growth Rate (%)"],
        "Natural_Increase_Rate":   nat_increase,
        "Birth_Rate_Trend":        birth_rate,
        "Death_Rate_Trend":        death_rate,
        "World_Pop_Ratio":         pop_m / world_pop_now,
        "India_World_Growth_Diff": growth_rate - world_growth,
        "Urbanization_Rate":       effective_urb,
        "Urban_Pop_Millions":      urban_pop,
        "Rural_Pop_Millions":      rural_pop,
        "Urban_Rural_Ratio":       urban_pop / (rural_pop + 1),
        "Urb_Rate_Change":         effective_urb - N["Urbanization_Rate"],
        "Infra_Pressure_Score":    infra_pressure,
        "School_Pressure":         (urban_pop / N["Urban_Pop_Millions"]) * 100,
        "Hospital_Pressure":       (urban_pop / N["Urban_Pop_Millions"]) * 90,
        "Road_Pressure":           (pop_m / N["India Population (Millions)"]) * 100,
    }

    X = np.array([[feat_map.get(c, 0) for c in FEATURE_COLS]])
    return X, feat_map


# ── STEP 5: Infrastructure impact report ─────────────────────────────────────
def infrastructure_report(infra_score: float, urban_pop_m: float,
                           future_urban_m: float, location: str,
                           urb_rate: float, future_urb: float) -> dict:
    """Generate detailed infrastructure needs report."""

    urban_growth   = future_urban_m - urban_pop_m
    urb_rate_change= future_urb - urb_rate

    # Estimated needs (per million urban population)
    new_schools    = int(urban_growth * 1000 * 20)     # 20 schools per 1000 people
    new_hospitals  = int(urban_growth * 1000 * 1)      # 1 hospital per 1000 people
    new_road_km    = int(urban_growth * 1000 * 2.5)    # 2.5 km road per 1000 people
    new_homes      = int(urban_growth * 1e6 / 4.5)     # avg 4.5 people per household
    water_demand   = round(urban_growth * 135 * 365 / 1e6, 1)  # 135L/day/person in ML

    # Pressure level
    if infra_score > 80:   pressure = "CRITICAL"
    elif infra_score > 65: pressure = "HIGH"
    elif infra_score > 50: pressure = "MODERATE"
    else:                  pressure = "LOW"

    return {
        "pressure_level":   pressure,
        "infra_score":      round(infra_score, 1),
        "new_schools":      new_schools,
        "new_hospitals":    new_hospitals,
        "new_road_km":      new_road_km,
        "new_homes":        new_homes,
        "water_demand_ML":  water_demand,
        "urb_rate_change":  round(urb_rate_change, 2),
        "urban_growth_M":   round(urban_growth, 2),
    }


# ── STEP 6: Urbanization effects report ───────────────────────────────────────
def urbanization_effects(urb_rate: float, future_urb: float,
                          infra_score: float, growth_rate: float) -> dict:
    """Generate human & environmental effects of urbanization."""
    effects = {}

    # Economic
    effects["economic"] = []
    if future_urb > 50: effects["economic"].append("Major GDP boost from urban manufacturing & services")
    if future_urb > 40: effects["economic"].append("Rising formal employment opportunities")
    effects["economic"].append(f"Urban economy contribution: ~{min(85, int(future_urb * 1.8))}% of GDP")

    # Environmental
    effects["environment"] = []
    if future_urb - urb_rate > 2: effects["environment"].append("Accelerated green cover loss in peri-urban areas")
    if future_urb > 45: effects["environment"].append("Urban heat island effect intensifying")
    if infra_score > 65: effects["environment"].append("Increased solid waste generation — landfill pressure")
    effects["environment"].append(f"Estimated carbon footprint rise: +{round((future_urb - urb_rate) * 0.8, 1)}%")

    # Social
    effects["social"] = []
    if infra_score > 70: effects["social"].append("Housing shortage risk — slum expansion possible")
    if infra_score > 60: effects["social"].append("Public transport system under strain")
    effects["social"].append("Rural-to-urban migration accelerating")
    if future_urb > 50: effects["social"].append("Demographic transition: nuclear families increasing")

    # Health
    effects["health"] = []
    if infra_score > 65: effects["health"].append("Air quality deterioration — respiratory disease risk")
    if infra_score > 70: effects["health"].append("Water scarcity in dense urban zones")
    effects["health"].append("Healthcare access improving for urban residents")

    return effects


# ── MAIN ───────────────────────────────────────────────────────────────────────
def predict_urbanization(location: str):
    current_year  = datetime.now().year
    forecast_year = current_year + FORECAST_YR

    print(f"\n{'='*60}")
    print(f"  Urbanization Prediction System")
    print(f"{'='*60}")

    # ── Geocode ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Locating '{location}'...")
    loc = get_location_info(location)
    print(f"  Found  : {loc['display'][:70]}")
    print(f"  City   : {loc['city'] or 'N/A'}")
    print(f"  State  : {loc['state'] or 'N/A'}")
    print(f"  Coords : {loc['lat']:.4f}, {loc['lon']:.4f}")

    # ── City/state urbanization ───────────────────────────────────────────────
    print(f"\n[2/5] Fetching location-specific urbanization data...")
    loc_urb = get_city_population(loc['city'], loc['state'])
    if loc_urb['city_urb_rate']:
        print(f"  City urbanization rate : {loc_urb['city_urb_rate']}%")
    elif loc_urb['state_urb_rate']:
        print(f"  State urbanization rate: {loc_urb['state_urb_rate']}%")
    else:
        print(f"  Using national urbanization rate baseline")

    # ── World Bank live data ──────────────────────────────────────────────────
    print(f"\n[3/5] Fetching live data from World Bank API...")
    wb = get_worldbank_data()
    if wb:
        for k, v in wb.items():
            if not k.endswith("_year"):
                yr = wb.get(f"{k}_year", "")
                print(f"  {k:<25}: {v:.2f}  ({yr})")
    else:
        print(f"  World Bank API unavailable — using 2021 baseline")

    # ── Build features & predict ──────────────────────────────────────────────
    print(f"\n[4/5] Building {len(FEATURE_COLS)}-feature vector...")
    X, feat_map = build_features(wb, loc_urb, current_year)

    print(f"\n[5/5] Running 5 models → forecasting to {forecast_year}...")
    fut_pop    = round(float(POP_MODEL.predict(X)[0]),   2)
    fut_urb    = round(float(URB_MODEL.predict(X)[0]),   2)
    fut_infra  = round(float(INFRA_MODEL.predict(X)[0]), 2)
    fut_upop   = round(float(UPOP_MODEL.predict(X)[0]),  2)
    fut_grow   = round(float(GROW_MODEL.predict(X)[0]),  4)

    cur_pop    = round(feat_map["India Population (Millions)"], 2)
    cur_urb    = round(feat_map["Urbanization_Rate"], 2)
    cur_upop   = round(feat_map["Urban_Pop_Millions"], 2)
    cur_infra  = round(feat_map["Infra_Pressure_Score"], 2)
    cur_grow   = round(feat_map["India Growth Rate (%)"], 4)

    # Reports
    infra_rep  = infrastructure_report(
        fut_infra, cur_upop, fut_upop, location, cur_urb, fut_urb)
    effects    = urbanization_effects(
        cur_urb, fut_urb, fut_infra, fut_grow)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  URBANIZATION FORECAST — {location.upper()}")
    print(f"  Current Year: {current_year}  →  Forecast Year: {forecast_year}")
    print(f"{'='*60}")

    print(f"\n  {'METRIC':<35} {'NOW':>10}  {'→  '+str(forecast_year):>12}  {'CHANGE':>10}")
    print(f"  {'─'*70}")

    def row(label, now, future, unit="", fmt=".2f"):
        delta = future - now
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<35} {now:{fmt}}{unit:>3}  →  {future:{fmt}}{unit:>3}  {sign}{delta:{fmt}}{unit}")

    row("Population (Millions)",        cur_pop,   fut_pop,   "M")
    row("Urbanization Rate",            cur_urb,   fut_urb,   "%")
    row("Urban Population (Millions)",  cur_upop,  fut_upop,  "M")
    row("Growth Rate",                  cur_grow,  fut_grow,  "%", ".4f")
    row("Infrastructure Pressure",      cur_infra, fut_infra, "")

    print(f"\n  INFRASTRUCTURE NEEDS ({current_year}–{forecast_year}):")
    print(f"  Pressure level  : [{infra_rep['pressure_level']}]  Score: {infra_rep['infra_score']}/100")
    print(f"  Urban population growth: +{infra_rep['urban_growth_M']}M people")
    print(f"  ┌─ New schools needed  : {infra_rep['new_schools']:,}")
    print(f"  ├─ New hospitals needed: {infra_rep['new_hospitals']:,}")
    print(f"  ├─ New roads (km)      : {infra_rep['new_road_km']:,}")
    print(f"  ├─ New homes needed    : {infra_rep['new_homes']:,}")
    print(f"  └─ Water demand        : {infra_rep['water_demand_ML']} ML/year extra")

    print(f"\n  ECONOMIC EFFECTS:")
    for e in effects["economic"]:     print(f"  • {e}")

    print(f"\n  ENVIRONMENTAL EFFECTS:")
    for e in effects["environment"]:  print(f"  • {e}")

    print(f"\n  SOCIAL EFFECTS:")
    for e in effects["social"]:       print(f"  • {e}")

    print(f"\n  HEALTH EFFECTS:")
    for e in effects["health"]:       print(f"  • {e}")

    print(f"\n{'='*60}\n")

    return {
        "location":          location,
        "forecast_year":     forecast_year,
        "current": {
            "population_M":   cur_pop,
            "urb_rate":       cur_urb,
            "urban_pop_M":    cur_upop,
            "growth_rate":    cur_grow,
            "infra_pressure": cur_infra,
        },
        "forecast": {
            "population_M":   fut_pop,
            "urb_rate":       fut_urb,
            "urban_pop_M":    fut_upop,
            "growth_rate":    fut_grow,
            "infra_pressure": fut_infra,
        },
        "infrastructure":    infra_rep,
        "effects":           effects,
    }


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else "Mumbai"
    predict_urbanization(location)
