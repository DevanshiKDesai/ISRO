from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class AOIRequest(BaseModel):
    lat: float
    lng: float
    north: float
    south: float
    east: float
    west: float
    aoi_name: str = "AOI"
    user_email: Optional[str] = None


class AlertEmailRequest(BaseModel):
    to_email: str
    aoi_name: str
    level: str
    summary: str
    lat: float
    lng: float

