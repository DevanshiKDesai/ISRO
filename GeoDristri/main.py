"""
Backward-compatible entrypoint for existing deployments.
"""

from app import app
from services.prediction_service import unified_predict

__all__ = ["app", "unified_predict"]
