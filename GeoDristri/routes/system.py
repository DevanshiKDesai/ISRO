from typing import Any

from fastapi import APIRouter

from services.model_registry import registry


router = APIRouter()


@router.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "running",
        "datasets": {
            "population": registry.df_pop is not None,
            "forest": registry.df_forest is not None,
            "crop": registry.df_crop is not None,
            "drought": registry.df_drought is not None,
        },
        "ml_models": registry.model_status,
        "version": "5.0",
    }


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ml_models": registry.model_status}

