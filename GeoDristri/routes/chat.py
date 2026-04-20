from typing import Any

from fastapi import APIRouter, Query, Response

from models.schemas import ChatRequest
from services.chat_service import DISPATCH, classify, smart_fallback
from services.export_service import dict_to_csv


router = APIRouter()


@router.post("/chat")
def chat_endpoint(req: ChatRequest, export: str = Query(default="json")) -> Any:
    msg = req.message.strip()
    intents = classify(msg)
    payload = {
        "reply": smart_fallback(msg) if intents == ["unknown"] else ("\n\n---\n\n".join(DISPATCH[i](msg) for i in intents[:2]) if len(intents) > 1 else DISPATCH[intents[0]](msg)),
        "intent": "unknown" if intents == ["unknown"] else intents[0],
    }
    if export.lower() == "csv":
        return Response(content=dict_to_csv(payload), media_type="text/csv")
    return payload

