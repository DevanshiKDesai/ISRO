from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.aoi import router as aoi_router
from routes.chat import router as chat_router
from routes.system import router as system_router
from routes.tools import router as tools_router


app = FastAPI(title="GeoDhrishti API", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(system_router)
app.include_router(chat_router)
app.include_router(aoi_router)
app.include_router(tools_router)

