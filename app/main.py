import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from engine.inference import DEFAULT_CONFIG, Searcher
from engine.scripts.prepare_dataset import load_config

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

cfg = load_config(DEFAULT_CONFIG)
state = {}


@asynccontextmanager
async def lifespan(app):
    state["searcher"] = Searcher(DEFAULT_CONFIG)
    yield
    state.clear()


app = FastAPI(title="Fashion Search", lifespan=lifespan)


class TextQuery(BaseModel):
    query: str = Field(max_length=200)


@app.get("/api/health")
def health():
    return {"status": "ok", "products": len(state["searcher"].ids)}


@app.post("/api/search/text")
def search_text(body: TextQuery):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty.")
    return {"results": state["searcher"].search_by_text(query)}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the image.")

    return {"results": state["searcher"].search_by_image(image)}


# mounted last so the /api routes win over the static catch-all
app.mount("/images", StaticFiles(directory=cfg["paths"]["images_dir"]), name="images")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
