## Components

### 2. `prepare_dataset.py`
- `load_catalog(config) -> DataFrame` with columns: `id, name, description, image_path`.
- Cleans nulls/bad lines, drops rows whose image file is missing, builds `description`.

### 3. `clip_encoder.py`
- Loads `CLIPModel` + `CLIPProcessor` once; `fp16` on CUDA.
- `encode_images(list[PIL.Image]) -> np.ndarray (N, 512)` and `encode_text(list[str]) -> np.ndarray`.
- Both return **L2-normalized float32** vectors so cosine sim = dot product.
- Convert images to RGB before processing.

### 4. `build_index.py` (run on GPU machine)
- Load catalog, iterate in batches (use a `DataLoader` with workers for image decoding), encode, stack.
- Save `artifacts/embeddings.npy` (N x 512, float32) and `artifacts/ids.npy` (N,) in the **same row order**.
- Progress bar; skip + log unreadable images (keep ids/embeddings aligned).
- Command: `python -m engine.build_index`

### 5. `searcher.py`
- Loads `embeddings.npy`, `ids.npy`, catalog metadata once at startup.
- `search(vec, k=10)`: `scores = embeddings @ vec`, `np.argpartition` for top-k, then sort those k descending.
- `search_by_text(query, k)` -> encode text -> `search`.
- `search_by_image(pil_image, k)` -> encode image -> `search`.
- Returns list of `{id, name, description, image_url, score}`.
- Text query: start with raw query; optionally test `"a photo of {query}"` prompt and keep whichever gives better results.

### 6. FastAPI `app/main.py`
| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve `static/index.html` |
| `/static/*` | GET | JS / CSS |
| `/images/{id}.jpg` | GET | Product images (mount `engine/data/images` as static files) |
| `/api/search/text` | POST | JSON `{"query": "red dress"}` -> top 10 results |
| `/api/search/image` | POST | multipart `file` -> top 10 results |
| `/api/health` | GET | Index size + device, for quick check |

- Load `Searcher` once in a lifespan/startup handler (not per request).
- Validate: empty query -> 400; non-image or oversize upload (e.g. > 5 MB) -> 400/413; undecodable image -> 400.
- Sync `def` endpoints (model inference is blocking; FastAPI runs them in a threadpool) or wrap in `run_in_threadpool`.

### 7. Frontend (`static/`)
- One page: text input + Search button, and an image file picker (with thumbnail preview of the chosen image) + Search button.
- `fetch` to the two endpoints; render results as a list of cards: **image, product name, description** (score optional, small).
- States: loading, no results, error message.
- Escape all text inserted into the DOM (use `textContent`, not `innerHTML`).

## Workflow across the two machines

2. **GPU machine:** `python -m engine.build_index` -> produces `engine/artifacts/`.

## Build order

1. `requirements.txt` + `network_configs.yml`
2. `catalog.py` (check row count after filtering, print a few descriptions)
3. `clip_encoder.py` (smoke test: encode 1 text + 1 image, check shape 512 and norm ~1)
4. `build_index.py` (test on first ~500 rows before the full run)
5. `searcher.py` (test from a REPL)
6. `app/main.py` endpoints (test with curl)
7. Frontend
8. Full-index run on GPU machine, then end-to-end check

## Verification

- Text: "red dress", "brown shirt", "black sneakers", "silver watch" -> top results plausibly match category and colour.
- Image: upload a catalog image -> it appears as rank 1 with score ~1.0, neighbours are similar items.
- Image: upload an image **not** in the catalog (e.g. a phone photo) -> sensible neighbours.
- Always returns exactly 10 results (or fewer only if catalog < 10); each has image, name, description.
- Bad input: empty query, non-image upload, huge file -> clean error, no server crash.
- `ids.npy` length equals `embeddings.npy` rows equals catalog size after filtering.