## Architecture

Offline (once, GPU machine)
44k images ──► CLIP image encoder ──► 44k × 512 vectors ──► embeddings.npy
styles.csv ──► name + description (plain text, NOT embedded) ──► looked up by id

Online (per search)
text query  ──► CLIP text encoder  ──┐
                                     ├─► 512-d vector ─► dot product vs 44k image vectors
uploaded img ─► CLIP image encoder ──┘                   ─► top 10 indices ─► ids
                                                         ─► join with CSV: image, name, description

## Embedding Process
Only the images get embedded. CSV metadata (name, description) is not sent through CLIP. It is only used to build the result cards, joined by id after the top 10 are found.

## Indexing Strategy
No index structure (flat storage). Vectors are just stored as one matrix in `embeddings.npy`. Why: the dataset is small (~44k vectors, ~90 MB), so an index like IVF/HNSW/FAISS gives no real benefit.

## Search Method
Brute-force cosine similarity (`embeddings @ query_vec`, then top 10). Why:
- Dataset is small, so a search takes well under a second.
- No loss of precision. Approximate indexes can miss true neighbours.
- Brute-force compares the query against every record, so the top 10 are guaranteed to be the exact best matches.

## Why cross-modal works: 
CLIP trains text and image encoders into one shared space. "red dress" text vector lands near red dress image vectors. So text query is compared against image vectors directly, and image query is compared against image vectors the same way. Vectors are L2-normalized, so dot product equals cosine similarity.

## Optional upgrade
Also embed the product name text and blend scores (e.g. 0.7*img_sim + 0.3*name_sim). Might help on brand or attribute queries, but it's not needed for the task.