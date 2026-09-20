import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.networks import CLIPEncoder
from engine.scripts.prepare_dataset import load_catalog, load_config

DEFAULT_CONFIG = "engine/configs/network_configs.yml"


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return i, Image.open(self.paths[i]).convert("RGB")
        except Exception:
            return i, None


def collate(batch):
    return batch


def build_embeddings(encoder, paths, batch_size, num_workers):
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )
    vectors, kept = [], []
    for batch in tqdm(loader, desc="Encoding"):
        good = [(i, img) for i, img in batch if img is not None]

        if not good:
            continue

        idx, images = zip(*good)
        vectors.append(encoder.encode_images(list(images)))
        kept.extend(idx)

    print(f"encoded {len(kept)} / {len(paths)} images")
    return np.concatenate(vectors), np.array(kept)


def build_index(cfg, limit=None, batch_size=None, num_workers=4):
    model_cfg = cfg["model"]
    batch_size = batch_size or model_cfg["batch_size"]

    catalog = load_catalog(cfg)
    if limit:
        catalog = catalog.head(limit)

    encoder = CLIPEncoder(model_cfg["base_model"], model_cfg["device"])
    embeddings, kept = build_embeddings(
        encoder, catalog["image_path"].tolist(), batch_size, num_workers
    )

    out_dir = Path(cfg["paths"]["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "ids.npy", catalog["id"].to_numpy()[kept])
    print(f"saved {embeddings.shape} to {out_dir}")


class Searcher:
    def __init__(self, config_path=DEFAULT_CONFIG):
        cfg = load_config(config_path)
        model_cfg = cfg["model"]
        self.top_k = model_cfg["top_k"]

        artifacts = Path(cfg["paths"]["artifacts_dir"])
        self.embeddings = np.load(artifacts / "embeddings.npy")
        self.ids = np.load(artifacts / "ids.npy")
        self.catalog = load_catalog(cfg).set_index("id")
        self.encoder = CLIPEncoder(model_cfg["base_model"], model_cfg["device"])

    def search(self, query_vec, k=None):
        k = min(k or self.top_k, len(self.ids))
        scores = self.embeddings @ query_vec
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        results = []
        for j in top:
            pid = int(self.ids[j])
            row = self.catalog.loc[pid]
            results.append({
                "id": pid,
                "name": row["name"],
                "description": row["description"],
                "image_url": f"/images/{pid}.jpg",
                "score": float(scores[j]),
            })
        return results

    def search_by_text(self, query, k=None):
        return self.search(self.encoder.encode_text(query)[0], k)

    def search_by_image(self, image, k=None):
        return self.search(self.encoder.encode_images([image])[0], k)


def print_results(results):
    for rank, r in enumerate(results, 1):
        print(f"{rank:>2}. {r['score']:.3f}  {r['id']}  {r['name']}  |  {r['description']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="embed all images and save the index")
    build.add_argument("--limit", type=int, default=None, help="only index the first N products")
    build.add_argument("--batch-size", type=int, default=None)
    build.add_argument("--num-workers", type=int, default=4)

    search = sub.add_parser("search", help="search the index by text or image")
    search.add_argument("query", nargs="?", help="text query, e.g. 'red dress'")
    search.add_argument("--image", help="path to a query image")
    search.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()

    if args.command == "build":
        build_index(load_config(args.config), args.limit, args.batch_size, args.num_workers)
    else:
        if bool(args.query) == bool(args.image):
            parser.error("search needs exactly one of: a text query or --image")
        searcher = Searcher(args.config)
        if args.image:
            results = searcher.search_by_image(Image.open(args.image), args.top_k)
        else:
            results = searcher.search_by_text(args.query, args.top_k)
        print_results(results)


if __name__ == "__main__":
    main()
