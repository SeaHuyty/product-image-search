import argparse
from pathlib import Path

import pandas as pd 
import yaml

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_styles(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.dropna(subset=["productDisplayName"])
    df["id"] = df["id"].astype(int)

    return df.reset_index(drop=True)


def build_description(row):
    def join(cols):
        vals = [row[c] for c in cols if pd.notna(row[c])]
        return " ".join(str(int(v)) if isinstance(v, float) else str(v) for v in vals)

        head = join(["gender", "usage", "baseColour", "articleType"])
        tail = ", ".join(x for x in [join(["subCategory"]), join(["season", "year"])] if x)
        
        return f"{head} - {tail}" if tail else head


def attach_image(df, images_dir):
    images_dir = Path(images_dir)

    df["image_path"] = df["id"].map(lambda i: str(images_dir / f"{i}.jpg"))
    exists = df["image_path"].map(lambda p: Path(p).exists())

    print(f"Dropped {(~exists).sum()} rows with no image")

    return df[exists].reset_index(drop=True)


def load_catalog(cfg):
    paths = cfg["paths"]
    df = read_styles(paths["styles_csv"])
    df = attach_image(df, paths["images_dir"])
    df["description"] = df.apply(build_description, axis=1)
    df = df.rename(columns={"productionDisplayName": "name"})

    return df[["id", "name", "description", "image_path"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="engine/configs/network_configs.yml")
    args = parser.parse_args()

    catalog = load_catalog(load_config(args.config))

    print(f"{len(catalog)} products")
    print(catalog.sample(5, random_state=0).to_string())