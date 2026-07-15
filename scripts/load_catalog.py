"""
Load the enriched catalog into Supabase.

Run once from the project root:
    python scripts/load_catalog.py

What it does:
  1. reads data/catalog.json (the enriched one)
  2. cleans dimensions (drop zeros / oversize): same as data notebook cell 35
  3. computes S/M/L size band by volume, per category (qcut)
  4. matches each product's (category, size_band) to the models table
  5. pushes everything to the products table
"""

import json
import sys
import pandas as pd

# src is a sibling of scripts/, so add project root to the path
sys.path.append("src")
from config import supabase  


CATALOG_PATH = "data/catalog.json"
MAX_MM = 3000   # anything bigger is bad data (matches the notebook)


def clean_dimensions(df):
    """Split into measured and unmeasured. Clean the measured ones."""
    has_dims = df.dropna(subset=["width_mm", "depth_mm", "height_mm"]).copy()

    # drop zeros and impossible sizes -- data notebook cell 35
    good = (has_dims[["width_mm", "depth_mm", "height_mm"]] > 0).all(axis=1)
    small = (has_dims[["width_mm", "depth_mm", "height_mm"]] <= MAX_MM).all(axis=1)
    measured = has_dims[good & small].copy()

    # everything the measured set dropped stays, but with no dimensions to band on
    unmeasured = df[~df["id"].isin(measured["id"])].copy()
    return measured, unmeasured


def add_size_bands(measured):
    """Tag S/M/L by volume within each category. Fallback to M if too few to split."""
    measured["volume"] = (
        measured["width_mm"] * measured["depth_mm"] * measured["height_mm"]
    )

    def band(group):
        if len(group) < 3:
            # can't split fewer than 3 into 3 groups -- label them M
            return pd.Series(["M"] * len(group), index=group.index)
        return pd.qcut(group["volume"], 3, labels=["S", "M", "L"])

    measured["size_band"] = (
        measured.groupby("category", group_keys=False).apply(band)
    )
    return measured


WARDROBE_KEYWORDS = ("wardrobe", "armoire", "armario", "closet")


def load_model_map():
    """Pull the models so we can match products to them.
    Keyed by (model_category, size_band), where model_category is the
    label used in the models table: bed / chair / dresser / wardrobe."""
    rows = supabase.table("models").select("id, category, size_band").execute().data
    return {(r["category"], r["size_band"]): r["id"] for r in rows}


def model_category_for(row):
    """Which model bucket does this product belong to, if any?

    Most categories map straight through. 'cabinet' is a mixed bag -- only
    the actual wardrobes should get the wardrobe model, so we check the
    name and style for a keyword. Everything else gets no model (None)."""
    cat = row["category"]

    if cat in ("bed", "chair", "dresser", "desk", "shelf"):
        return cat

    if cat == "cabinet":
        blob = f"{row.get('name') or ''} {row.get('style') or ''}".lower()
        if any(k in blob for k in WARDROBE_KEYWORDS):
            return "wardrobe"

    return None   # no model for this product yet


def to_record(row, model_map):
    """Turn one dataframe row into a dict the products table accepts."""
    size_band = row.get("size_band")
    if pd.isna(size_band):
        size_band = None

    # figure out the model bucket, then look up the id for that bucket + size
    model_cat = model_category_for(row)
    model_id = model_map.get((model_cat, size_band)) if model_cat else None

    def clean(v):
        # pandas NaN -> None so Postgres stores null, not the string "nan"
        return None if pd.isna(v) else v

    def clean_int(v):
        return None if pd.isna(v) else int(v)

    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "brand": clean(row.get("brand")),
        "color": clean(row.get("color")),
        "style": clean(row.get("style")),
        "width_mm": clean_int(row.get("width_mm")),
        "depth_mm": clean_int(row.get("depth_mm")),
        "height_mm": clean_int(row.get("height_mm")),
        "volume_mm3": clean_int(row.get("volume_mm3")),
        "footprint_mm2": clean_int(row.get("footprint_mm2")),
        "size_band": size_band,
        "storage_score": clean_int(row.get("storage_score")),
        "room_types": row.get("room_types") or [],
        "storage_indicator": row.get("storage_indicator") or {},
        "description": None,       # AI-filled later
        "image_url": None,
        "model_id": model_id,
        "embedding_index": None,   # filled when we do embeddings
    }


def main():
    df = pd.read_json(CATALOG_PATH)
    print(f"read {len(df)} products from {CATALOG_PATH}")

    measured, unmeasured = clean_dimensions(df)
    print(f"  measured: {len(measured)}   unmeasured: {len(unmeasured)}")

    measured = add_size_bands(measured)
    print("  size bands per category:")
    print(measured.groupby("category")["size_band"].value_counts().to_string())

    # unmeasured rows get no band
    unmeasured["size_band"] = None

    model_map = load_model_map()
    print(f"  loaded {len(model_map)} models to match against")

    all_rows = pd.concat([measured, unmeasured], ignore_index=True)
    records = [to_record(r, model_map) for _, r in all_rows.iterrows()]

    matched = sum(1 for r in records if r["model_id"] is not None)
    print(f"  {matched}/{len(records)} products matched to a 3D model")

    # upsert so re-running doesn't create duplicates
    supabase.table("products").upsert(records).execute()
    print(f"done. {len(records)} products in Supabase.")


if __name__ == "__main__":
    main()