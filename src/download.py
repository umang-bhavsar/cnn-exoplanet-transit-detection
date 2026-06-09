"""
Download KOI labels from NASA Exoplanet Archive and Kepler light curves via lightkurve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from tqdm import tqdm

from src.config import load_config


def fetch_koi_catalog(output_path: Path) -> pd.DataFrame:
    """Query the cumulative KOI table from NASA Exoplanet Archive."""
    print("Fetching KOI catalog from NASA Exoplanet Archive...")
    table = NasaExoplanetArchive.query_criteria(table="cumulative")
    df = table.to_pandas()

    keep_cols = [
        "kepid",
        "kepoi_name",
        "koi_disposition",
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_time0bk",
        "koi_quarters",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    df["kepid"] = pd.to_numeric(df["kepid"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["kepid"])
    df["kepid"] = df["kepid"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} KOI rows → {output_path}")
    return df


def build_download_manifest(
    catalog: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    max_per_class: int,
    max_total: int,
) -> pd.DataFrame:
    """Select balanced-ish subset of confirmed planets vs false positives."""
    pos = catalog[catalog["koi_disposition"] == positive_label].drop_duplicates("kepid")
    neg = catalog[catalog["koi_disposition"] == negative_label].drop_duplicates("kepid")

    pos = pos.head(max_per_class)
    neg = neg.head(max_per_class)

    pos = pos.assign(label=1)
    neg = neg.assign(label=0)

    manifest = pd.concat([pos, neg], ignore_index=True)
    manifest = manifest.drop_duplicates("kepid").head(max_total)
    return manifest


def _configure_lightkurve_cache(cache_dir: Path) -> None:
    """Use project-local cache so downloads work in sandboxed/CI environments."""
    import lightkurve as lk

    lk_cache = cache_dir / ".lightkurve_cache"
    lk_cache.mkdir(parents=True, exist_ok=True)
    lk.conf.cache_dir = str(lk_cache)


def download_light_curve(kepid: int, cache_dir: Path, mission: str = "Kepler") -> dict:
    """Download and cache a single star's stitched light curve."""
    import lightkurve as lk

    _configure_lightkurve_cache(cache_dir)

    cache_file = cache_dir / f"KIC_{kepid}.pkl"
    meta_file = cache_dir / f"KIC_{kepid}.json"

    if cache_file.exists() and meta_file.exists():
        lc = pd.read_pickle(cache_file)
        with meta_file.open() as f:
            meta = json.load(f)
        return {"kepid": kepid, "status": "cached", "meta": meta, "lc": lc}

    try:
        search = lk.search_lightcurve(f"KIC {kepid}", mission=mission)
        if len(search) == 0:
            return {"kepid": kepid, "status": "not_found", "meta": {}, "lc": None}

        collection = search.download_all()
        lc = collection.stitch()

        flux = lc.pdcsap_flux if lc.pdcsap_flux is not None else lc.sap_flux
        if flux is None:
            return {"kepid": kepid, "status": "no_flux", "meta": {}, "lc": None}

        record = {
            "time": np.asarray(lc.time.value, dtype=np.float64),
            "flux": np.asarray(flux.value, dtype=np.float64),
        }
        meta = {
            "kepid": kepid,
            "n_points": len(record["time"]),
            "mission": mission,
            "flux_type": "pdcsap" if lc.pdcsap_flux is not None else "sap",
        }

        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(record).to_pickle(cache_file)
        with meta_file.open("w") as f:
            json.dump(meta, f)

        time.sleep(0.3)  # polite rate limiting for MAST
        return {"kepid": kepid, "status": "downloaded", "meta": meta, "lc": record}

    except Exception as exc:
        return {"kepid": kepid, "status": f"error:{exc}", "meta": {}, "lc": None}


def run_download(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    dl = cfg["download"]
    cache_dir = Path(dl["cache_dir"])
    _configure_lightkurve_cache(cache_dir)

    catalog_path = cache_dir.parent / "catalogs" / "koi.csv"
    if not catalog_path.exists():
        catalog = fetch_koi_catalog(catalog_path)
    else:
        catalog = pd.read_csv(catalog_path)

    manifest = build_download_manifest(
        catalog,
        positive_label=dl["positive_label"],
        negative_label=dl["negative_label"],
        max_per_class=dl["max_per_class"],
        max_total=dl["max_total"],
    )

    results = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Downloading light curves"):
        result = download_light_curve(
            kepid=int(row["kepid"]),
            cache_dir=cache_dir,
            mission=cfg["data"]["mission"],
        )
        results.append(
            {
                "kepid": row["kepid"],
                "kepoi_name": row.get("kepoi_name", ""),
                "label": int(row["label"]),
                "koi_disposition": row.get("koi_disposition", ""),
                "koi_period": row.get("koi_period"),
                "koi_duration": row.get("koi_duration"),
                "koi_depth": row.get("koi_depth"),
                "koi_time0bk": row.get("koi_time0bk"),
                "download_status": result["status"],
            }
        )

    manifest_df = pd.DataFrame(results)
    manifest_path = Path(dl["cache_dir"]).parent / "catalogs" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)

    ok = manifest_df["download_status"].isin(["downloaded", "cached"]).sum()
    print(f"Download complete: {ok}/{len(manifest_df)} light curves available")
    print(f"Manifest → {manifest_path}")
    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kepler light curves for KOI targets")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()
    run_download(args.config)


if __name__ == "__main__":
    main()
