"""
Streamlit demo — search a Kepler star, view phase-folded light curve, get CNN prediction.
Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.download import _configure_lightkurve_cache, download_light_curve
from src.predict import load_model, predict_from_cache, predict_light_curve, preprocess_from_arrays

st.set_page_config(
    page_title="Exoplanet Transit Detector",
    page_icon="🪐",
    layout="wide",
)

# Demo stars from downloaded KOI manifest
DEMO_STARS = {
    "K00752.01 — confirmed planet": 10797460,
    "K00755.01 — confirmed planet": 10854555,
    "K00754.01 — false positive": 10848459,
    "K00114.01 — false positive": 6721123,
}


@st.cache_resource
def get_model():
    ckpt = ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        return None, None, None, None
    return load_model(ckpt)


def plot_folded_curve(phase, flux, title: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(phase, flux, ".", markersize=1, alpha=0.5, color="#4fc3f7")
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_global_view(global_view: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(global_view, color="#81c784", linewidth=0.8)
    ax.set_xlabel("Phase bin")
    ax.set_ylabel("Scaled flux")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def main():
    st.title("Exoplanet Transit Detector")
    st.caption("1D CNN on phase-folded Kepler light curves · BLS + focal loss · real NASA data")

    model, device, cfg, seq_len = get_model()
    if model is None:
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select a star")
        demo_choice = st.selectbox("Demo targets", ["Custom KIC ID"] + list(DEMO_STARS.keys()))
        if demo_choice == "Custom KIC ID":
            kepid = st.number_input("Kepler ID (KIC)", min_value=1, value=11904151, step=1)
        else:
            kepid = DEMO_STARS[demo_choice]
            st.info(f"KIC {kepid}")

        fetch_live = st.checkbox("Fetch from MAST if not cached", value=False)
        run_btn = st.button("Analyze", type="primary")

    with col2:
        st.subheader("About the pipeline")
        st.markdown(
            """
            1. **Download** PDCSAP flux from NASA MAST (Kepler mission)
            2. **BLS periodogram** finds periodic transit-shaped dips
            3. **Phase-fold** at orbital period to stack every transit
            4. **1D CNN** classifies the 2048-point Global View
            """
        )

    if not run_btn:
        st.info("Select a star and click **Analyze**.")
        return

    cache_dir = ROOT / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _configure_lightkurve_cache(cache_dir)
    lc_path = cache_dir / f"KIC_{kepid}.pkl"

    with st.spinner("Fetching light curve..."):
        if fetch_live or not lc_path.exists():
            result = download_light_curve(int(kepid), cache_dir)
            if result["lc"] is None:
                st.error(f"Could not download light curve for KIC {kepid}: {result['status']}")
                return
            time = np.array(result["lc"]["time"])
            flux = np.array(result["lc"]["flux"])
        else:
            df = pd.read_pickle(lc_path)
            time = df["time"].values
            flux = df["flux"].values

    with st.spinner("Running BLS + CNN inference..."):
        try:
            pred = predict_from_cache(int(kepid))
            meta = {"phase": pred.get("phase"), "folded_flux": pred.get("folded_flux")}
            if pred.get("phase") is None:
                _, meta = preprocess_from_arrays(time, flux, cfg)
        except FileNotFoundError:
            pred = predict_light_curve(time, flux)
            _, meta = preprocess_from_arrays(time, flux, cfg)

    prob = pred["probability"]
    label = pred["prediction"]
    period = pred["period"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", label)
    m2.metric("P(planet)", f"{prob:.1%}")
    m3.metric("Orbital period", f"{period:.3f} d")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if meta.get("phase") is not None:
            st.pyplot(plot_folded_curve(meta["phase"], meta["folded_flux"], "Phase-folded light curve"))
    with c2:
        st.pyplot(plot_global_view(pred["global_view"], "CNN input (Global View, 2048 bins)"))

    if prob >= 0.5:
        st.success(
            f"CNN detected a transit-like signal (confidence {prob:.1%}). "
            "Periodic flux dips are consistent with a planet crossing the star."
        )
    else:
        st.warning(
            f"No strong transit signal detected (confidence {prob:.1%}). "
            "Flux variations may be stellar activity, eclipsing binary, or noise."
        )


if __name__ == "__main__":
    main()
