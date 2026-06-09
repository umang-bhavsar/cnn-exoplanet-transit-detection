# Exoplanet Transit Detector

Trained a **1D CNN** to detect tiny brightness dips in real **NASA Kepler** light curves — the signature of a planet transiting its star.

## Pipeline

```
KOI catalog (NASA Exoplanet Archive)
    → lightkurve download (MAST)
    → detrend + BLS periodogram
    → phase-fold at orbital period
    → resample to 2048-point Global View
    → 1D CNN + focal loss
    → Streamlit demo
```

## Quick start

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (~30–60 min depending on download speed)
bash scripts/run_pipeline.sh

# Or step by step
python -m src.download
python -m src.preprocess
python -m src.train
python -m src.evaluate

# Interactive demo
streamlit run app/streamlit_app.py
```

## Project structure

```
stars/
├── configs/default.yaml    # hyperparameters & data limits
├── src/
│   ├── download.py         # KOI catalog + lightkurve caching
│   ├── preprocess.py       # BLS, phase-fold, resample
│   ├── model.py            # 1D CNN
│   ├── losses.py           # focal loss
│   ├── train.py            # training loop
│   ├── evaluate.py         # PR-AUC, confusion matrix
│   └── predict.py          # single-star inference
├── app/streamlit_app.py    # demo UI
└── data/                   # gitignored — downloaded at runtime
```

## Data sources

| Source | URL | Use |
|--------|-----|-----|
| KOI catalog | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Labels: CONFIRMED vs FALSE POSITIVE |
| Light curves | [MAST via lightkurve](https://lightkurve.github.io/) | PDCSAP flux time series |

No API key required.

## Key design choices

| Choice | Why |
|--------|-----|
| **Phase folding** | Stacks every orbit so transit dips align and add constructively |
| **BLS periodogram** | Same box-shaped transit search astronomers use (Astropy BLS) |
| **Global View (2048 bins)** | Fixed-length CNN input; NASA DL pipeline convention |
| **Focal loss** | Down-weights easy negatives; critical at <1% positive rate |
| **Star-level splits** | No data leakage — all orbits from one star stay in one fold |
| **PR-AUC** | Proper metric for imbalanced detection (not accuracy) |

## Results (80-star subset: 40 confirmed, 40 false positives)

| Model | Test PR-AUC | Notes |
|-------|-------------|-------|
| **1D CNN + focal loss** | **0.78** | Raw phase-folded light curves |
| SMOTE + GradientBoosting | 0.67 | Tabular features (period, depth, std) |

Metrics intentionally use **PR-AUC** and **recall** — not accuracy — because class balance is ~50/50 here but real sky surveys are <<1% positive.

## Scaling up

Edit `configs/default.yaml`:

```yaml
download:
  max_per_class: 80    # default 40
  max_total: 500       # default 200
train:
  epochs: 60
```

## Interview talking points

1. **Transit method** — planet blocks ~0.01–1% of starlight; periodic U-shaped dips.
2. **Class imbalance** — most stars have no detectable transits; accuracy is misleading.
3. **False positives** — eclipsing binaries mimic transits; model is one vetting stage.
4. **Focal loss** — focuses learning on hard/rare positives (Lin et al. 2017).
5. **BLS + fold** — domain-informed preprocessing, not label leakage.

## License

MIT — NASA data is public domain.
