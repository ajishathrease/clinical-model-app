# Breast Ultrasound Risk Decision-Support — Research Prototype

**RESEARCH / EDUCATIONAL PROTOTYPE ONLY — NOT a validated medical device.**
This is a final-year student project demonstrating a multimodal (image +
clinical data) deep learning pipeline with uncertainty estimation and
explainability. It must not be used for real clinical decision-making.

## Model

Multimodal fusion of an EfficientNet-B0 ultrasound image encoder and a
small MLP over 4 clinical features (age, a menopausal-status proxy derived
from age, family history, palpable lump), trained with Monte Carlo Dropout
for uncertainty estimation. See `train.py` for full details, including
documented limitations of the clinical feature engineering.

## Dataset

Trained on the **BrEaST-Lesions-USG** dataset (Pawłowska et al., *Scientific
Data*, 2024), 256 breast ultrasound scans from 256 patients, released under
a **CC-BY 4.0** license. Please cite the original paper if you build on this
work; the raw dataset is not committed to this repo (see `.gitignore`) —
only the trained artifacts under `artifacts/` are.

## Repository structure

```
.
├── app.py                       # Streamlit dashboard
├── train.py                     # training script
├── utils.py                     # MC Dropout, Grad-CAM, SHAP, PDF, referral logic
├── requirements.txt
├── .gitignore                   # excludes data/ (raw dataset) and __pycache__
└── artifacts/
    ├── model.pth                # trained weights (required)
    ├── scaler.pkl                # fitted clinical StandardScaler (required)
    └── background_clinical.npy  # SHAP background sample (required)
```

## Running locally

```bash
pip install -r requirements.txt

# 1) Train (produces artifacts/model.pth, scaler.pkl, background_clinical.npy)
python train.py --clinical_xlsx data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx --image_dir data/images

# 2) Launch the app
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub, **including** the `artifacts/` folder (the
   three files above) but **not** the raw `data/` folder — it's excluded
   via `.gitignore` since redistributing the raw dataset isn't necessary
   for the app to run.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Select this repo/branch and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud auto-installs `requirements.txt`.
5. Free-tier notes: ~1GB RAM, app sleeps after ~12h of inactivity (wakes
   automatically on next visit, just slower to load), free-tier repos
   should be public (private is allowed but limited to one app).

If the app hits the memory ceiling (SHAP + EfficientNet + Streamlit can
add up), reduce the default MC Dropout passes / SHAP sample count in
`app.py`'s "Advanced settings" or trim the background sample size in
`train.py`.
