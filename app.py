"""
app.py
------
Phase 3: Streamlit dashboard for the breast ultrasound + clinical data
multimodal risk decision-support prototype.

RESEARCH / EDUCATIONAL PROTOTYPE ONLY. Requires artifacts/model.pth,
artifacts/scaler.pkl, and artifacts/background_clinical.npy, all produced
by train.py.

Note on the clinical input form
--------------------------------
The trained model (see train.py) uses 4 clinical features derived from the
BrEaST-Lesions-USG dataset: age, a menopausal-status PROXY (age >= 50), 
family history, and palpable lump. There is no independent "menopausal
status" or "prior biopsy" field in that dataset, so this form does NOT ask
for them separately — doing so would imply they affect the model's output
when they don't. Menopausal status is computed automatically from the
age you enter; prior biopsy history is not used by this model at all
(see train.py docstring for why it was excluded rather than faked).
"""

import os
import uuid
import tempfile

import numpy as np
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as T

from train import CLINICAL_FEATURES, MultimodalBreastCancerNet  # noqa: F401 (kept for clarity/reference)
from utils import (
    DISCLAIMER_TEXT,
    load_model_and_scaler,
    mc_dropout_predict,
    risk_category,
    GradCAM,
    overlay_gradcam,
    compute_shap_values,
    plot_shap_bar,
    get_referral_recommendation,
    generate_pdf_report,
)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pth")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
BACKGROUND_PATH = os.path.join(ARTIFACTS_DIR, "background_clinical.npy")
IMAGE_SIZE = 224
DEVICE = "cpu"  # Hugging Face Spaces free tier is CPU-only


# --------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------
@st.cache_resource
def get_model_and_scaler():
    return load_model_and_scaler(MODEL_PATH, SCALER_PATH, device=DEVICE)


@st.cache_resource
def get_shap_background():
    return np.load(BACKGROUND_PATH)


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Same deterministic (non-augmented) preprocessing used for val/inference in train.py."""
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image.convert("RGB")).unsqueeze(0)  # (1, 3, H, W)


def build_clinical_raw(age: float, family_history: bool, palpable_lump: bool) -> np.ndarray:
    """
    Reproduce the exact feature engineering used in train.py:
    [age, menopausal_status_proxy, family_history, palpable_lump]
    """
    menopausal_status_proxy = 1.0 if age >= 50 else 0.0
    row = np.array([[
        float(age),
        menopausal_status_proxy,
        float(family_history),
        float(palpable_lump),
    ]], dtype=np.float32)
    assert row.shape[1] == len(CLINICAL_FEATURES)
    return row


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Breast US Risk Decision-Support (Prototype)", layout="wide")

st.title("Breast Ultrasound Risk Decision-Support \u2014 Research Prototype")
st.error(DISCLAIMER_TEXT)

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(BACKGROUND_PATH)):
    st.warning(
        "No trained model found in `artifacts/`. Run `train.py` first to produce "
        "`model.pth`, `scaler.pkl`, and `background_clinical.npy` before using this app."
    )
    st.stop()


# --------------------------------------------------------------------------
# Sidebar: patient inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
    family_history = st.radio("Family history of breast/ovarian cancer", ["No", "Yes"]) == "Yes"
    palpable_lump = st.radio("Palpable lump present", ["No", "Yes"]) == "Yes"

    st.caption(
        "Menopausal status is estimated automatically from age (a common clinical "
        "proxy: 50+ \u2192 postmenopausal) and is not a separate input, since the "
        "training dataset does not contain a real menopausal-status field. Prior "
        "biopsy history is not used by this model \u2014 see train.py notes."
    )

    location = st.text_input("Patient location / city (for referral guidance)", "")

    st.header("Ultrasound Scan")
    uploaded_image = st.file_uploader("Upload a breast ultrasound image", type=["png", "jpg", "jpeg"])

    with st.expander("Advanced settings"):
        mc_passes = st.slider("MC Dropout passes", min_value=5, max_value=30, value=10)
        shap_samples = st.slider("SHAP samples (higher = slower, more precise)",
                                  min_value=20, max_value=200, value=60, step=20)

    analyze_clicked = st.button("Analyze Scan", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
if analyze_clicked:
    if uploaded_image is None:
        st.warning("Please upload an ultrasound image before analyzing.")
        st.stop()

    model, scaler, checkpoint = get_model_and_scaler()
    background = get_shap_background()

    pil_image = Image.open(uploaded_image)
    image_tensor = preprocess_image(pil_image)
    clinical_raw = build_clinical_raw(age, family_history, palpable_lump)
    clinical_scaled = scaler.transform(clinical_raw)
    clinical_tensor = torch.tensor(clinical_scaled, dtype=torch.float32)

    with st.spinner("Running multimodal model (MC Dropout inference)..."):
        mc_result = mc_dropout_predict(model, image_tensor, clinical_tensor,
                                        n_passes=mc_passes, device=DEVICE)
    risk_level = risk_category(mc_result["mean_probability"])

    with st.spinner("Generating Grad-CAM explanation..."):
        gradcam = GradCAM(model)
        cam, _ = gradcam.generate(image_tensor, clinical_tensor, device=DEVICE)
        gradcam_overlay = overlay_gradcam(pil_image, cam, image_size=IMAGE_SIZE)

    with st.spinner("Computing SHAP clinical feature attribution (this can take a moment on CPU)..."):
        shap_values = compute_shap_values(
            model, scaler, image_tensor, clinical_raw.flatten(), background,
            device=DEVICE, n_samples=shap_samples,
        )

    referral = get_referral_recommendation(risk_level, location or None)

    # ---- Layout: images ----
    st.subheader("Scan & Grad-CAM Explanation")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(pil_image, caption="Original ultrasound", use_container_width=True)
    with img_col2:
        st.image(gradcam_overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)

    # ---- Layout: metrics ----
    st.subheader("Model Output")
    m1, m2, m3 = st.columns(3)
    m1.metric("Malignancy probability", f"{mc_result['mean_probability'] * 100:.1f}%")
    m2.metric("Model uncertainty", mc_result["uncertainty_level"],
              help=f"std={mc_result['std']:.3f} over {mc_result['n_passes']} passes")
    m3.metric("Risk category", risk_level)

    ci_low, ci_high = mc_result["ci_95"]
    st.caption(f"Approximate 95% range: {ci_low * 100:.1f}% \u2013 {ci_high * 100:.1f}%")

    # ---- Layout: SHAP ----
    st.subheader("Clinical Feature Contribution (SHAP)")
    shap_fig = plot_shap_bar(shap_values, feature_names=CLINICAL_FEATURES)
    st.pyplot(shap_fig)

    # ---- Layout: referral ----
    st.subheader("Referral Guidance")
    st.info(
        f"**Suggested action:** {referral['action']}  \n"
        f"**Specialist type:** {referral['specialist_type']}  \n"
        f"**Timeframe:** {referral['timeframe']}"
    )
    if referral.get("search_query"):
        st.caption(f"Suggested search: \u201c{referral['search_query']}\u201d \u2014 {referral['note']}")
    else:
        st.caption(referral["note"])

    # ---- PDF export ----
    st.subheader("Download Report")
    work_dir = tempfile.mkdtemp()
    run_id = uuid.uuid4().hex[:8]
    gradcam_path = os.path.join(work_dir, f"gradcam_{run_id}.png")
    shap_path = os.path.join(work_dir, f"shap_{run_id}.png")
    pdf_path = os.path.join(work_dir, f"report_{run_id}.pdf")

    gradcam_overlay.save(gradcam_path)
    shap_fig.savefig(shap_path, dpi=150)

    patient_meta = {
        "Age": age,
        "Family history": "Yes" if family_history else "No",
        "Palpable lump": "Yes" if palpable_lump else "No",
        "Location": location or "Not provided",
    }

    generate_pdf_report(
        pdf_path, patient_meta, mc_result, risk_level, referral,
        gradcam_image_path=gradcam_path, shap_plot_path=shap_path,
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF diagnostic summary",
            data=f.read(),
            file_name="breast_us_risk_report.pdf",
            mime="application/pdf",
        )

else:
    st.info("Fill in patient details and upload an ultrasound image in the sidebar, then click **Analyze Scan**.")
