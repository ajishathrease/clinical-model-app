"""
utils.py
--------
Phase 2: Core medical/XAI utility functions used by app.py.

RESEARCH / EDUCATIONAL PROTOTYPE ONLY. See train.py docstring for dataset
and modelling caveats (this file assumes the 4-feature clinical model
trained by train.py on the BrEaST-Lesions-USG dataset).

Contents
--------
1. load_model_and_scaler   - load a trained checkpoint + clinical scaler
2. mc_dropout_predict       - Monte Carlo Dropout uncertainty estimation
3. risk_category            - map a probability to a Low/Moderate/High label
4. GradCAM                  - Grad-CAM heatmap generation + overlay helper
5. compute_shap_values      - SHAP attribution over the 4 clinical features
6. plot_shap_bar            - bar chart of SHAP attributions
7. get_referral_recommendation - risk-based referral guidance (NOT a real
   hospital directory — see function docstring)
8. generate_pdf_report      - assemble everything into a downloadable PDF
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import joblib

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)

from train import MultimodalBreastCancerNet, CLINICAL_FEATURES


DISCLAIMER_TEXT = (
    "RESEARCH / EDUCATIONAL PROTOTYPE ONLY \u2014 NOT a validated medical device. "
    "This output must not be used for real clinical decision-making and does not "
    "replace evaluation by a qualified healthcare professional."
)


# --------------------------------------------------------------------------
# 1. Model / scaler loading
# --------------------------------------------------------------------------
def load_model_and_scaler(model_path: str = "artifacts/model.pth",
                           scaler_path: str = "artifacts/scaler.pkl",
                           device: str = "cpu"):
    """Load the trained multimodal model and the fitted clinical scaler."""
    checkpoint = torch.load(model_path, map_location=device)

    model = MultimodalBreastCancerNet(
        backbone_name=checkpoint["backbone"],
        dropout_p=checkpoint["dropout_p"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, checkpoint


# --------------------------------------------------------------------------
# 2. Monte Carlo Dropout uncertainty estimation
# --------------------------------------------------------------------------
def mc_dropout_predict(model, image_tensor: torch.Tensor, clinical_tensor: torch.Tensor,
                        n_passes: int = 10, device: str = "cpu") -> dict:
    """
    Run N stochastic forward passes with dropout ACTIVE (model.train()) to
    estimate predictive uncertainty via Monte Carlo Dropout.

    image_tensor / clinical_tensor: single-sample batches, shape (1, ...).

    Returns a dict with mean probability, std, an approximate 95% CI, and a
    qualitative Low/Moderate/High uncertainty label. This is an approximate,
    illustrative uncertainty measure — NOT a calibrated clinical confidence
    interval.
    """
    model.train()  # keep dropout layers active for MC sampling
    image_tensor = image_tensor.to(device)
    clinical_tensor = clinical_tensor.to(device)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logit = model(image_tensor, clinical_tensor)
            prob = torch.sigmoid(logit)
            probs.append(prob.cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    mean_prob = float(np.mean(probs))
    std_prob = float(np.std(probs))
    ci_low = max(0.0, mean_prob - 1.96 * std_prob)
    ci_high = min(1.0, mean_prob + 1.96 * std_prob)

    if std_prob < 0.05:
        uncertainty_level = "Low"
    elif std_prob < 0.15:
        uncertainty_level = "Moderate"
    else:
        uncertainty_level = "High"

    model.eval()  # restore deterministic mode for any subsequent calls

    return {
        "mean_probability": mean_prob,
        "std": std_prob,
        "ci_95": (ci_low, ci_high),
        "uncertainty_level": uncertainty_level,
        "n_passes": n_passes,
        "raw_probs": probs.tolist(),
    }


def risk_category(mean_probability: float) -> str:
    """
    Map a mean predicted probability to a coarse Low/Moderate/High risk
    label. These thresholds (0.3 / 0.6) are illustrative defaults for this
    student prototype, NOT a clinically validated cutoff (e.g. not BI-RADS
    aligned) — tune them against your own validation results and say so in
    your report.
    """
    if mean_probability < 0.30:
        return "Low"
    elif mean_probability < 0.60:
        return "Moderate"
    else:
        return "High"


# --------------------------------------------------------------------------
# 3. Grad-CAM
# --------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM over the final convolutional block of the EfficientNet
    backbone (`model.image_backbone.features[-1]`), which is the last
    feature map before global pooling.
    """

    def __init__(self, model: MultimodalBreastCancerNet, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.image_backbone.features[-1]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, clinical_tensor: torch.Tensor,
                 device: str = "cpu"):
        """
        Returns (cam, malignancy_probability) where cam is a normalized
        [0, 1] 2D numpy array the same spatial size as the target layer's
        activation map (needs resizing to the display image size).
        """
        self.model.eval()
        image_tensor = image_tensor.clone().to(device).requires_grad_(True)
        clinical_tensor = clinical_tensor.to(device)

        logit = self.model(image_tensor, clinical_tensor)
        prob = torch.sigmoid(logit)

        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # global-average-pool gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, float(prob.item())


def overlay_gradcam(original_image: Image.Image, cam: np.ndarray,
                     image_size: int = 224, alpha: float = 0.4) -> Image.Image:
    """Resize the CAM to `image_size`, colorize it, and blend with the original scan."""
    cam_resized = cv2.resize(cam, (image_size, image_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    base = np.array(original_image.resize((image_size, image_size)).convert("RGB"))
    overlay = (alpha * heatmap + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(overlay)


# --------------------------------------------------------------------------
# 4. SHAP feature attribution (clinical branch)
# --------------------------------------------------------------------------
def compute_shap_values(model, scaler, image_tensor: torch.Tensor,
                         clinical_raw_row: np.ndarray, background_clinical_raw: np.ndarray,
                         device: str = "cpu", n_samples: int = 100):
    """
    Explain the clinical branch's contribution to the malignancy probability
    for ONE patient case, holding that patient's image fixed.

    clinical_raw_row       : shape (len(CLINICAL_FEATURES),), UNSCALED values
    background_clinical_raw: shape (n_background, len(CLINICAL_FEATURES)), UNSCALED
                              -- keep this SMALL (e.g. 20-50 rows sampled
                              from the training set); each SHAP sample
                              triggers a full forward pass through the
                              image backbone, which is comparatively slow.

    Returns a 1D numpy array of SHAP values aligned with CLINICAL_FEATURES.
    """
    import shap  # imported lazily - only needed when explaining a prediction

    def predict_fn(clinical_batch_raw: np.ndarray) -> np.ndarray:
        clinical_batch_raw = np.asarray(clinical_batch_raw, dtype=np.float32)
        clinical_scaled = scaler.transform(clinical_batch_raw)
        clinical_batch_tensor = torch.tensor(clinical_scaled, dtype=torch.float32, device=device)

        batch_size = clinical_batch_tensor.shape[0]
        image_batch = image_tensor.to(device).repeat(batch_size, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            logits = model(image_batch, clinical_batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(predict_fn, background_clinical_raw)
    shap_values = explainer.shap_values(clinical_raw_row.reshape(1, -1), nsamples=n_samples)

    return np.array(shap_values).flatten()


def plot_shap_bar(shap_values: np.ndarray, feature_names=None, out_path: str = None):
    """Horizontal bar chart of SHAP attributions, sorted by magnitude."""
    feature_names = feature_names or CLINICAL_FEATURES
    values = np.array(shap_values).flatten()

    order = np.argsort(np.abs(values))[::-1]
    labels = [feature_names[i] for i in order]
    vals = values[order]
    colors_ = ["#d62728" if v > 0 else "#1f77b4" for v in vals]  # red=pushes toward malignant

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels[::-1], vals[::-1], color=colors_[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on malignancy probability)")
    ax.set_title("Clinical Feature Contribution")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 5. Referral guidance
# --------------------------------------------------------------------------
# NOTE: This is NOT a real, verified hospital/clinic directory. Hard-coding
# specific institution names here would risk presenting fabricated or
# outdated facilities as real medical referrals, which is unsafe. Instead,
# this returns a generic action + specialist type + a ready-to-use search
# query string. Wire the search_query up to a real geocoding/places API
# (e.g. Google Places) in app.py if you want actual nearby facility names.
REFERRAL_GUIDANCE = {
    "Low": {
        "action": "Routine follow-up",
        "specialist_type": "Primary care physician / routine screening program",
        "timeframe": "Continue routine screening as advised by a clinician",
    },
    "Moderate": {
        "action": "Further diagnostic imaging recommended",
        "specialist_type": "Breast imaging radiologist / breast health clinic",
        "timeframe": "Short-interval follow-up imaging, timing to be confirmed by a clinician",
    },
    "High": {
        "action": "Prompt specialist referral recommended",
        "specialist_type": "Breast surgeon / oncologist for diagnostic work-up (e.g. biopsy)",
        "timeframe": "Discuss urgent scheduling with a healthcare provider as soon as possible",
    },
}


def get_referral_recommendation(risk_level: str, location: str = None) -> dict:
    """
    Return risk-based referral guidance. Does NOT return real hospital
    names — see module note above.
    """
    guidance = dict(REFERRAL_GUIDANCE.get(risk_level, REFERRAL_GUIDANCE["Moderate"]))
    guidance["risk_level"] = risk_level

    if location:
        guidance["search_query"] = f"{guidance['specialist_type']} near {location}"
        guidance["note"] = (
            "This prototype does not maintain a verified hospital directory. "
            "Use the search_query above with a maps/places service to find actual "
            "nearby facilities; do not treat this as a verified referral list."
        )
    else:
        guidance["search_query"] = None
        guidance["note"] = "No location provided \u2014 cannot suggest a search query."

    return guidance


# --------------------------------------------------------------------------
# 6. PDF report generation
# --------------------------------------------------------------------------
def generate_pdf_report(output_path: str, patient_meta: dict, mc_result: dict,
                         risk_level: str, referral: dict,
                         gradcam_image_path: str = None, shap_plot_path: str = None) -> str:
    """
    Assemble a downloadable PDF diagnostic summary.

    patient_meta : dict of display fields, e.g. {"Age": 45, "Location": "..."}
    mc_result    : output of mc_dropout_predict()
    risk_level   : output of risk_category()
    referral     : output of get_referral_recommendation()
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], textColor=colors.red, fontSize=9
    )

    story = [
        Paragraph("Breast Ultrasound Risk Decision-Support Report", styles["Title"]),
        Paragraph(DISCLAIMER_TEXT, disclaimer_style),
        Spacer(1, 0.5 * cm),
    ]

    # --- Patient metadata table ---
    meta_rows = [[str(k), str(v)] for k, v in patient_meta.items()]
    meta_table = Table([["Field", "Value"]] + meta_rows, colWidths=[6 * cm, 9 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a4a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # --- Model output ---
    story.append(Paragraph("Model Output", styles["Heading2"]))
    ci_low, ci_high = mc_result["ci_95"]
    story.append(Paragraph(
        f"Mean estimated malignancy probability: {mc_result['mean_probability'] * 100:.1f}% "
        f"(approx. 95% range: {ci_low * 100:.1f}%\u2013{ci_high * 100:.1f}%)",
        styles["Normal"],
    ))
    story.append(Paragraph(
        f"Model uncertainty level: {mc_result['uncertainty_level']} "
        f"(std over {mc_result['n_passes']} stochastic passes: {mc_result['std']:.3f})",
        styles["Normal"],
    ))
    story.append(Paragraph(f"Risk category: {risk_level}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # --- Grad-CAM ---
    if gradcam_image_path and os.path.exists(gradcam_image_path):
        story.append(Paragraph("Grad-CAM Visual Explanation", styles["Heading2"]))
        story.append(RLImage(gradcam_image_path, width=9 * cm, height=9 * cm))
        story.append(Spacer(1, 0.5 * cm))

    # --- SHAP ---
    if shap_plot_path and os.path.exists(shap_plot_path):
        story.append(Paragraph("Clinical Feature Contribution (SHAP)", styles["Heading2"]))
        story.append(RLImage(shap_plot_path, width=14 * cm, height=9 * cm))
        story.append(Spacer(1, 0.5 * cm))

    # --- Referral guidance ---
    story.append(Paragraph("Referral Guidance", styles["Heading2"]))
    for key in ("action", "specialist_type", "timeframe", "note"):
        if key in referral and referral[key]:
            story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {referral[key]}",
                                    styles["Normal"]))
    story.append(Spacer(1, 1 * cm))

    story.append(Paragraph(
        "This document was generated automatically by a student prototype system for "
        "academic/demonstration purposes. It has not been clinically validated and "
        "must not be used for real diagnostic or treatment decisions.",
        disclaimer_style,
    ))

    doc.build(story)
    return output_path
