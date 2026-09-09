"""
train.py
--------
Phase 1: Multimodal Breast Ultrasound (BUS) + Clinical Data Training Script.
Adapted to the BrEaST-Lesions-USG dataset (Dec 15, 2023 release).

RESEARCH / EDUCATIONAL PROTOTYPE ONLY.
This model is NOT a validated medical device and must not be used for real
clinical decision-making. It is a final-year student project intended to
demonstrate a multimodal deep learning pipeline.

Architecture
------------
- Vision backbone : EfficientNet-B0 or B2 (torchvision, ImageNet-pretrained)
- Clinical branch : Small MLP over tabular features
- Fusion          : Concatenate image embedding + clinical embedding -> MLP head -> 1 logit
- Uncertainty     : Dropout layers are left active at inference time by utils.py
                    (MC Dropout) to estimate predictive uncertainty.

Expected data layout
---------------------
data/
  BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx   (as provided)
  images/
    case001.png
    case001_tumor.png   (tumor mask - not used by this script, kept for
                          future work e.g. lesion-cropping or ROI attention)
    case002.png
    ...

Extract the provided zip into `data/images/` before running, e.g.:
    mkdir -p data/images
    unzip BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip -d data/images_raw
    mv data/images_raw/BrEaST-Lesions_USG-images_and_masks/*.png data/images/

IMPORTANT — honest notes about this dataset (read before using clinically
or in your report):
1. There is NO true `patient_id` field in this release, only `CaseID`
   (one row per ultrasound exam). We therefore split at the CASE level.
   If the same patient contributed multiple exams, this could still leak
   across train/val — the source dataset does not give us a way to check
   this, so it should be stated as a limitation in your report.
2. `menopausal_status` is NOT provided by this dataset. It is approximated
   here as `age >= 50` (a common clinical convention for postmenopausal
   status), which is a coarse proxy, not ground truth. This is clearly
   flagged in the code and should be flagged in your report too.
3. `prior_biopsy` (history of a PREVIOUS biopsy) is NOT provided and is
   intentionally NOT derived from the `Verification` column
   ("confirmed by biopsy" / "confirmed by follow-up care"), because that
   column describes how THIS case's diagnosis was confirmed — using it as
   an input feature would leak information correlated with the label
   itself. This feature is dropped rather than faked. If you later obtain
   a real prior-biopsy-history field, add it back into CLINICAL_FEATURES.
4. `Classification` has 3 raw values: benign (154), malignant (98),
   normal (4, no visible lesion). This script maps benign + normal -> 0
   and malignant -> 1 for a binary task. Given there are only 4 "normal"
   cases, consider excluding them (see EXCLUDE_NORMAL flag) if you want a
   cleaner "lesion" classification task instead.
"""

import os
import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as T
import torchvision.models as models
import joblib


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Only features we can honestly derive from this dataset release.
# See module docstring notes 2 and 3 for why menopausal_status is a proxy
# and prior_biopsy is excluded entirely.
CLINICAL_FEATURES = [
    "age",
    "menopausal_status_proxy",  # age >= 50 heuristic — see docstring note 2
    "family_history",
    "palpable_lump",
]

EXCLUDE_NORMAL = False   # set True to drop the 4 "normal" (no-lesion) cases
CLINICAL_SHEET_NAME = "BrEaST-Lesions-USG clinical dat"


@dataclass
class TrainConfig:
    clinical_xlsx: str = "data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    image_dir: str = "data/images"
    backbone: str = "efficientnet_b0"   # or "efficientnet_b2"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    dropout_p: float = 0.3
    seed: int = 42
    out_dir: str = "artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Clinical feature engineering (BrEaST-specific)
# --------------------------------------------------------------------------
def _to_numeric_age(value) -> float:
    """Age column mixes ints and the literal string 'not available'."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def derive_clinical_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the modelling dataframe (image_filename, clinical features, label)
    from the raw BrEaST clinical Excel sheet.
    """
    df = raw_df.copy()

    df["age"] = df["Age"].apply(_to_numeric_age)

    signs = df["Signs"].fillna("").astype(str)
    symptoms = df["Symptoms"].fillna("").astype(str)

    df["palpable_lump"] = signs.str.contains("palpable", case=False).astype(int)
    df["family_history"] = symptoms.str.contains(
        "family history of breast/ovarian cancer", case=False
    ).astype(int)

    # menopausal_status_proxy is filled in AFTER age imputation (see
    # load_and_split), since it depends on the (train-median-imputed) age.

    df["label"] = (df["Classification"].astype(str).str.strip().str.lower() == "malignant").astype(int)

    if EXCLUDE_NORMAL:
        df = df[df["Classification"].astype(str).str.strip().str.lower() != "normal"].reset_index(drop=True)

    out = df[["CaseID", "Image_filename", "age", "family_history", "palpable_lump", "label"]].copy()
    out = out.rename(columns={"CaseID": "case_id", "Image_filename": "image_filename"})
    return out


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class BUSDataset(Dataset):
    """
    Loads paired (ultrasound image, clinical features, label) samples.
    Clinical features must already be numeric and present as columns
    matching CLINICAL_FEATURES; scaling is applied via a pre-fit scaler.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, scaler: StandardScaler,
                 image_size: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.scaler = scaler
        self.augment = augment

        base_transforms = [T.Resize((image_size, image_size))]
        if augment:
            base_transforms += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        base_transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                        std=[0.229, 0.224, 0.225]),
        ]
        self.transform = T.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        clinical_raw = row[CLINICAL_FEATURES].values.astype(np.float32).reshape(1, -1)
        clinical_scaled = self.scaler.transform(clinical_raw).astype(np.float32).flatten()
        clinical_tensor = torch.tensor(clinical_scaled, dtype=torch.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return image, clinical_tensor, label


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ClinicalMLP(nn.Module):
    """Small dense network for tabular clinical features."""

    def __init__(self, in_features: int, hidden: int = 32, out_features: int = 16,
                 dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalBreastCancerNet(nn.Module):
    """
    Fuses an EfficientNet image embedding with a clinical MLP embedding,
    then classifies via a small fused head.

    Dropout is present in both branches and the head, so that at inference
    time `model.train()` (dropout active) + repeated forward passes gives
    Monte Carlo Dropout uncertainty estimates (see utils.py).
    """

    def __init__(self, backbone_name: str = "efficientnet_b0",
                 clinical_in_features: int = len(CLINICAL_FEATURES),
                 dropout_p: float = 0.3, pretrained: bool = True):
        super().__init__()

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            img_embed_dim = backbone.classifier[1].in_features
        elif backbone_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b2(weights=weights)
            img_embed_dim = backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Replace classifier with a Dropout-only stub -> use the pooled
        # feature vector as our image embedding, keeping a dropout layer
        # active for MC Dropout at inference time.
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
        )
        self.image_backbone = backbone
        self.image_embed_dim = img_embed_dim

        self.clinical_mlp = ClinicalMLP(
            in_features=clinical_in_features,
            hidden=32,
            out_features=16,
            dropout_p=dropout_p,
        )

        fused_dim = self.image_embed_dim + 16
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),  # raw logit; sigmoid applied outside (BCEWithLogitsLoss)
        )

    def forward(self, image, clinical):
        img_feat = self.image_backbone(image)          # (B, img_embed_dim)
        clin_feat = self.clinical_mlp(clinical)         # (B, 16)
        fused = torch.cat([img_feat, clin_feat], dim=1)
        logit = self.fusion_head(fused)                 # (B, 1)
        return logit.squeeze(1)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def load_and_split(cfg: TrainConfig):
    raw_df = pd.read_excel(cfg.clinical_xlsx, sheet_name=CLINICAL_SHEET_NAME)
    df = derive_clinical_dataframe(raw_df)

    # No true patient_id in this dataset release — split at case level and
    # document this as a limitation (see module docstring note 1).
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction, random_state=cfg.seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["case_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Impute missing age using the TRAIN median only (avoids leaking val
    # statistics into train), then derive the menopausal proxy from the
    # now-complete age column.
    train_age_median = train_df["age"].median()
    for split_df in (train_df, val_df):
        split_df["age"] = split_df["age"].fillna(train_age_median)
        split_df["menopausal_status_proxy"] = (split_df["age"] >= 50).astype(int)

    # Fit scaler on TRAIN ONLY.
    scaler = StandardScaler()
    scaler.fit(train_df[CLINICAL_FEATURES].values.astype(np.float32))

    return train_df, val_df, scaler, train_age_median


# --------------------------------------------------------------------------
# Training / validation loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, clinical, labels in loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images, clinical)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_df, val_df, scaler, age_median = load_and_split(cfg)
    print(f"Train cases: {len(train_df)} | Val cases: {len(val_df)}")
    print(f"Train label balance -> malignant: {train_df['label'].mean():.2%}")
    print(f"Val label balance   -> malignant: {val_df['label'].mean():.2%}")
    print(f"Train age median used for imputation: {age_median:.1f}")

    train_ds = BUSDataset(train_df, cfg.image_dir, scaler, cfg.image_size, augment=True)
    val_ds = BUSDataset(val_df, cfg.image_dir, scaler, cfg.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = MultimodalBreastCancerNet(
        backbone_name=cfg.backbone,
        dropout_p=cfg.dropout_p,
        pretrained=True,
    ).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    model_path = os.path.join(cfg.out_dir, "model.pth")
    scaler_path = os.path.join(cfg.out_dir, "scaler.pkl")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, cfg.device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, cfg.device, train=False)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": cfg.backbone,
                "dropout_p": cfg.dropout_p,
                "clinical_features": CLINICAL_FEATURES,
                "age_median_for_imputation": age_median,
            }, model_path)
            print(f"  -> New best model saved to {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"Clinical scaler saved to {scaler_path}")

    # Save a small REAL (unscaled) sample of training clinical rows for SHAP's
    # KernelExplainer background at inference time — deployed apps ship only
    # model.pth/scaler.pkl/background_clinical.npy, not the raw dataset, and
    # SHAP needs actual reference rows rather than fabricated ones.
    bg_size = min(40, len(train_df))
    background = train_df[CLINICAL_FEATURES].sample(n=bg_size, random_state=cfg.seed).values.astype(np.float32)
    background_path = os.path.join(cfg.out_dir, "background_clinical.npy")
    np.save(background_path, background)
    print(f"SHAP background sample ({bg_size} rows) saved to {background_path}")

    print("Training complete.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train multimodal breast cancer risk model on BrEaST-Lesions-USG")
    p.add_argument("--clinical_xlsx", default="data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx")
    p.add_argument("--image_dir", default="data/images")
    p.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "efficientnet_b2"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="artifacts")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_model(config)
