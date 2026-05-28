# app.py  --  Pix2Pix TU-Graz interactive demo
# Run:  D:/entornos/deep_learning_env/python.exe app.py

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 256
CKPT_BASE      = Path("checkpoints")
TEST_LABEL_DIR = Path("data/test/label")  # semantic label maps → model input
TEST_REAL_DIR  = Path("data/test/real")   # real aerial photos  → ground truth

# Ordered best → worst by test PSNR
MODELS_CONFIG = [
    ("1  P2-Full  (ResNet + FM + Aug + Transfer)",  CKPT_BASE / "improved/exp_P2_full/best_G.pth",  "resnet"),
    ("2  P2-Aug   (UNet  + FM + Augmentation)",     CKPT_BASE / "improved/exp_P2_aug/best_G.pth",   "unet"),
    ("3  P1-FM    (UNet  + Feature Matching)",      CKPT_BASE / "improved/exp_P1_fm/best_G.pth",    "unet"),
    ("4  P1-Perc  (UNet  + Perceptual Loss)",       CKPT_BASE / "improved/exp_P1_perc/best_G.pth",  "unet"),
    ("5  P1-L1    (UNet  + L1 only)",               CKPT_BASE / "improved/exp_P1_l1/best_G.pth",    "unet"),
    ("6  Baseline (UNet  + L1, vanilla)",           CKPT_BASE / "baseline/best_G.pth",              "unet"),
]

# ── Architecture definitions (inline — no dependency on inference.py) ─────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True,
                 dropout=False, innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        use_bias = not use_bn
        if down:
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias)]
            if not outermost:
                layers += [nn.LeakyReLU(0.2, False)]
            if use_bn and not outermost and not innermost:
                layers += [nn.BatchNorm2d(out_ch)]
        else:
            layers = [nn.ReLU(False), nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=use_bias)]
            if use_bn and not outermost:
                layers += [nn.BatchNorm2d(out_ch)]
            if dropout:
                layers += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64):
        super().__init__()
        self.e1 = UNetBlock(in_ch, nf,    down=True, use_bn=False, outermost=True)
        self.e2 = UNetBlock(nf,    nf*2,  down=True)
        self.e3 = UNetBlock(nf*2,  nf*4,  down=True)
        self.e4 = UNetBlock(nf*4,  nf*8,  down=True)
        self.e5 = UNetBlock(nf*8,  nf*8,  down=True)
        self.e6 = UNetBlock(nf*8,  nf*8,  down=True)
        self.e7 = UNetBlock(nf*8,  nf*8,  down=True)
        self.e8 = UNetBlock(nf*8,  nf*8,  down=True, use_bn=False, innermost=True)
        self.d1 = UNetBlock(nf*8,   nf*8, down=False, dropout=True)
        self.d2 = UNetBlock(nf*8*2, nf*8, down=False, dropout=True)
        self.d3 = UNetBlock(nf*8*2, nf*8, down=False, dropout=True)
        self.d4 = UNetBlock(nf*8*2, nf*8, down=False)
        self.d5 = UNetBlock(nf*8*2, nf*4, down=False)
        self.d6 = UNetBlock(nf*4*2, nf*2, down=False)
        self.d7 = UNetBlock(nf*2*2, nf,   down=False)
        self.d8 = nn.Sequential(
            nn.ReLU(False),
            nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); e8=self.e8(e7)
        d1=self.d1(e8)
        d2=self.d2(torch.cat([d1,e7],1)); d3=self.d3(torch.cat([d2,e6],1))
        d4=self.d4(torch.cat([d3,e5],1)); d5=self.d5(torch.cat([d4,e4],1))
        d6=self.d6(torch.cat([d5,e3],1)); d7=self.d7(torch.cat([d6,e2],1))
        return self.d8(torch.cat([d7,e1],1))


class ResNetUNetGenerator(nn.Module):
    def __init__(self, out_ch=3, pretrained=False):
        super().__init__()
        enc = tv_models.resnet18(weights=None)
        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc1 = enc.layer1; self.enc2 = enc.layer2
        self.enc3 = enc.layer3; self.enc4 = enc.layer4
        def up(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=False))
        self.dec4 = up(512, 256); self.dec3 = up(512, 128)
        self.dec2 = up(256, 64);  self.dec1 = up(128, 64)
        self.dec0 = up(128, 32)
        self.head = nn.Sequential(nn.Conv2d(32, out_ch, 1), nn.Tanh())

    def forward(self, x):
        e0=self.enc0(x); ep=self.pool(e0)
        e1=self.enc1(ep); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        d4=self.dec4(e4)
        d3=self.dec3(torch.cat([d4,e3],1)); d2=self.dec2(torch.cat([d3,e2],1))
        d1=self.dec1(torch.cat([d2,e1],1)); d0=self.dec0(torch.cat([d1,e0],1))
        return self.head(d0)

# ── Load all models once at startup ───────────────────────────────────────────
def _load_model(ckpt_path: Path, arch: str):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if arch == "resnet":
        model = ResNetUNetGenerator(out_ch=3)
    else:
        model = UNetGenerator(in_ch=3, out_ch=3, nf=64)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()

print(f"Device: {DEVICE}")
LOADED_MODELS = []
for label, ckpt, arch in MODELS_CONFIG:
    if ckpt.exists():
        LOADED_MODELS.append((label, _load_model(ckpt, arch)))
        print(f"  [OK] {label}")
    else:
        print(f"  [SKIP] {ckpt.name} not found")
print(f"Ready — {len(LOADED_MODELS)} models loaded.\n")

TEST_IMAGES = sorted(TEST_LABEL_DIR.glob("*.png"))

# ── Single-image inference ─────────────────────────────────────────────────────
@torch.no_grad()
def _infer(pil_img: Image.Image, model) -> Image.Image:
    t = TF.to_tensor(TF.resize(pil_img.convert("RGB"), [IMG_SIZE, IMG_SIZE]))
    t = (t * 2 - 1).unsqueeze(0).to(DEVICE)
    out = model(t)
    out = (out.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(out)

# ── Random sample: pick image, run all models ──────────────────────────────────
def random_sample():
    if not TEST_IMAGES:
        return []
    img_path  = random.choice(TEST_IMAGES)
    real_path = TEST_REAL_DIR / img_path.name

    label_pil = Image.open(img_path).convert("RGB")
    real_pil  = Image.open(real_path).convert("RGB") if real_path.exists() else label_pil

    gallery = [
        (label_pil, "Input label map"),
        (real_pil,  "Ground truth (real aerial photo)"),
    ]
    for name, model in LOADED_MODELS:
        gallery.append((_infer(label_pil, model), name))
    return gallery

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Pix2Pix TU-Graz") as demo:
    gr.Markdown(
        "## Pix2Pix TU-Graz — Label Map to Aerial Photo\n"
        "Press the button to pick a **random test image** and run all 6 models.  \n"
        "Results are ordered **best to worst** by test PSNR."
    )

    btn = gr.Button("New random image", variant="primary", size="lg")

    gallery = gr.Gallery(
        label="Input | Ground Truth | Models (best to worst)",
        columns=4,
        height=500,
        object_fit="contain",
        show_label=True,
    )

    btn.click(fn=random_sample, inputs=[], outputs=gallery)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
