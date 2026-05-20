"""
gui.py — Pix2Pix TU-Graz  ·  Desktop GUI
Run:  python.exe gui.py
"""

import random
import threading
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 256
IMG_DISP       = 220          # display size in pixels
CKPT_BASE      = Path("checkpoints")
TEST_LABEL_DIR = Path("data/test/label")
TEST_REAL_DIR  = Path("data/test/real")

MODELS_CONFIG = [
    ("P2-Full\nResNet+FM+Aug+TL",  CKPT_BASE / "improved/exp_P2_full/best_G.pth",  "resnet"),
    ("P2-Aug\nUNet+FM+Aug",        CKPT_BASE / "improved/exp_P2_aug/best_G.pth",   "unet"),
    ("P1-FM\nUNet+FeatMatch",      CKPT_BASE / "improved/exp_P1_fm/best_G.pth",    "unet"),
    ("P1-Perc\nUNet+Perceptual",   CKPT_BASE / "improved/exp_P1_perc/best_G.pth",  "unet"),
    ("P1-L1\nUNet+L1",             CKPT_BASE / "improved/exp_P1_l1/best_G.pth",    "unet"),
    ("Baseline\nUNet vanilla",     CKPT_BASE / "baseline/best_G.pth",              "unet"),
]

# ── Model definitions ──────────────────────────────────────────────────────────
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
    def forward(self, x): return self.model(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64):
        super().__init__()
        self.e1=UNetBlock(in_ch,nf,   down=True,use_bn=False,outermost=True)
        self.e2=UNetBlock(nf,   nf*2, down=True)
        self.e3=UNetBlock(nf*2, nf*4, down=True)
        self.e4=UNetBlock(nf*4, nf*8, down=True)
        self.e5=UNetBlock(nf*8, nf*8, down=True)
        self.e6=UNetBlock(nf*8, nf*8, down=True)
        self.e7=UNetBlock(nf*8, nf*8, down=True)
        self.e8=UNetBlock(nf*8, nf*8, down=True,use_bn=False,innermost=True)
        self.d1=UNetBlock(nf*8,   nf*8,down=False,dropout=True)
        self.d2=UNetBlock(nf*8*2, nf*8,down=False,dropout=True)
        self.d3=UNetBlock(nf*8*2, nf*8,down=False,dropout=True)
        self.d4=UNetBlock(nf*8*2, nf*8,down=False)
        self.d5=UNetBlock(nf*8*2, nf*4,down=False)
        self.d6=UNetBlock(nf*4*2, nf*2,down=False)
        self.d7=UNetBlock(nf*2*2, nf,  down=False)
        self.d8=nn.Sequential(nn.ReLU(False),nn.ConvTranspose2d(nf*2,out_ch,4,2,1),nn.Tanh())
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(e1);e3=self.e3(e2);e4=self.e4(e3)
        e5=self.e5(e4);e6=self.e6(e5);e7=self.e7(e6);e8=self.e8(e7)
        d1=self.d1(e8)
        d2=self.d2(torch.cat([d1,e7],1));d3=self.d3(torch.cat([d2,e6],1))
        d4=self.d4(torch.cat([d3,e5],1));d5=self.d5(torch.cat([d4,e4],1))
        d6=self.d6(torch.cat([d5,e3],1));d7=self.d7(torch.cat([d6,e2],1))
        return self.d8(torch.cat([d7,e1],1))


class ResNetUNetGenerator(nn.Module):
    def __init__(self, out_ch=3):
        super().__init__()
        enc=tv_models.resnet18(weights=None)
        self.enc0=nn.Sequential(enc.conv1,enc.bn1,enc.relu)
        self.pool=enc.maxpool
        self.enc1=enc.layer1;self.enc2=enc.layer2
        self.enc3=enc.layer3;self.enc4=enc.layer4
        def up(ic,oc):
            return nn.Sequential(nn.ConvTranspose2d(ic,oc,4,2,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=False))
        self.dec4=up(512,256);self.dec3=up(512,128)
        self.dec2=up(256,64); self.dec1=up(128,64)
        self.dec0=up(128,32)
        self.head=nn.Sequential(nn.Conv2d(32,out_ch,1),nn.Tanh())
    def forward(self,x):
        e0=self.enc0(x);ep=self.pool(e0)
        e1=self.enc1(ep);e2=self.enc2(e1);e3=self.enc3(e2);e4=self.enc4(e3)
        d4=self.dec4(e4)
        d3=self.dec3(torch.cat([d4,e3],1));d2=self.dec2(torch.cat([d3,e2],1))
        d1=self.dec1(torch.cat([d2,e1],1));d0=self.dec0(torch.cat([d1,e0],1))
        return self.head(d0)


def load_model(ckpt_path, arch):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m  = ResNetUNetGenerator() if arch == "resnet" else UNetGenerator()
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


@torch.no_grad()
def infer(pil_img, model):
    t = TF.to_tensor(TF.resize(pil_img.convert("RGB"), [IMG_SIZE, IMG_SIZE]))
    t = (t * 2 - 1).unsqueeze(0).to(DEVICE)
    out = model(t).squeeze(0).cpu().clamp(-1, 1)
    return TF.to_pil_image((out + 1) / 2)


def pil_to_ctk(pil_img, size=IMG_DISP):
    pil_img = pil_img.resize((size, size), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))


# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pix2Pix · TU-Graz  —  Label Map → Aerial Photo")
        self.geometry("1540x720")
        self.resizable(True, True)
        self.configure(fg_color="#1a1a2e")

        self._loaded_models = []
        self._test_images   = sorted(TEST_LABEL_DIR.glob("*.png"))
        self._img_refs      = []   # keep ImageTk refs alive

        self._build_ui()
        self._load_models_async()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color="#16213e", corner_radius=0, height=64)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top,
            text="🛰  Pix2Pix  ·  TU-Graz  Aerial Photo Synthesis",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#e0e0ff",
        ).pack(side="left", padx=24, pady=14)

        self._status_lbl = ctk.CTkLabel(
            top, text="Loading models…",
            font=ctk.CTkFont(size=13), text_color="#7788aa"
        )
        self._status_lbl.pack(side="right", padx=24)

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color="#0f3460", corner_radius=0, height=52)
        ctrl.pack(fill="x", side="top")
        ctrl.pack_propagate(False)

        self._btn = ctk.CTkButton(
            ctrl,
            text="▶  Generate  (random image)",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#e94560", hover_color="#c73050",
            corner_radius=8, width=240, height=36,
            command=self._on_generate,
            state="disabled",
        )
        self._btn.pack(side="left", padx=18, pady=8)

        img_names = [p.name for p in self._test_images]
        self._img_var = ctk.StringVar(value="Random")
        self._dropdown = ctk.CTkOptionMenu(
            ctrl,
            values=["Random"] + img_names,
            variable=self._img_var,
            width=200, height=36,
            font=ctk.CTkFont(size=12),
            fg_color="#1a1a2e", button_color="#0f3460",
        )
        self._dropdown.pack(side="left", padx=6, pady=8)

        ctk.CTkLabel(
            ctrl, text="← pick a specific image or leave Random",
            font=ctk.CTkFont(size=11), text_color="#7788aa"
        ).pack(side="left", padx=6)

        # ── Main scrollable area ───────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(
            self, fg_color="#1a1a2e", scrollbar_button_color="#0f3460"
        )
        scroll.pack(fill="both", expand=True)

        self._panels = []

        # ── Row 0: Input  +  Ground Truth ─────────────────────────
        row0 = ctk.CTkFrame(scroll, fg_color="transparent")
        row0.pack(pady=(16, 6))

        for title, accent in [
            ("Input\nLabel Map",      "#4fc3f7"),
            ("Ground Truth\nReal Photo", "#81c784"),
        ]:
            self._panels.append(
                self._make_panel(row0, title, accent, IMG_DISP + 40, side="left", padx=24)
            )

        # ── Rows 1-2: 3 + 3 model panels ─────────────────────────────────────
        model_cfgs = [(m[0], "#ffb74d") for m in MODELS_CONFIG]

        for row_idx in range(2):
            row_frame = ctk.CTkFrame(scroll, fg_color="transparent")
            row_frame.pack(pady=6)
            for col_idx in range(3):
                cfg_idx = row_idx * 3 + col_idx
                title, accent = model_cfgs[cfg_idx]
                self._panels.append(
                    self._make_panel(row_frame, title, accent, IMG_DISP, side="left", padx=14)
                )

    def _make_panel(self, parent, title, accent, img_size, side="left", padx=10):
        frame = ctk.CTkFrame(parent, fg_color="#16213e", corner_radius=12,
                             border_width=1, border_color="#2a2a4a")
        frame.pack(side=side, padx=padx)

        bar = ctk.CTkFrame(frame, fg_color=accent, corner_radius=8, height=4)
        bar.pack(fill="x", padx=6, pady=(6, 0))

        ctk.CTkLabel(
            frame, text=title,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=accent, justify="center"
        ).pack(pady=(6, 4))

        placeholder = ctk.CTkLabel(
            frame, text="—",
            width=img_size, height=img_size,
            fg_color="#0d0d1a", corner_radius=8,
            text_color="#444466", font=ctk.CTkFont(size=28),
            cursor="hand2",
        )
        placeholder.pack(padx=8, pady=(0, 8))
        placeholder._pil_image = None
        placeholder.bind("<Button-1>", lambda e, lbl=placeholder, t=title: self._open_zoom(lbl, t))
        return placeholder

    # ── Model loading (background thread) ────────────────────────────────────
    def _load_models_async(self):
        def _worker():
            for name, ckpt, arch in MODELS_CONFIG:
                if ckpt.exists():
                    m = load_model(ckpt, arch)
                    self._loaded_models.append((name, m))
            n = len(self._loaded_models)
            self.after(0, self._on_models_ready, n)
        threading.Thread(target=_worker, daemon=True).start()

    # ── Zoom popup ────────────────────────────────────────────────────────────
    def _open_zoom(self, label_widget, title):
        # Only open if there's an actual image loaded
        if not hasattr(label_widget, "_pil_image") or label_widget._pil_image is None:
            return
        zoom = ctk.CTkToplevel(self)
        zoom.title(title.replace("\n", "  ·  "))
        zoom.configure(fg_color="#0d0d1a")
        zoom.grab_set()
        pil = label_widget._pil_image
        # Fit to screen (max 800px)
        max_side = 800
        w, h = pil.size
        scale = min(max_side / w, max_side / h, 1.0)
        disp = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=disp, dark_image=disp,
                               size=(disp.width, disp.height))
        lbl = ctk.CTkLabel(zoom, image=ctk_img, text="")
        lbl.image = ctk_img   # keep ref
        lbl.pack(padx=16, pady=16)
        ctk.CTkButton(
            zoom, text="Close", width=120,
            fg_color="#e94560", hover_color="#c73050",
            command=zoom.destroy
        ).pack(pady=(0, 14))

    def _on_models_ready(self, n):
        self._status_lbl.configure(text=f"✓  {n} models ready   ·   Device: {DEVICE}")
        self._btn.configure(state="normal")

    # ── Generate ──────────────────────────────────────────────────────────────
    def _on_generate(self):
        if not self._test_images:
            return
        self._btn.configure(state="disabled", text="Running…")
        self._status_lbl.configure(text="Generating…")
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        sel = self._img_var.get()
        if sel == "Random":
            img_path = random.choice(self._test_images)
        else:
            img_path = TEST_LABEL_DIR / sel

        real_path  = TEST_REAL_DIR / img_path.name
        label_pil  = Image.open(img_path).convert("RGB")
        real_pil   = Image.open(real_path).convert("RGB") if real_path.exists() else label_pil

        results = [label_pil, real_pil]
        for _, model in self._loaded_models:
            results.append(infer(label_pil, model))

        self.after(0, self._update_panels, results, img_path.name)

    def _update_panels(self, images, fname):
        self._img_refs.clear()
        for i, (panel, pil_img) in enumerate(zip(self._panels, images)):
            size = IMG_DISP + 40 if i < 2 else IMG_DISP   # input/GT larger
            ctk_img = pil_to_ctk(pil_img, size)
            self._img_refs.append(ctk_img)
            panel._pil_image = pil_img   # store original for zoom
            panel.configure(image=ctk_img, text="")
        self._btn.configure(state="normal", text="▶  Generate  (random image)")
        self._status_lbl.configure(text=f"✓  {fname}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
