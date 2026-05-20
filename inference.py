"""

Usage:
    python inference.py --input label_map.png --model checkpoints/best_model.pth --output result.png
    python inference.py --input label_map.png --model checkpoints/best_model.pth --output result.png --arch resnet
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from PIL import Image


# ── Architecture definitions (mirrors notebook) ────────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True,
                 dropout=False, innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        use_bias = not use_bn

        if down:
            conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias)
            layers = [conv]
            if not outermost:
                layers += [nn.LeakyReLU(0.2, False)]
            if use_bn and not outermost and not innermost:
                layers += [nn.BatchNorm2d(out_ch)]
        else:
            conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=use_bias)
            layers = [nn.ReLU(False), conv]
            if use_bn and not outermost:
                layers += [nn.BatchNorm2d(out_ch)]
            if dropout:
                layers += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    """U-Net 256 Generator — 8 encoder levels with skip connections."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, nf: int = 64):
        super().__init__()
        self.e1 = UNetBlock(in_ch,  nf,    down=True, use_bn=False, outermost=True)
        self.e2 = UNetBlock(nf,     nf*2,  down=True)
        self.e3 = UNetBlock(nf*2,   nf*4,  down=True)
        self.e4 = UNetBlock(nf*4,   nf*8,  down=True)
        self.e5 = UNetBlock(nf*8,   nf*8,  down=True)
        self.e6 = UNetBlock(nf*8,   nf*8,  down=True)
        self.e7 = UNetBlock(nf*8,   nf*8,  down=True)
        self.e8 = UNetBlock(nf*8,   nf*8,  down=True, use_bn=False, innermost=True)

        # d1=innermost (no skip), d8=outermost — matches notebook numbering
        self.d1 = UNetBlock(nf*8,        nf*8, down=False, dropout=True)
        self.d2 = UNetBlock(nf*8*2,      nf*8, down=False, dropout=True)
        self.d3 = UNetBlock(nf*8*2,      nf*8, down=False, dropout=True)
        self.d4 = UNetBlock(nf*8*2,      nf*8, down=False)
        self.d5 = UNetBlock(nf*8*2,      nf*4, down=False)
        self.d6 = UNetBlock(nf*4*2,      nf*2, down=False)
        self.d7 = UNetBlock(nf*2*2,      nf,   down=False)
        self.d8 = nn.Sequential(
            nn.ReLU(False),
            nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.d8(torch.cat([d7, e1], 1))


class ResNetUNetGenerator(nn.Module):
    """U-Net generator with ResNet-18 encoder (Phase 2 architecture)."""

    def __init__(self, out_ch: int = 3, pretrained: bool = False):
        super().__init__()
        enc = tv_models.resnet18(weights=None)

        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc1 = enc.layer1
        self.enc2 = enc.layer2
        self.enc3 = enc.layer3
        self.enc4 = enc.layer4

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
            )

        self.dec4 = up_block(512,       256)
        self.dec3 = up_block(256 + 256, 128)
        self.dec2 = up_block(128 + 128, 64)
        self.dec1 = up_block(64  + 64,  64)
        self.dec0 = up_block(64  + 64,  32)
        self.head = nn.Sequential(
            nn.Conv2d(32, out_ch, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        d0 = self.dec0(torch.cat([d1, e0], dim=1))
        return self.head(d0)


# ── Architecture auto-detection ────────────────────────────────────────────────

def detect_arch(state_dict: dict) -> str:
    """Infer architecture from checkpoint key names."""
    keys = list(state_dict.keys())
    if any(k.startswith("enc0.") for k in keys):
        return "resnet"
    return "unet"


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(model_path: Path, arch: str, device: torch.device) -> nn.Module:
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    if arch == "auto":
        arch = detect_arch(state_dict)
        print(f"Auto-detected architecture: {arch}")

    if arch == "resnet":
        model = ResNetUNetGenerator(out_ch=3, pretrained=False)
    else:
        model = UNetGenerator(in_ch=3, out_ch=3, nf=64)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model: nn.Module, input_path: Path, output_path: Path,
                  img_size: int, device: torch.device) -> None:
    img = Image.open(input_path).convert("RGB")
    tensor = TF.to_tensor(TF.resize(img, [img_size, img_size]))
    tensor = tensor * 2 - 1  # [0,1] -> [-1,1]
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out = (out.squeeze(0).cpu().clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    result = TF.to_pil_image(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix inference: label map -> aerial photo"
    )
    parser.add_argument("--input",  required=True,  type=Path,
                        help="Path to input label-map image (PNG/JPG)")
    parser.add_argument("--model",  required=True,  type=Path,
                        help="Path to generator checkpoint (.pth)")
    parser.add_argument("--output", required=True,  type=Path,
                        help="Path to save output image (PNG/JPG)")
    parser.add_argument("--arch",   default="auto",
                        choices=["auto", "unet", "resnet"],
                        help="Generator architecture (default: auto-detect)")
    parser.add_argument("--size",   default=256, type=int,
                        help="Input image size (default: 256)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, or cuda:N (default: auto)")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args.model, args.arch, device)
    run_inference(model, args.input, args.output, args.size, device)


if __name__ == "__main__":
    main()
