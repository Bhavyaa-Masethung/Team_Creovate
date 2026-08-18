import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class KLAResidualNet(nn.Module):
    """
    Architecture reconstructed to exactly match the submitted checkpoint:
      head:       1 -> 64
      body:       8 residual blocks
      body_conv:  64 -> 64
      upsample:   64 -> 4
      PixelShuffle(2): 4 -> 1
    """
    def __init__(self, channels=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.body_conv = nn.Conv2d(
            channels, channels, 3, padding=1, bias=True
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.head(x)
        skip = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + skip
        x = self.upsample(x)
        return x


def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "best_model.pth"
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Missing model weights: {model_path}"
        )

    # The checkpoint is a local file included with the submission.
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )
    else:
        state = checkpoint

    # Remove DataParallel prefix if present.
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    model = KLAResidualNet(channels=64, num_blocks=8)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def restore(model, arr):
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected grayscale HxW or HxWx1 input, got {arr.shape}"
        )

    # Remove invalid numerical values and keep the network input bounded.
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )
    arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape

    # Training inputs were 128x128 and targets were 256x256.
    # For the provided 256x256 test inputs, reduce to the trained
    # 128x128 input size before the learned 2x restoration.
    if (h, w) == (256, 256):
        small = F.interpolate(
            torch.from_numpy(arr)[None, None].to(DEVICE),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )
    elif h % 2 == 0 and w % 2 == 0:
        small = F.interpolate(
            torch.from_numpy(arr)[None, None].to(DEVICE),
            size=(h // 2, w // 2),
            mode="bilinear",
            align_corners=False
        )
    else:
        small = torch.from_numpy(arr)[None, None].to(DEVICE)

    output = model(small)

    # Guarantee the required target resolution.
    output = F.interpolate(
        output,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    output = output.squeeze(0).squeeze(0).cpu().numpy()
    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )
    output = np.clip(output, 0.0, 1.0).astype(np.float32)

    return output


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_dir, "*.npy"))
    )

    print("=" * 70)
    print("KLA SEMICON AI HACKATHON 2026 - TEAM CREOVATE")
    print("=" * 70)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading model...")
    model = load_model()
    print("Checkpoint weights loaded successfully.")
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    print("=" * 70)
    print("TEST DATA")
    print("=" * 70)
    print("Input directory :", input_dir)
    print("Output directory:", output_dir)
    print("Images found    :", len(files))

    failed = 0

    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    for i, path in enumerate(files, 1):
        try:
            arr = np.load(path)
            restored = restore(model, arr)

            # Same filename as input, as required by the evaluator.
            out_path = os.path.join(
                output_dir,
                os.path.basename(path)
            )
            np.save(out_path, restored)

        except Exception as exc:
            failed += 1
            print(
                f"FAILED: {os.path.basename(path)} | {exc}"
            )

        if i == 1 or i % 100 == 0 or i == len(files):
            print(
                f"Progress: {i}/{len(files)}"
            )

    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print("Input images       :", len(files))
    print("Successful outputs :", len(files) - failed)
    print("Failed outputs     :", failed)
    print("Output format      : .npy")
    print("Output dtype       : float32")
    print("Output values      : [0,1]")
    print(
        "FINAL STATUS       :",
        "PASS" if failed == 0 else "CHECK FAILED FILES"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
