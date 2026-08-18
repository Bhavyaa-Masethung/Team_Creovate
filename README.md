# Team Creovate — KLA Semiconductor AI Hackathon 2026

## AI-Based Restoration of Degraded Images

This repository contains the offline inference solution for the KLA problem statement.

## Structure

```text
Team_Creovate/
├── run.py
├── requirements.txt
├── README.md
└── models/
    └── best_model.pth
```

## Execution

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py D:\Test_NoisyLR D:\Team_Creovate\test_result
```

The program:
- reads every `.npy` input;
- creates the output directory automatically;
- generates exactly one `.npy` output per input;
- preserves the input filename;
- produces grayscale `float32` arrays;
- clips outputs to `[0,1]`;
- removes NaN and Inf values;
- runs fully offline;
- uses CUDA when an NVIDIA GPU is available.

## Model

The supplied `best_model.pth` contains the trained 0.63M-parameter residual restoration network.

The checkpoint architecture is:
- 1-channel input
- 64 feature channels
- 8 residual blocks
- residual body convolution
- 2x PixelShuffle reconstruction

No internet connection, API key, or model download is required during inference.

## Final output requirement

The required submission output is `.npy`.

PNG conversion is kept separate and is intended only for visual inspection.
