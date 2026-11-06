# Degradation as Biomarker or Shortcut?

Code and results for the study **"Degradation as Biomarker or Shortcut? Image-Quality Information in Medical Image Classifiers under Domain Shift"** (MIDL 2027 submission).

Image degradation (blur, noise, exposure, contrast) is usually treated as a nuisance in medical imaging, yet it can also carry label information through acquisition site, device or patient state. This repository tests whether a classifier should use that information (a *biomarker*) or be prevented from using it (a *shortcut*), on two tasks with an external test set each:

| Part | Task | Source (train / ID test) | External test |
|---|---|---|---|
| Part 1 | Referable diabetic retinopathy (fundus) | APTOS-2019 | Messidor-2 |
| Part 2 | Pneumonia (chest radiograph) | NIH ChestX-ray14 | Kermany pediatric |

## Repository structure

```
degradation-aware-medical-ai/
├── Part 1/                  # Fundus: APTOS-2019 -> Messidor-2
│   ├── mainv1.ipynb         # full pipeline (Kaggle T4 notebook)
│   ├── features/            # degradation features and cached features
│   ├── results/             # tables (CSV), figures, per-image predictions
│   └── __huggingface_repos__.json
└── Part 2/                  # CXR: NIH ChestX-ray14 -> Kermany
    ├── mainv2.ipynb
    ├── features/
    ├── results/
    └── __huggingface_repos__.json
```

Each notebook is self-contained and writes its outputs to `ckpt/`, `features/` and `results/` next to it.

## What the notebooks do

1. **Degradation features (11):** four no-reference IQA scores (BRISQUE, NIQE, PIQE, MUSIQ via `pyiqa`) and seven hand-crafted statistics (blur, luminance mean and SD, colourfulness, dark channel, RMS contrast, dynamic range). Constant features are dropped (colourfulness on greyscale radiographs).
2. **Probes:** degradation-only classifiers with shuffled negative controls, plus external transfer.
3. **Arms**, trained end to end on six ImageNet backbones (ResNet-50, ConvNeXt-T, ViT-S/16, DenseNet-121, EfficientNet-B0, Swin-T) with five seeds (42, 7, 2024, 11, 99):
   - `baseline`
   - `harness`: an MLP embeds the degradation vector and is concatenated with the backbone features
   - `harness_ctrl`: same, with degradation vectors shuffled across training images
   - `suppress`: gradient-reversal adversary predicting the BRISQUE quintile
   - ResNet-50 only: `aug`, `aug_strong`, `harness_brisque`
   - `deg_only`: logistic regression on degradation features, no image
4. **Test-time intervention:** the trained harness is evaluated with its degradation vector zeroed or shuffled.
5. **Forensic probes** on the penultimate features (BRISQUE-quintile accuracy, ridge R², domain accuracy).
6. **Statistics:** seed-ensemble AUC, paired bootstrap over images (and a variant that also resamples seeds), Holm correction, per-seed sign consistency, difference-in-differences across the two tasks.

## Key results

- Degradation features alone predict the label in distribution (AUC 0.902 fundus, 0.621 CXR; shuffled controls at chance), but the association transfers poorly. On CXR they also encode the AP/PA view (AUC 0.971).
- Harness minus baseline, external AUC pooled over six backbones: **+0.016** on fundus, **−0.014** on CXR. These single-dataset effects are small and mostly not significant once seed variation is included.
- Pre-specified test-time contrast (fundus drop minus CXR drop when the degradation vector is zeroed): **+0.025**, 95% CI [0.013, 0.036], positive on all six backbones, also when seeds are resampled.
- Gradient reversal does not remove degradation information from the features: a fresh probe still recovers the BRISQUE quintile far above the 0.2 chance level.
- The difference-in-differences for the harness is not significant when seeds are resampled (post-hoc analysis).

See the paper for all tables, limitations and the full statistical protocol.

## Datasets

The datasets are public and are not included in this repository. Download them from their original sources and set the paths at the top of each notebook:

- APTOS 2019 Blindness Detection (Kaggle)
- Messidor-2
- NIH ChestX-ray14
- Kermany pediatric pneumonia (Chest X-ray images)

Follow each dataset's licence and terms of use.

## Reproducing

- **Compute:** Kaggle notebook, one NVIDIA T4 GPU.
- **Order:** run `Part 1/mainv1.ipynb` and `Part 2/mainv2.ipynb` independently; each produces its own results folder.
- **Main dependencies:** PyTorch, `timm`, `pyiqa`, scikit-learn, pandas, NumPy, OpenCV, Matplotlib.
- **Settings:** AdamW, cosine schedule, 15 epochs, batch size 32, class-weighted cross-entropy, mixed precision; backbone learning rate 3e-5 (DenseNet-121 5e-5, EfficientNet-B0 1e-4), head learning rate 1e-3; checkpoint with the best validation AUC.

The `ckpt/` folders can be large. If the repository exceeds GitHub's file-size limits, track checkpoints with Git LFS or host them elsewhere and link them here.

## Citation



## Contact

Kazi Rifat Ahmed, Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh. Email: rifat.swe@diu.edu.bd

## License

Add a license file before making the repository public. Code and results are shared for research purposes; the datasets keep their own licences.
