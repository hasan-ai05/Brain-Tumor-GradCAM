# Brain Tumor Detection + Grad-CAM

EfficientNet-B0 fine-tuned on 7,023 brain MRI scans to classify four tumor types.
Grad-CAM produces a red heatmap that highlights exactly where in the MRI
the model detected evidence of a tumor — without any pixel-level annotations.

---

## The Problem

A radiologist examining a brain MRI needs two things: the tumor type and its location.
A classifier alone answers the first question. Adding Grad-CAM answers both.

The gradient-based heatmap is generated from the model's own internal representations —
no additional labeling, no segmentation masks, no second model.

---

## Dataset

**Source:** Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)  
**Records:** 7,200 MRI scans — 1,800 per class (balanced)  
**Split:** 70% Train / 15% Val / 15% Test — stratified

| Class | Description |
|---|---|
| glioma | Malignant tumor originating from glial cells |
| meningioma | Grows from membranes surrounding the brain — usually benign |
| pituitary | Pituitary gland tumor — affects hormone regulation |
| notumor | Healthy scan — control class |

**Unlike HAM10000, this dataset is balanced:** 1,800 images per class.
No class weighting needed. Standard CrossEntropyLoss was used.

---

## Model

**EfficientNet-B0** pretrained on ImageNet — 4,012,672 parameters.  
Input size: 224×224. Fine-tuned for 15 epochs on T4 GPU.

---

## Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.6733 | 82.1% | 0.3519 | 87.96% |
| 2 | 0.2353 | 92.1% | 0.1940 | 93.98% |
| 5 | 0.0624 | 97.8% | 0.0814 | 97.78% |
| 15 | — | — | — | best saved |

Validation accuracy tracks training accuracy closely throughout all 15 epochs.
Train loss and val loss converge — no overfitting. The balanced dataset and
sufficient sample size (7,200 images across 4 classes) prevent the gap
seen in the skin cancer project.

![Training Curves](images/training_curves.png)

---

## Results

**Test Accuracy: 97.41%**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.99 | 0.94 | 0.97 | 270 |
| meningioma | 0.95 | 0.96 | 0.95 | 270 |
| notumor | 0.97 | 1.00 | 0.99 | 270 |
| pituitary | 0.99 | 0.99 | 0.99 | 270 |

**Meningioma is the hardest class:** F1=0.95 — visually similar to glioma
in some orientations. Every other class exceeds F1=0.97.

The notumor class achieves Recall=1.00 — no healthy scan was misclassified
as containing a tumor, which is the most clinically critical false positive direction.

![Confusion Matrix](images/confusion_matrix.png)

---

## Grad-CAM

Grad-CAM computes the gradient of the predicted class score with respect
to the activations of the last convolutional layer (conv_head).
High-gradient regions receive a high weighting — these appear red in the heatmap.

**What the heatmaps show:**

For glioma and meningioma: the red region lands on the bright mass visible
in the MRI — not in a surrounding healthy area.

For pituitary: the heatmap concentrates at the base of the brain where the
pituitary gland sits — the anatomically correct location.

For notumor: the heatmap is diffuse and low-intensity — no single region
dominates, which is the expected result for a healthy scan.

The model learned tumor-specific spatial features without any location supervision.

![Grad-CAM Output](images/gradcam_brain.png)

---

## Comparison with Skin Cancer Project

| | Brain Tumor | Skin Cancer |
|---|---|---|
| Dataset size | 7,200 | 10,015 |
| Classes | 4 (balanced) | 7 (imbalanced) |
| Test Accuracy | 97.41% | 78.71% |
| Overfitting | None | Mild after epoch 4 |
| Class weights | Not needed | Required |

The accuracy gap is explained by two factors: fewer classes and balanced data.
Both projects use the same model architecture and explainability method.

---

## How to Run

```bash
pip install kagglehub timm torch torchvision pandas numpy matplotlib
jupyter notebook Brain.ipynb
```

Dataset downloads automatically via `kagglehub`. GPU recommended.

---

## Where to Place Screenshots

```
brain-tumor-gradcam/
    images/
        sample_mri.png          Section 2 — one sample per class
        training_curves.png     Section 6 — loss and accuracy
        confusion_matrix.png    Section 10 — 4x4 heatmap
        gradcam_brain.png       Section 9 — heatmap overlays
    README.md
    Brain.ipynb
```

---

## Project Structure

```
Brain.ipynb        main notebook (10 sections)
README.md          this file
images/            screenshots from notebook output
```

---

*Part of an 8-project AI Engineering portfolio — Hasan Akhras*
