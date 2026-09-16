# Task 2: Multi-Modal Cardiac Image Fusion

## Dataset
**Heart CT & MRI Dataset** — Kaggle  
Link: https://www.kaggle.com/datasets/ziya07/heart-ct-and-mri-dataset

Download and place a matched pair of slices (same cardiac cross-section, same slice index) as:
```
Task_2_Cardiac_Fusion/
  data/
    sample_ct.png    ← one CT slice (grayscale)
    sample_mri.png   ← the paired MRI slice (grayscale)
```

Both files must be co-registered (same anatomy, same slice position).

---

## Pipeline Description

### Step 1 — Load & Align
Loaded as grayscale uint8 via `cv2.IMREAD_GRAYSCALE`.  
If the CT and MRI shapes differ, the MRI is resized to match CT dimensions using bilinear interpolation.  
**Note:** This is geometric convenience only — true registration (affine/deformable) is outside this lab's scope.

### Step 2 — Histogram Equalization (independent, per-modality)
`cv2.equalizeHist` is applied to CT and MRI **separately** before any blending.

**Why independent?**  
CT and MRI use fundamentally different intensity scales (CT in Hounsfield Units; MRI in arbitrary scanner units).  
Equalizing them independently normalises both to the [0, 255] range so the subsequent weighted blend is a genuine information merge — not an accidental brightness competition between scanners.

### Step 3 — Color Mapping
| Modality | Colormap | Rationale |
|----------|----------|-----------|
| CT | `COLORMAP_BONE` | Preserves the clinical grey-white feel; bone vs. soft tissue reads intuitively |
| MRI | `COLORMAP_JET` | Maps intensity to blue→green→red; subtle soft-tissue gradients become distinct hues |

Both become BGR uint8 `(H, W, 3)` matrices.

### Step 4 — Weighted Fusion
```
fused = cv2.addWeighted(ct_color, α=0.65, mri_color, β=0.35, γ_scalar=0)
```

**Weight rationale:**
- **CT weight = 0.65 (heavier):** Preserves sharp anatomical boundaries (cardiac wall, valves, pericardium). CT's structural edges are the diagnostic anchor.
- **MRI weight = 0.35 (lighter):** Injects soft-tissue contrast information (myocardial fibrosis, trabeculation, oedema) without overwhelming the structural signal.
- **Sum = 1.0:** Keeps the output in the valid [0, 255] range without a scalar offset.

These are starting values. In a clinical context they would be tuned per patient, imaging protocol, and diagnostic question.

### Step 5 — Logarithmic Transformation
```
s = C · log(1 + r),   where r ∈ [0, 1]
```
Expands the low-intensity (dark) region of the fused image — specifically the blood-filled cardiac chambers which tend to be crushed near black after weighted addition. `log1p` prevents domain errors at r=0.

### Step 6 — Power-Law (Gamma) Transformation
```
s = r^γ,   γ = 0.6,   r ∈ [0, 1]
```
Applied on top of the log output. γ < 1 brightens midtones (myocardium, connective tissue) while keeping pure blacks and whites anchored. Prevents bright cardiac walls from being blown out to pure white.

### Step 7 — Comparative Analysis
Side-by-side display of all 7 stages plus quantitative mean ± std per stage to validate the transforms are operating as intended.

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA`   | 0.65  | CT blend weight in `addWeighted` |
| `BETA`    | 0.35  | MRI blend weight |
| `LOG_C`   | 1.0   | Log transform scale constant |
| `GAMMA_VAL` | 0.6 | Power-law exponent (< 1 = midtone brightening) |

---

## Output Files

| File | Description |
|------|-------------|
| `01_raw_modalities.png` | Raw CT and MRI side-by-side |
| `02_histogram_equalization.png` | Histograms before/after equalization |
| `02_equalized_images.png` | Visual comparison after equalization |
| `03_color_heatmaps.png` | False-color heatmaps (BONE / JET) |
| `04_weighted_fusion.png` | CT + MRI weighted blend |
| `05_log_transform.png` | Fusion before/after log transform |
| `06_gamma_transform.png` | Gamma correction comparison |
| `07_full_pipeline_comparison.png` | Full 7-stage comparison |
| `08_quantitative_stats.png` | Mean ± std bar chart across stages |
| `ct_bone_heatmap.png` | Saved CT heatmap |
| `mri_jet_heatmap.png` | Saved MRI heatmap |
| `fused_weighted.png` | Raw weighted fusion |
| `fused_log.png` | Log-transformed fusion |
| `fused_final.png` | Final enhanced fusion |

---

## Requirements
See `requirements.txt` in the root directory.

## Running the Notebook
```bash
jupyter notebook modal_fusion.ipynb
```
Run all cells in order (Kernel → Restart & Run All).
