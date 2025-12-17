# Overview of MMDet-Model Comparison
This evaluation compares three MM-Det (for object detection) models tested on the **micro dataset** (just a few images), focusing on **performance, model size, and precision**.
The images were evaluated with a human in the loop, i.e. I documented the performance manually.
This serves as the baseline which model to choose for running all approx. 1000 construction site images.

## Medium Model — RTMDet-M
**Config:** `rtmdet_m_8xb32-300e_coco.py`

| Metric | Value |
|------|------|
| **Weights Size** | 224 MB |
| **Inference Speed (CPU)** | 1.3 it/s (images/sec) |
| **True Positives (TP)** | 8 |
| **False Positives (FP)** | 0 |
| **Precision** | **1.00** |

Result:
- Fast, but not perfect with regards to TP rate.

---

### RCNN Model — Cascade RCNN X101
**Config:** `cascade-rcnn_x101_64x4d_fpn_20e_coco.py`

| Metric | Value |
|------|------|
| **Weights Size** | 510 MB |
| **Inference Speed (CPU)** | 0.1 it/s |
| **True Positives (TP)** | 17 |
| **False Positives (FP)** | 9 |
| **Precision** | **0.65** |

- this one was disappointing because it was slow and had a high FP rate.
- the TP was great though
---

### Large Model — RTMDet-X
**Config:** `rtmdet_x_8xb32-300e_coco.py`

| Metric | Value |
|------|------|
| **Weights Size** | 396 MB |
| **Inference Speed (CPU)** | 0.5 it/s |
| **True Positives (TP)** | 10 |
| **False Positives (FP)** | 0 |
| **Precision** | **1.00** |

- most suitable: slow, low FP but still high TP
- smaller footprint than RCNN (510MB vs 396 MB)
- since I evaluated "only" 1000 images, the inference would still only take 2000 seconds of time; i.e. 30minutes 
- to summarize: strong overall performance
This model will be used for **further processing and experimentation**
---
