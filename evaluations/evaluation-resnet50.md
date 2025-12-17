# Resnet50: Model Training Performance Summary

This document summarizes the performance of the machine learning model trained for **30 epochs**.  
Metrics are reported for both **training** and **validation** sets to evaluate learning behavior and generalization.
The entire source for the calculations can be found at the very bottom of this file.

---

## Metrics Used

- **Loss** – Optimization objective (the lower, the better)
- **Accuracy** – Overall correctness
- **Precision** – Correct positive predictions
- **Recall** – Ability to capture positive cases
- **F1 Score**(primary metric I want to optimize) – Is calculated with precision and recall ([source](https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/)). It's a famous metric for evaluating ML Models. I use it as my primary metric.

---

## Performance at multiple epoch stages
### Epoch 1 (Early Training)

| Dataset     | Loss   | Accuracy | Precision | Recall | **F1 Score** |
|-------------|--------|----------|-----------|--------|--------------|
| Training    | 0.3571 | 0.855    | 0.738     | 0.403  | 0.521        |
| Validation  | 0.1474 | 0.944    | 0.850     | 0.872  | 0.861        |

**Observation:**  
- The model starts with moderate accuracy.
- Recall is initially low, indicating early difficulty in detecting positive cases.
- Validation performance is strong from the beginning, suggesting good initial generalization.
---

### Epoch 10 (in the middle)

| Dataset     | Loss   | Accuracy | Precision | Recall | **F1 Score** |
|-------------|--------|----------|-----------|--------|--------------|
| Training    | 0.0688 | 0.972    | 0.902     | 0.961  | 0.931        |
| Validation  | 0.1900 | 0.934    | 0.861     | 0.795  | 0.827        |

**Observation:**  
- Training metrics significantly improve.

---
### Epoch 30 (Final Epoch, Final Model)

| Dataset     | Loss   | Accuracy | Precision | Recall | **F1 Score** |
|-------------|--------|----------|-----------|--------|--------------|
| Training    | 0.0798 | 0.967    | 0.895     | 0.942  | 0.918        |
| Validation  | 0.2196 | 0.954    | 0.857     | 0.923  | 0.889        |

**Observation:**  
- Validation performance improves.
- High recall and F1 score indicate strong positive-class detection.
- Training and validation metrics remain close → good generalization.

---

## Conclusion
- Great progress: The model converges effectively within 30 epochs.
- Overfitting: No severe overfitting observed.
- F1-score: Final validation **F1 score of 0.889** indicates a strong balance between all evaluation metrics.
- The model seems suitable for deployment and ready for Assignment3.
---


## Source for calculations above:
python resnet50.py \
      --images-dir ../datasets/medium/original \
      --csv-path ../datasets/medium/labels.csv \
      --epochs 30 \
      --batch-size 8 \
      --max-boxes 10 \
      --freeze-backbone
Using device: cpu
Epoch 01/30 | Training: train-loss 0.3571, train-accuracy 0.855 train-precision 0.738, train-recall 0.403, train-F1 0.521 | Validation: val-loss 0.1474, val-accuracy 0.944 val-precision 0.850 val-recall 0.872 val-F1 0.861
Epoch 02/30 | Training: train-loss 0.1169, train-accuracy 0.961 train-precision 0.873, train-recall 0.935, train-F1 0.903 | Validation: val-loss 0.1361, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 03/30 | Training: train-loss 0.1242, train-accuracy 0.962 train-precision 0.887, train-recall 0.922, train-F1 0.904 | Validation: val-loss 0.1306, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 04/30 | Training: train-loss 0.1135, train-accuracy 0.954 train-precision 0.873, train-recall 0.896, train-F1 0.885 | Validation: val-loss 0.1858, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 05/30 | Training: train-loss 0.0942, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918 | Validation: val-loss 0.1365, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 06/30 | Training: train-loss 0.0925, train-accuracy 0.964 train-precision 0.894, train-recall 0.929, train-F1 0.911 | Validation: val-loss 0.1585, val-accuracy 0.954 val-precision 0.841 val-recall 0.949 val-F1 0.892
Epoch 07/30 | Training: train-loss 0.0838, train-accuracy 0.971 train-precision 0.897, train-recall 0.961, train-F1 0.928 | Validation: val-loss 0.1586, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 08/30 | Training: train-loss 0.0937, train-accuracy 0.963 train-precision 0.898, train-recall 0.916, train-F1 0.907 | Validation: val-loss 0.1505, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 09/30 | Training: train-loss 0.0770, train-accuracy 0.969 train-precision 0.901, train-recall 0.948, train-F1 0.924 | Validation: val-loss 0.1521, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 10/30 | Training: train-loss 0.0688, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931 | Validation: val-loss 0.1900, val-accuracy 0.934 val-precision 0.861 val-recall 0.795 val-F1 0.827
Epoch 11/30 | Training: train-loss 0.0702, train-accuracy 0.969 train-precision 0.917, train-recall 0.929, train-F1 0.923 | Validation: val-loss 0.1636, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 12/30 | Training: train-loss 0.0781, train-accuracy 0.966 train-precision 0.894, train-recall 0.935, train-F1 0.914 | Validation: val-loss 0.1615, val-accuracy 0.934 val-precision 0.861 val-recall 0.795 val-F1 0.827
Epoch 13/30 | Training: train-loss 0.0720, train-accuracy 0.962 train-precision 0.903, train-recall 0.903, train-F1 0.903 | Validation: val-loss 0.1759, val-accuracy 0.954 val-precision 0.841 val-recall 0.949 val-F1 0.892
Epoch 14/30 | Training: train-loss 0.0737, train-accuracy 0.966 train-precision 0.899, train-recall 0.929, train-F1 0.914 | Validation: val-loss 0.2194, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 15/30 | Training: train-loss 0.0664, train-accuracy 0.973 train-precision 0.903, train-recall 0.968, train-F1 0.934 | Validation: val-loss 0.1799, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Epoch 16/30 | Training: train-loss 0.0727, train-accuracy 0.975 train-precision 0.924, train-recall 0.948, train-F1 0.936 | Validation: val-loss 0.2262, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 17/30 | Training: train-loss 0.0672, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931 | Validation: val-loss 0.2284, val-accuracy 0.939 val-precision 0.846 val-recall 0.846 val-F1 0.846
Epoch 18/30 | Training: train-loss 0.0644, train-accuracy 0.969 train-precision 0.901, train-recall 0.948, train-F1 0.924 | Validation: val-loss 0.3002, val-accuracy 0.929 val-precision 0.903 val-recall 0.718 val-F1 0.800
Epoch 19/30 | Training: train-loss 0.0763, train-accuracy 0.966 train-precision 0.894, train-recall 0.935, train-F1 0.914 | Validation: val-loss 0.2131, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 20/30 | Training: train-loss 0.0599, train-accuracy 0.975 train-precision 0.919, train-recall 0.955, train-F1 0.936 | Validation: val-loss 0.1844, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Epoch 21/30 | Training: train-loss 0.0536, train-accuracy 0.971 train-precision 0.897, train-recall 0.961, train-F1 0.928 | Validation: val-loss 0.2101, val-accuracy 0.944 val-precision 0.850 val-recall 0.872 val-F1 0.861
Epoch 22/30 | Training: train-loss 0.0527, train-accuracy 0.976 train-precision 0.930, train-recall 0.948, train-F1 0.939 | Validation: val-loss 0.2123, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Epoch 23/30 | Training: train-loss 0.0639, train-accuracy 0.972 train-precision 0.907, train-recall 0.955, train-F1 0.930 | Validation: val-loss 0.1956, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 24/30 | Training: train-loss 0.0502, train-accuracy 0.977 train-precision 0.915, train-recall 0.974, train-F1 0.943 | Validation: val-loss 0.1960, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 25/30 | Training: train-loss 0.0480, train-accuracy 0.978 train-precision 0.920, train-recall 0.974, train-F1 0.946 | Validation: val-loss 0.2334, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Epoch 26/30 | Training: train-loss 0.0597, train-accuracy 0.980 train-precision 0.937, train-recall 0.961, train-F1 0.949 | Validation: val-loss 0.1622, val-accuracy 0.939 val-precision 0.865 val-recall 0.821 val-F1 0.842
Epoch 27/30 | Training: train-loss 0.0696, train-accuracy 0.967 train-precision 0.905, train-recall 0.929, train-F1 0.917 | Validation: val-loss 0.2003, val-accuracy 0.959 val-precision 0.860 val-recall 0.949 val-F1 0.902
Epoch 28/30 | Training: train-loss 0.0562, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931 | Validation: val-loss 0.2421, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Epoch 29/30 | Training: train-loss 0.0687, train-accuracy 0.978 train-precision 0.936, train-recall 0.955, train-F1 0.945 | Validation: val-loss 0.1836, val-accuracy 0.944 val-precision 0.868 val-recall 0.846 val-F1 0.857
Epoch 30/30 | Training: train-loss 0.0798, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918 | Validation: val-loss 0.2196, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Saved: models/resnet50_fused_image_meta.pth
