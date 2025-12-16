## 1. Activate conda environment
1. Inside /applied-deep-learning/mmdetecton
```
   conda activate openmmlab
```

## 2. Download model
1. Select model from folder /configs
    - e.g. /configs/rtmdet/rtmdet_m_8xb32-300e_coco
2. Inside folder /applied-deep-learning/mmdetection
```
    mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest .
```
3. See generated files (one .py file being the model; one .pth file being the weights)
   - e.g. in /mmdetection/rtmdet_m_8xb32-300e_coco.py and rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth

## 3. Execute model
1. Select model (.py and .pth)
2. Edit mmdet.py
3. Run mmdet.py
```
   python3 mmdet.py
```


# Test Results for Micro Dataset
- Comparing the 3 largest on the microdataset
  - Medium
    - rtmdet_m_8xb32-300e_coco.py
    - file size of weights-file: 224 MB
    - Performance: 1.3 it/s on CPU (1.3 images per second)
    - TP: 8
    - FP: 0
    - Precision (TP/(TP+FP)): 1.0
  - RCNN
    - cascade-rcnn_x101_64x4d_fpn_20e_coco.py
    - file size of weights-file: 510 MB
    - Performance: 0.1 it/s on CPU
    - TP: 17
    - FP: 9
    - Precision (TP/(TP+FP)): 0.65
  - Large
    - rtmdet_x_8xb32-300e_coco.py
    - file size of weights-file: 396 MB
    - Performance: 0.5 it/s on CPU
    - TP: 10
    - FP: 0
    - Precision (TP/(TP+FP)): 1.0
Conclusion: The large model (rtmdet_x_8xb32-300e_coco.py) will be chosen for further processing.