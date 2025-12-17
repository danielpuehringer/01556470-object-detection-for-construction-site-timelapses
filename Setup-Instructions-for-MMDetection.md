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