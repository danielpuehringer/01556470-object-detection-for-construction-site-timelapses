# Installing prerequisistes
1. Create and activate a new conda environment:
```
conda activate openmmlab
```

If you don't have this environment yet, create it with:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

2.Run the following command install all required software:
```
pip install -r requirements.txt
```
Note: If you need CUDA support, install PyTorch according to:
https://pytorch.org/get-started/locally/

3. Run the following command to install MMDetection:

4. Run generate_labels_csv

5. Run resnet50-complex to train new model:
python model/resnet50-complex.py \
  --images-dir datasets/test-dataset/test-dataset-input \
  --csv-path datasets/test-dataset/labels.csv \
  --epochs 10 \
  --batch-size 8 \
  --max-boxes 10 \
  --freeze-backbone
