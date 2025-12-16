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

