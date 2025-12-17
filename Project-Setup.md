# Installing prerequisistes

## Prerequisites 
- MMDetection: find the full installation guide here [at the official website](https://mmdetection.readthedocs.io/en/latest/get_started.html)
  - In this tutorial you will find a detailed version of the installation steps:
    - Install Miniconda
    - Create and activate a new conda environment:
      ```
      conda create --name openmmlab python=3.8 -y
      conda activate openmmlab
      ```
      - Install Pytorch (either on GPU or CPU platform)
      - Install MIM and MMEngine, MMCV
      - Verify installation

## Project Workflow
1. Use MMDet to run object detection on images
   - This generates a /vis and a /preds folder and a file for each image
   - /vis contains images with bounding boxes drawn on them --> this folder is just for you to visually verify the detection results
   - /preds contains text files resulting from the detection (including bounding boxes, detection objects and type of detected objects)
2. Generate CSV file called labels.csv based on the preds folder
3. Use the generated labels.csv file to train a new ResNet50 model using resnet50.py
   - Run the model training script by executing the following in
    ```bash
    python model/resnet50.py \
      --images-dir ./datasets/medium/original \
      --csv-path ./datasets/medium/labels.csv \
      --epochs 30 \
      --batch-size 8 \
      --max-boxes 10 \
      --freeze-backbone
    ```