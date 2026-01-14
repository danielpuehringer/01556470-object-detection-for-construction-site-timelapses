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

# Project Workflow (semi-automated with human in the loop)
1. Use MMDet to run object detection on images
   - This generates a /vis and a /preds folder and a file for each image
   - /vis contains images with bounding boxes drawn on them --> this folder is just for you to visually verify the detection results
   - /preds contains text files resulting from the detection (including bounding boxes, detection objects and type of detected objects)
2. Generate CSV file called labels.csv based on the preds folder
3. Use the generated labels.csv file to train a new ResNet50 model using resnet50.py
   - Run the model training script by executing the following in
    ```bash
    python resnet50.py \
      --images-dir ../datasets/medium/original \
      --csv-path ../datasets/medium/labels.csv \
      --epochs 30 \
      --batch-size 8 \
      --max-boxes 10 \
      --freeze-backbone
    ```
4. The training will generate a model file in the outputs/ folder and look like this:
```text
Epoch 30/30 | Training: train-loss 0.0798, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918 | Validation: val-loss 0.2196, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
Saved: models/resnet50_fused_image_meta.pth
```

This the end of the project setup. Since this "Bring your data" approach ends with running a simple neural network, no further inference is planned.

# Run prototype:
To run the React prototype, follow these steps:
1. Install the latest version of Node.js from [nodejs.org](https://nodejs.org/) and NPM (Node Package Manager).
2. Navigate into directory /web-frontend
3. Install node modules by running 'npm install'
4. Start the development server by running 'npm run dev'
5. Open your web browser and go to 'http://localhost:5173' to view the application.