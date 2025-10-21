# Scalable object detection for construction site images for timelapse videos
A student project by [Daniel Martin Pühringer](dapinger.at/cv) @ [TU Vienna](https://informatics.tuwien.ac.at/) for the course [Applied Deep Learning(194.077)](https://tiss.tuwien.ac.at/course/educationDetails.xhtml?dswid=9885&dsrid=874&semester=2025W&courseNr=194077) by [Alexander Pacha](https://www.linkedin.com/in/apacha/).

## Description
This project aims to helping users to save time on their generation of timelapse videos of construction sites.

### Problem
Since timelapse videos of construction sites require best suited images (approx. 1200 images for a one minute video) of large image datasets, video producers have to manually view all images and select only a small percentage (approx. 5%) which seem to be the most suitable ones for a timelapse video.

### Image data (examples)
<img src="https://github.com/user-attachments/assets/4cae9512-cd29-4c16-bc21-a32450c9add4" alt="app-hauptkamera_00Zne_2019-11-19_14_44_19_745" width="25%">

![app-hauptkamera_00Zne_2020-06-09_16_25_29_619](https://github.com/user-attachments/assets/e6e43f10-921e-45b6-a15a-e77464118f9c)
![app-hauptkamera_00Zne_2020-10-22_16_45_03_816](https://github.com/user-attachments/assets/e285c8b6-f40f-4689-8fdd-29f893d98fea)
![app-hauptkamera_00Zne_2021-05-11_17_00_59_886](https://github.com/user-attachments/assets/469b1897-0846-4d1d-9c61-9bef61afe04a)

### General idea and proposed solution
In order to provide value to users, trained ML models (see below for details) could be applied to pre-select interesting images which indicate a high amount of activity (i.e. many people/construction vehicles on a picture) and help user to find suitable images for a given dataset of construction sites. This pre-selection would reduce the time of carefully previewing and selecting images and therefore reduce the production time of timelapse videos.

## Project Details:
### Project type: Bring your own data
There are four categories from which to choose from:
- **Bring your own data: Provide a suitable dataset and annotate them for supervised learning**
- Bring your own method: re-implement a neural network architecture for a given dataset to improve results
- Beat the classics: solve an existing problem (where deep learning is not used) by applying deep learning to do the same thing and try to improve results
- Beat the stars: Implement a new deep learning algorithm which can be found in scientific papers and try to beat the state of the art

### Justification for project type and short summary
Since I have my own data (large amount of images), I want to collect a comporehensive dataset that is suitable for training deep neural networks.
The project includes collecting a large number of samples in an automated (not semi-automated!) way by annotating thousands of images with the help of an already trained object detection model. Based on this dataset, I will run & train a simple neural network.

### Detailed approach
Due to my personal interest I would like to slightly adjust my chosen project type.
I already have approx. 30.000 images of one construction site over a timespan of multiple months. Due to privacy issues this dataset will not be made publicly availalbe. However it will be shared with the lecturers.
Due to the vast amount of images, I am planning to fully automate the annotation of the images and focus on models which enable a high throughput of images. My idea is to use the [MMDetection](https://mmdetection.readthedocs.io/en/latest/overview.html) library which offers existing models (see [user guide on using trained models](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html) as well as [the matching github repository](https://github.com/open-mmlab/mmdetection) for details) and was recommended in the lecture. Based on [these performance benchmarks](https://mmdetection.readthedocs.io/en/latest/model_zoo.html#training-speed-benchmark) the MMDetection implementation would also be suitable due to its high throughput of images.
After this fully automated approach to generate data which can be used for learning, I would like to train a new CNN model similar to the code shown in the lecture. I will use PyTorch for this, as seen in [the lecture on CNNs](https://youtu.be/da9PA7mtZwo?si=0VxtIwgICjoZMDQx&t=1260).

The freshly trained model should not only be able detect objects, but also intended to count them for each image. This would mean that the model has to store one additional integer for each image.

## Detailed task breakdown and rough time estimations --> Total time estimated lies between 40 and 73 hours
- Prerequisites (2-5 hours)
  - Downloading the dataset and installing required software
  - Splitting the dataset into multiple sizes (small, medium, large) (e.g. every 200th images should be put into the small dataset)
  - Reading important parts of the MMDetection library by using [this guide](https://mmdetection.readthedocs.io/en/latest/overview.html)
- Generating labels (i.e. bounding boxes) for given data via already trained MMDetection model
  - Installing, deploying, executing and troubleshooting the MMDetection library by following [this tutorial](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html) (2-10 hours)
  - Tweaking the output (by changing parameters) in the desired way (image as input; array with bounding box coordinates or sth similar as output) (2-8 hours)
  - Executing the already trained MMDetection model on the small subset of the training data (1-2 hours)
  - Troubleshooting and adjusting the code so that it can be executed on the medium and large dataset. (1-3 hours)
- Refurbishing and uploading the newly gathered dataset to be used for training a new CNN (2-4 hours)
  - Uploading massive amount of data onto dropbox so that the data is stored properly
  - Troubleshooting (5 hours as a buffer)
- Training a new CNN model based on gathered dataset
  - Familiarizing with pytorch and environment (2-4 hours)
  - Trying out code examples from lecture (1-3 hours)
  - Find suitable CNN architectures by running them on tiny datasets (3 hours)
  - Running multiple promising CNN architectures on the medium/large dataset and compare them regarding performance and ML specific evaluation metrics (accuracy, F1-score, etc.) (10 hours)
  - Adding logic for counting the bounding boxes for each image and storing the data accordingly (1-8 hours)
  - Running most suitable model on large dataset (1 hour; will be run overnight)
- Project wrapup (10 hours)
  - Writing the report
  - Creating the presentation
  - Presentation rehearsal

## Scientific literature
- Papers from the lecture on CNNs
- [Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
- [R‑FCN: Object Detection via Region‐based Fully Convolutional Networks (Dai et al., 2016)](https://arxiv.org/abs/1605.06409)
- [CNN Based 2D Object Detection Techniques: A Review (Zhao et al., 2024)](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1437664/full)
- [LeYOLO: Lightweight, Scalable and Efficient CNN Architecture for Object Detection (Hollard et al., 2024)](https://arxiv.org/html/2406.14239v1)
