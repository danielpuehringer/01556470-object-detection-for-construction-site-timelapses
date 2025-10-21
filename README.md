# Scalable object detection for construction site images for timelapse videos
A student project by [Daniel Martin Pühringer](dapinger.at/cv) @ [TU Vienna](https://informatics.tuwien.ac.at/) for the course [Applied Deep Learning(194.077)](https://tiss.tuwien.ac.at/course/educationDetails.xhtml?dswid=9885&dsrid=874&semester=2025W&courseNr=194077) by [Alexander Pacha](https://www.linkedin.com/in/apacha/).

## Description
This project aims to helping users to save time on their generation of timelapse videos of construction sites.

### Problem
Since timelapse videos of construction sites require best suited images (approx. 1200 images for a one minute video) of large image datasets, video producers have to manually view all images and select only a small percentage (approx. 5%) which seem to be the most suitable ones for a timelapse video.
<img src="https://github.com/user-attachments/assets/4cae9512-cd29-4c16-bc21-a32450c9add4" alt="app-hauptkamera_00Zne_2019-11-19_14_44_19_745" width="50%">

### General idea and proposed solution
In order to provide value to users, trained ML models (see below for details) could be applied to pre-select interesting images which indicate a high amount of activity (i.e. many people/construction vehicles on a picture) and help user to find suitable images for a given dataset of construction sites. This pre-selection would reduce the time of carefully previewing and selecting images and therefore reduce the production time of timelapse videos.

## Project Details:
### Project type: Bring your own data
There are four categories from which to choose from:
- **Bring your own data: Provide a suitable dataset and annotate them for supervised learning**
- Bring your own method: re-implement a neural network architecture for a given dataset to improve results
- Beat the classics: solve an existing problem (where deep learning is not used) by applying deep learning to do the same thing and try to improve results
- Beat the stars: Implement a new deep learning algorithm which can be found in scientific papers and try to beat the state of the art

### Approach
Due to my personal interest I would like to slightly adjust my chosen project type.
I already have approx. 30.000 images of one construction site over a timespan of multiple months. Due to privacy issues this dataset will not be made publicly availalbe. However it will be shared with the lecturers.
Due to the vast amount of images, I am planning to fully automate the annotation of the images and focus on models which enable a high throughput of images. My idea is to use the [MMDetection](https://mmdetection.readthedocs.io/en/latest/overview.html) library which offers existing models (see [user guide on using trained models](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html) as well as [the matching github repository](https://github.com/open-mmlab/mmdetection) for details) and was recommended in the lecture. Based on [these performance benchmarks](https://mmdetection.readthedocs.io/en/latest/model_zoo.html#training-speed-benchmark) the MMDetection implementation would also be suitable due to its high throughput of images.
After this fully automated approach to generate data which can be used for learning, I would like to train a new CNN model similar to the code shown in the lecture. I will use PyTorch for this, as seen in [the lecture on CNNs](https://youtu.be/da9PA7mtZwo?si=0VxtIwgICjoZMDQx&t=1260).

The freshly trained model should not only be able detect objects, but also intended to count them for each image. This would mean that the model has to store one additional integer for each image.

## Estimations
- Prerequisites
  - Downloading the dataset and installing required software
- 

## Scientific literature
