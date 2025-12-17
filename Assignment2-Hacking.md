## Summarized Workflow
A detailed technical workflow can be found in README.md, here is a brief overview with the essential aspects:

I chose the "Bring your own data" approach:

**Here is a link to the dataset and to my large model files:** [Dropbox: Large files from project](https://www.dropbox.com/scl/fo/lxhoqpkq92szqxygj1564/AMBnRrU8rcyKcxuMHnb4j9g?rlkey=bdcnr2hto9ao7dwjpjwchjs1m&st=zff5o0dj&dl=0) 

If you want to reproduce my work, this is how it's done:
1. Use MMDet model to run object detection of medium dataset
2. Generate a labels.csv file by processing the output of the model above
3. Manually label the entire 1000 images of the testset or used the committed labels.csv file
4. Use the generated labels.csv file to train a new ResNet50 model which is already pre-trained
5. The training of the ResNet50 model will generate a model file which is the final result of this project.

## Error metric I specified
- Initially accuracy, however after conducting more research, I decided for the F1 score.
- I calculate multiple metrics for my ResNet50 model though

## Target value of F1 metric that I want to achieve
- F1: above 0.9 (I did not really think about it that much; it seemed like a high goal)

## Actually achieved value of F1-score
- Train: F1=0.918
- Validation: F1=0.889

## Detailed task breakdown with amount of time I spent on each task
The following lists my tasks in chronological order and adds the approx time spent 

- [3h] **Prerequisites and setup**: Trying to get MMDet to work
  - sadly I started of with the MMDet repo instead of my own, which led to an embarassing problems as soon as I wanted to push my commits
- [2h] **Model-evaluation**: Running many models for details
  - 60% threshold seemed good for accurately detecting person objects
  - Results can be found in ./evaluations/evaluation-mmdet-models.md
  - Decided for _rtmdet_x_8xb32-300e_coco.py_
- [1.5h] **Refining mmdet.py**: Ran mmdet.py (which runs the object detection models) on small dataset to refine how it generates output (threshold of 60%, location of target directory,...)
- [1.5h] **Running object detection on real dataset**: Ran all images with it, manually cross checked results
  - Info: Running the object detection model requires a /input folder storing the images and a /output/vis for storing the images with the object detection on it as well as the /output/preds for
  - storing .json files which have important metadata such as detected objects (incl. type of detected object; such as person or car), threshold of the prediction as well
  - as the coordinates of the bounding boxes.
- [0.5h] **Gathered feedback for Assignment1**: Pivot with professor after feedback of assignment 1: instead of training a new model from scratch with the data from mmdet, I decided to use a pre-trained model (CNN, Vision Transformer) and train the model to label new images as **interesting/not interesting**
- [3.5h] **Deep research**: Investigated further into /preds files and researched how to parse it to gain many features
  - The structure was more difficult than expected, so it took longer
- [3h] **Enhanced research about labels** Researched approach where I can put my additional labels best into action, these labels are:
  - Number of detected persons (num_person)
  - Number of detected vehicles (num_vehicles)
  - Average confidence score for person detections (avg_person_conf)
  - Total number of bounding boxes (total_boxes)
- [4h] **Generating labels**: Enhanced automated processing of /preds files and create a **labels.csv** which default 0 for interesting/non interesting section
- [6h] **!!! Labeling, core part of my project** labeled approx 1000 images with the following approach
  - This sounds easy at first, but I had to look through the results of object detection /vis files, which get generated alongside /preds files
  - Decision metric:
  - If more than 1 person detected by detection model: mark as interesting
  - Then I manually looked through the remaining images with 1 detected person and marked them with "1" (interesting) or "0"(not interesting)
  - If the weather is good on image with 1 person: mark as interesting
  - If the weather is bad on image with 1 person or person is not really visible in the middle of the image: mark as not interesting
  - Find strong positives and strong negatives as discussed in the feedback session with the professor. I documented those strong results and migh present them in the final presentation
- [3h] **Generating an advanced model with hybrid architecture (resnet50) out of dataset**: Decided for resnet50 with normal images and /preds results from MMDet- 
  - Goal/Motivatoin: use the raw images (without object detection), but use the processed data from the /preds files to generate good predictions
  - Details on Resnet50: Multi-modal classification model using a pre-trained ResNet50 for image features and a separate MLP for metadata features.
      - Idea: learn two distinct types of data simultaneously
      - Backbone: The models uses a pre-trained ResNet-50 backbone (trained on ImageNet) to extract rich features from the image, it loads weights which are pre-trained on that dataset
          - I decided to use _--freeze-backbone_ to freeze the backbone; this prevents overfitting
      - After googling/chatGPTing around, I came across a very interesting approach/architecture:
      - Small Metadata Network (nn.Sequential) for structured data from **labels.csv** --> great for leveraging the rich results I already had from the /preds
      - Classification Head (nn.Sequential) that operates on the fused features
      - Regularization was added to prevent overfitting (important!)
      - Binary target variable: interesting/not interesting
- [1h] **Design of model evaluation**: Thought about how to evaluate the model.
  - I had a strong focus on accuracy, precision, recall and f1 score (I learned those 4 in university)
  - I decided to mainly focus on the F1 score since it has a balance between precision and recall and is broadly used across the industry
- [1h] **Running the model**: Running the model with the hyperparameter of 10 epochs
- [2h] **Optimized model**: Improved results by using more epochs (adjusted hyperparameter of epochs from 10 to 30):
```
Epoch 01/30 | Training: train-loss 0.3571, train-accuracy 0.855 train-precision 0.738, train-recall 0.403, train-F1 0.521 | Validation: val-loss 0.1474, val-accuracy 0.944 val-precision 0.850 val-recall 0.872 val-F1 0.861
Epoch 10/30 | Training: train-loss 0.0688, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931 | Validation: val-loss 0.1900, val-accuracy 0.934 val-precision 0.861 val-recall 0.795 val-F1 0.827
Epoch 30/30 | Training: train-loss 0.0798, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918 | Validation: val-loss 0.2196, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
```
- [3h] **Brushing up project for submission**:
  - Sadly I had to fix a git issue (I was working on the official repo of MMDet when I commited my changes, so I had to init a new repo
  - Brushing things up, creating documents, formatted md files