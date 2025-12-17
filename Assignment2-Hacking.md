# TODO add documentation

# Workflow


- Error Metric: 

## amount of time you spent on each task
the following lists my tasks in chronoloigal order and adds the aproxx time spent 

- Trying to get MMDet to work
- running many models (see mmdet.py for a list of all models used
- ran mmdet on small dataset to refine how it generates output (threshold of 60%, 
- results can be found in ./evaluations/evaluation-mmdet-models.md
- decided for a model: rtmdet_x_8xb32-300e_coco.py and ran all images with it
- pivot with professor after feedback of assignment 1: instead of training a new model from scratch with the data from mmdet, instead use a pre trained model (CNN, Vision Transformer) and train the model to label new images as interesting/not interesting
- investigated further into preds file and researched how to parse it to gain many features
- automated processing of preds files and create a labels.csv which default 0 for interesting/non interesting section
	- structure: 
- labeled approx 1000 images with the following approach
- researched approach where I can put my additional labels best into action, these labels are:
	- Number of detected persons (num_person)
	- Number of detected vehicles (num_vehicles)
	- Average confidence score for person detections (avg_person_conf)
	- Total number of bounding boxes (total_boxes)
- decided for resnet50 with normal images and /preds results from MMDet
- resnet50: Multi-modal classification model using a pre-trained ResNet50 for image features and a separate MLP for metadata features.
	- Idea: learn two distinct types of data simoultanieously
	- Backbone: The models uses a pre-trained ResNet-50 backbone (trained on ImageNet) to extract rich features from the image, it loads weichts pre-trained on that dataset
		- i decided to use --freeze-backbone to freeze the backbone; prevents overfitting
	- Small Metadata Network (nn.Sequential) for structured data from labels.csv
	- Classificatoin Head (nn.Sequential) that operates on the fused features
	- Regularization was added to prevent overfitting (important!)
	- Binary target variable: interesting/not interesting
- fought about how to evaluate the model. strong focus on accurary, precision, recall and f1 score (I learned those 4 in university)
- running the model with the hyperparameter of 10 epochs
- improved results by using more epcohs:

Epoch 01/30 | Training: train-loss 0.3571, train-accuracy 0.855 train-precision 0.738, train-recall 0.403, train-F1 0.521 | Validation: val-loss 0.1474, val-accuracy 0.944 val-precision 0.850 val-recall 0.872 val-F1 0.861
Epoch 10/30 | Training: train-loss 0.0688, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931 | Validation: val-loss 0.1900, val-accuracy 0.934 val-precision 0.861 val-recall 0.795 val-F1 0.827
Epoch 30/30 | Training: train-loss 0.0798, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918 | Validation: val-loss 0.2196, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889


## the error metric you specified
- initaly accuracy, however then after research: F1

## the target of that error metric that you want to achieve:
- f1: above 0.9 ( I did not really think about it that much)
- 

## the actually achieved value of that metric
- train-F1 0.918; val-F1 0.889