from mmdet.apis import DetInferencer

inputMicro = 'input/testset-micro'
input_DataSetMedium = 'input/testset-medium'

#Info for test-image.jpg: this image refers to original name 'app-hauptkamera_00Zne_2021-04-29_09_45_54_496.jpg'
inputForTesting = 'input/test-dataset'
outputForTesting = 'outputs/test-dataset-results'

modelTiny = 'rtmdet_tiny_8xb32-300e_coco.py'
weightsTiny = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
outputMicroTiny = 'outputs/testset-micro-tiny'

modelSmall = 'rtmdet_s_8xb32-300e_coco.py'
weightsSmall = 'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
outputMicroSmall = 'outputs/testset-micro-small'

modelMedium = 'rtmdet_m_8xb32-300e_coco.py'
weightsMedium = 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
outputMicroMedium = 'outputs/testset-micro-medium'


modelRcnn = 'cascade-rcnn_x101_64x4d_fpn_20e_coco.py'
weightsRcnn = 'cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
outputMicroRcnn = 'outputs/testset-micro-rcnn'

modelXetraLarge = 'rtmdet_x_8xb32-300e_coco.py'
weightsXetraLarge = 'rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
outputMicroXetraLarge = 'outputs/testset-micro-xetra-large'
output_DataSetMedium_ModelXetraLarge = 'outputs/dataset-medium_model-extra-large'

inferencer = DetInferencer(modelXetraLarge,
                           weightsXetraLarge,
                           device='cpu')
# result = inferencer(input_DataSetMedium, out_dir=output_DataSetMedium_ModelXetraLarge, return_vis=True, pred_score_thr=0.6, draw_pred=True, no_save_pred=False)
result = inferencer(inputForTesting, out_dir=outputForTesting, return_vis=True, pred_score_thr=0.6, draw_pred=True, no_save_pred=False)
# Note: Result has the following json structure:
# 'predictions': [{'labels': [...], 'scores': [...], 'bboxes': [...]}], 'visualization': [array()]}


# NOTE: print(result['predictions'][1]['labels']); should no be possible (at least not for one image), but print(result['predictions'][0]['labels'] is possible
'''print("\n\n============= TEST =============")
print(len(result['predictions'][1]['labels']))
print(len(result['predictions'][1]['scores']))
print(len(result['predictions'][1]['bboxes']))
print(len(result['visualization']))

## see chatgpt:
import json

input_path = "demo.json"
output_path = "demo_thr0.6.json"
score_thr = 0.6

with open(input_path, "r") as f:
    data = json.load(f)

# Depending on how MMDet saved it, you might have:
# - data["predictions"] = list of per-image dicts, OR
# - data = list of per-image dicts directly

if isinstance(data, dict) and "predictions" in data:
    preds = data["predictions"]
else:
    preds = data

for pred in preds:
    labels = pred["labels"]
    scores = pred["scores"]
    bboxes = pred["bboxes"]

    # indices to keep
    keep_idx = [i for i, s in enumerate(scores) if s >= score_thr]

    pred["labels"] = [labels[i] for i in keep_idx]
    pred["scores"] = [scores[i] for i in keep_idx]
    pred["bboxes"] = [bboxes[i] for i in keep_idx]

# write filtered json
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)
    '''