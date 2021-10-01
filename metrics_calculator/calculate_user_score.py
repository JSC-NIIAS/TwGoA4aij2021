import os
from scorer import Scorer
public_image_names=[file for file in os.listdir("/path_to_public_images")]
private_image_names=[file for file in os.listdir("/path_to_private_images")]
path_to_gt_jsons="/path_to_gt_jsons"
path_to_gt_masks="/path_to_gt_masks"
path_to_user_pred_jsons="/path_to_user_pred_images"
path_to_user_pred_masks="/path_to_user_pred_masks"
gt_names=[file for file in os.listdir(path_to_gt_jsons)]
pred_names=[file for file in os.listdir(path_to_user_pred_jsons)]
assert(len(gt_names)==len(pred_names))
assert(len(list(set(gt_names) & set(pred_names)))==len(gt_names))
gt_names=[file for file in os.listdir(path_to_gt_masks)]
pred_names=[file for file in os.listdir(path_to_user_pred_masks)]
assert(len(gt_names)==len(pred_names))
assert(len(list(set(gt_names) & set(pred_names)))==len(gt_names))
metrics_calculator = Scorer(public_image_names,private_image_names)
for file in os.listdir(path_to_user_pred_masks):
    metrics_calculator.calculate_panoptic_quality(os.path.join(path_to_gt_jsons,file+".json"),os.path.join(path_to_user_pred_jsons,file+".json"),os.path.join(path_to_gt_masks,file),os.path.join(path_to_user_pred_masks,file))
public_score,private_score = metrics_calculator.calculate_result()