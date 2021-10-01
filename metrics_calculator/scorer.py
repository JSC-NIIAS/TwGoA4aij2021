import json
import numpy as np
from PIL import Image

class Scorer(object):
    def __init__(self,public_image_names,private_images_names):
        self.public_image_names=public_image_names
        self.public_scores=[]
        self.private_images_names=private_images_names
        self.private_scores=[]
        self.iou_threshold=0.5
        self.seg_classes_idx = [6, 7, 10]

    def compute_iou_bbox(self,bb1, bb2):
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def find_ious_bbox(self,bbox, pred_bboxes):
        ious = [self.compute_iou_bbox(bbox, pred_bboxes[i]) for i in range(len(pred_bboxes))]
        if len(ious) == 0:
            return None, None
        iou_max = max(ious)
        if iou_max > self.iou_threshold:
            return ious.index(iou_max), iou_max
        else:
            return None, None

    def mask_iou(self,mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / (np.sum(union) + 1e-10)
        return iou

    def open_files(self,gt_bbox_json, pred_bbox_json, gt_mask, pred_mask):
        with open(gt_bbox_json) as gt_bbox_ann:
            gt_bbox_data = json.load(gt_bbox_ann)
        gt_bbox_ann.close()
        with open(pred_bbox_json) as pred_bbox_ann:
            pred_bbox_data = json.load(pred_bbox_ann)
        pred_bbox_ann.close()
        gt_mask = np.array(Image.open(gt_mask).convert("L"), dtype=np.uint8)
        pred_mask = np.array(Image.open(pred_mask).convert("L"), dtype=np.uint8)
        return gt_bbox_data,pred_bbox_data,gt_mask,pred_mask

    def calculate_panoptic_quality(self,gt_bbox_json, pred_bbox_json, gt_mask, pred_mask):
        true_positive = 0
        false_negative = 0
        false_positive = 0
        ious = []
        gt_masks = []
        pred_masks = []
        classes_gt = {"Human": [], "Wagon": [], "Car": [], "SignalE": [], "SignalF": [], "TrailingSwitchR": [],
                      "TrailingSwitchL": [], "TrailingSwitchNV": [], "FacingSwitchR": [], "FacingSwitchL": [],
                      "FacingSwitchNV": []}
        classes_gt_copy = {"Human": [], "Wagon": [], "Car": [], "SignalE": [], "SignalF": [], "TrailingSwitchR": [],
                           "TrailingSwitchL": [], "TrailingSwitchNV": [], "FacingSwitchR": [], "FacingSwitchL": [],
                           "FacingSwitchNV": []}
        classes_pred = {"Human": [], "Wagon": [], "Car": [], "SignalE": [], "SignalF": [], "TrailingSwitchR": [],
                        "TrailingSwitchL": [], "TrailingSwitchNV": [], "FacingSwitchR": [], "FacingSwitchL": [],
                        "FacingSwitchNV": []}
        gt_bbox_data, pred_bbox_data, gt_mask, pred_mask=self.open_files(gt_bbox_json, pred_bbox_json, gt_mask, pred_mask)

        for bbox in gt_bbox_data["bb_objects"]:
            classes_gt[bbox["class"]].append(bbox)
        for bbox in pred_bbox_data["bb_objects"]:
            classes_pred[bbox["class"]].append(bbox)

        for class_name in classes_gt.keys():
            for bbox in classes_gt[class_name]:
                max_iou_index_pred, iou = self.find_ious_bbox(bbox, classes_pred[class_name])
                if max_iou_index_pred is not None:
                    classes_pred[class_name].pop(max_iou_index_pred)
                    ious.append(iou)
                    true_positive += 1
                else:
                    classes_gt_copy[class_name].append(bbox)
        false_negative += sum([len(classes_gt_copy[class_name]) for class_name in classes_gt_copy.keys()])
        false_positive += sum([len(classes_pred[class_name]) for class_name in classes_gt.keys()])

        for class_idx in self.seg_classes_idx:
            gt_masks.append(gt_mask[gt_mask == class_idx].astype(int))
            pred_masks.append(pred_mask[pred_mask == class_idx].astype(int))
        mask_ious = [self.mask_iou(gt_masks[i], pred_masks[i]) for i in range(len(self.seg_classes_idx))]
        for iou in mask_ious:
            if iou > self.iou_threshold:
                ious.append(iou)
                true_positive += 1
            if iou > 0 and iou <= self.iou_threshold:
                false_positive += 1
            if iou == 0:
                false_negative += 1

        PQ = sum(ious) / (true_positive + 0.5 * false_positive + 0.5 * false_negative)
        if gt_bbox_json.strip(".json") in self.public_image_names:
            self.public_scores.append(PQ)
        if gt_bbox_json.strip(".json") in self.public_image_names:
            self.private_scores.append(PQ)

    def calculate_result(self):
        return sum(self.public_scores)/len(self.public_scores),sum(self.private_scores)/len(self.private_scores)