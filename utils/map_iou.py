import numpy as np
from tqdm import tqdm
import nms
import torch


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


# helper function to calculate IoU
def special_iou(box1, box2):
    x11, y11, w12, h12 = box1
    x21, y21, w22, h22 = box2
    y22, x22 = y21 + h22, x21 + w22
    y12, x12 = y11 + h12, x11 + w12
    area1 = w12 * h12
    area2 = w22 * h22

    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)


def compute_maps(df, bbxr, clfr=None, k=2, coeff=None, iou_thresh=.3):
    if coeff is None:
        coeff = np.sum(np.argmax(clfr, axis=1) == 1) / clfr.shape[0]
        dist = np.bincount(np.argmax(clfr, axis=1))

    max_score = [max([pred['score'] for pred in patient]) for patient in bbxr]
    threshold = sorted(max_score, reverse=True)[int(len(bbxr) * coeff)]

    mAPs = list()
    for pid, sample in tqdm(enumerate(bbxr)):
        key = sample[0]['image_id'].split('.')[0]
        annot = df.query('patientId==@key').drop(['patientId'], axis=1).values.astype(np.int)
        annot[:, 2] += annot[:, 0]
        annot[:, 3] += annot[:, 1]

        if 'bbox' in sample[0].keys():
            scores_over_thresh = np.array([k['score'] > .05 for k in sample if k['score']])
            uzft = np.array([
                np.concatenate([
                    [k['score']], k['bbox']
                ], axis=0) 
                if len(k['bbox']) else
                np.zeros((5, ))
                for k in sample
            ])[scores_over_thresh]
            uzft = np.concatenate([np.ones((uzft.shape[0], 1)), uzft], axis=1)

            chosen = list()

            if len(uzft):
                for i in range(4):
                    ref_bbx = uzft[np.argmax(uzft[:, 1])]
                    if ref_bbx[1] < threshold:
                        break
                    zious = np.array([special_iou(ref_bbx[2:], u[2:]) for u in uzft])
                    selected = uzft[zious > .55]
                    
                    selected = (selected[:, 2:] * selected[:, 1:2]).sum(axis=0) / selected[:, 1].sum()

                    selected = ref_bbx[2:]
                    chosen.append(np.concatenate([ref_bbx[:2], selected]))
                    uzft = uzft[zious < .3]
                    if len(uzft) == 0:
                        break
                chosen = np.array(chosen)
            if (not len(uzft)) or (not len(chosen)):
                top_k_annots = np.zeros((0, 4))
                score = []
            else:
                top_k_annots = chosen[:, 2:]
                score = chosen[:, 1]
        else:
            top_k_annots = np.zeros((0, 4))
            score = []
        areas = (top_k_annots[:, 2] - top_k_annots[:, 0]) * (top_k_annots[:, 3] - top_k_annots[:, 1])
        top_k_annots = top_k_annots[areas > 0]
        score = np.array(score)[areas > 0]
        mAPs.append(map_iou(annot, top_k_annots, score))
    return np.mean([mAP for mAP in mAPs if mAP is not None]), threshold
