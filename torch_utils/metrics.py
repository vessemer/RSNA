import numpy as np
import torch
from collections import defaultdict


THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def aggregate(metrics):
    grouped = defaultdict(list)
    keys = metrics[0].keys()
    for key in keys:
        for meter in metrics:
            grouped[key].append(meter[key])

    for key in keys:
        meter = sum([meter for meter in grouped[key]])
        if isinstance(meter, torch.Tensor):
            meter = meter.data.cpu().numpy() 
        grouped[key] = meter / len(grouped[key])
    return grouped


class Metric:
    def __init__(self):
        self.name = self.__class__.__name__

    def __call__(self, pred, target):
        return { self.name: self.estimate(pred, target) }

    def estimate(self, pred, target):
        pass


class IoU(Metric):
    def __init__(self, threshold=.5):
        super().__init__()
        self.threshold = threshold

    def estimate(self, pred, target):
        pred = pred >= self.threshold
        intersection = np.sum((target * pred) > 0)
        union = np.sum((target + pred) > 0) + 1e-7  # avoid division by zero
        return intersection / union


class F2(Metric):
    def __init__(self):
        super().__init__()

    def estimate(self, pred, target):
        # a correct prediction on no ships in image would have F2 of zero (according to formula),
        # but should be rewarded as 1
        if np.sum(target) == np.sum(pred) == 0:
            return 1.

        f2_total = 0.
        for t in THRESHOLDS:
            tp, fp, fn = 0, 0, 0
            ious = {}
            for i, mt in enumerate(target):
                found_match = False
                for j, mp in enumerate(pred):
                    miou = iou(mt, mp)
                    ious[100 * i + j] = miou # save for later
                    if miou >= t:
                        found_match = True
                if not found_match:
                    fn += 1

            for j, mp in enumerate(pred):
                found_match = False
                for i, mt in enumerate(target):
                    miou = ious[100 * i + j]
                    if miou >= t:
                        found_match = True
                        break
                if found_match:
                    tp += 1
                else:
                    fp += 1
            f2 = (5 * tp) / (5 * tp + 4 * fn + fp)
            f2_total += f2

        return f2_total / len(thresholds)
