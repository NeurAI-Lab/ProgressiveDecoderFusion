import torch

from detectron2.structures import Instances, Boxes, PolygonMasks


class FCOSTargetTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, bboxes=None, labels=None, inst_depths=None,
                 inst_masks=None):
        target = Instances(self.img_size)
        target.gt_boxes = Boxes(bboxes)
        labels = torch.tensor(labels).long()
        target.gt_labels = labels
        if inst_depths is not None:
            inst_depths = torch.tensor(inst_depths)
            target.gt_inst_depth = inst_depths
        if inst_masks is not None:
            if type(inst_masks) is list and len(inst_masks) != 0:
                inst_masks = torch.stack(inst_masks)
            elif isinstance(inst_masks, PolygonMasks):
                pass
            target.gt_inst_seg = inst_masks
        return target, []


class InstSegTargetTransform:
    def __init__(self, cfg, image_size, **kwargs):
        self.cfg = cfg
        self.image_size = image_size

    def __call__(self, boxes, inst_masks):
        if inst_masks.ndim == 3:
            return [inst_masks[:, :, i] for i in
                    range(list(inst_masks.shape)[2])]
        else:
            return []

    def decode_inst_mask(self, box, en_mask_ts, image_size=None):
        # TODO: complete this...
        raise NotImplementedError
