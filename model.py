from functools import partial
from typing import List, Dict

import torch
import torch.jit
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import maskrcnn_inference, maskrcnn_loss, fastrcnn_loss

# background=0, building=1
NUM_CLASSES = 2


def forward(self, features, proposals, image_shapes, targets=None, task_heads=None):
    """
    Arguments:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
            assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)
    task_results = {task_head.name: task_head(box_features) for task_head in task_heads}

    result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
    losses = {}
    if self.training:
        assert labels is not None and regression_targets is not None
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets)

        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
        }

        # Calculate losses for all the tasks
        for name, preds in task_results.items():
            actuals = []
            for idxs, target in zip(matched_idxs, targets):
                actuals.append(target[name][idxs])
            actuals = torch.cat(actuals, dim=0)
            # Discount the loss for task so that model still prioritises learning
            # the box / mask properly
            losses[f"loss_{name}"] = F.cross_entropy(preds, actuals)

    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        if len(boxes_per_image) == 1:
            task_results_list = task_results
        else:
            task_results_list = {name: preds.split(boxes_per_image, 0) for name, preds in task_results.items()}

        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "task_results": {name: preds[i] for name, preds in task_results_list.items()}
                }
            )

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            assert matched_idxs is not None
            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        else:
            mask_logits = torch.tensor(0)
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}
        if self.training:
            assert targets is not None
            assert pos_matched_idxs is not None
            assert mask_logits is not None

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(
                mask_logits, mask_proposals,
                gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {
                "loss_mask": rcnn_loss_mask
            }
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

    return result, losses


class ClassificationTaskHead(nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        return self.head(x)


def multitask_maskrcnn_resnet50_fpn(
        **kwargs
):
    task_heads = [
        ClassificationTaskHead(name, num_classes)
        for name, num_classes in [("roof_style", 4), ("roof_material", 4)]
    ]
    task_heads = [head.cuda() for head in task_heads]
    model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES, **kwargs)
    [model.add_module(head.name, head) for head in task_heads]

    model.roi_heads.forward = partial(forward, model.roi_heads, task_heads=task_heads)
    return model
