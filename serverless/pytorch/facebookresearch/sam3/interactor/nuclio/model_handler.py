# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os

import cv2
import numpy as np
import torch
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl : ybr + 1, xtl : xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


class ModelHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=os.environ["SAM3_CHECKPOINT"],
            load_from_HF=False,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=self.device)

    @torch.inference_mode()
    def handle(self, image, pos_points, neg_points, obj_bbox):
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            state = self.processor.set_image(image)

            points = list(pos_points) + list(neg_points)
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_labels = (
                np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
                if points
                else None
            )
            box = np.array(obj_bbox, dtype=np.float32).reshape(4) if obj_bbox else None

            # a single click is ambiguous, so let the model propose several masks
            multimask = box is None and len(points) == 1
            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask,
            )

        return (masks[int(np.argmax(scores))] > 0).astype(np.uint8)

    @torch.inference_mode()
    def detect(self, image, labels, threshold):
        results = []
        self.processor.set_confidence_threshold(threshold)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            # the image embedding is computed once and reused for every text prompt
            state = self.processor.set_image(image)
            for label in labels:
                state = self.processor.set_text_prompt(label, state)
                masks = state["masks"].squeeze(1).cpu().numpy().astype(np.uint8)
                scores = state["scores"].float().cpu().numpy()

                for mask, score in zip(masks, scores):
                    ys, xs = np.nonzero(mask)
                    if xs.size == 0:
                        continue
                    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contour = max(contours, key=cv2.contourArea)
                    if len(contour) < 3:
                        continue

                    results.append(
                        {
                            "confidence": str(float(score)),
                            "label": label,
                            "points": contour.ravel().tolist(),
                            "mask": to_cvat_mask(box, mask),
                            "type": "mask",
                        }
                    )
        return results
