# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os

import torch
from ultralytics import YOLO


class ModelHandler:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(os.environ["YOLO_WEIGHTS"])
        self.labels = self.model.names
        # pose models: keypoints of each object are returned as a separate "points" shape
        self.keypoints_label = os.environ.get("YOLO_KEYPOINTS_LABEL", "Point")
        self.keypoints_threshold = float(os.environ.get("YOLO_KEYPOINTS_THRESHOLD", "0.5"))

    def infer(self, image, threshold):
        prediction = self.model.predict(
            image, conf=threshold, device=self.device, half=self.device == 0, verbose=False
        )[0]

        boxes = prediction.boxes
        keypoints = prediction.keypoints
        results = []
        for i, (box, cls, conf) in enumerate(
            zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
        ):
            results.append(
                {
                    "confidence": str(conf),
                    "label": self.labels[int(cls)],
                    "points": box,
                    "type": "rectangle",
                }
            )

            if keypoints is None or keypoints.conf is None:
                continue

            # objects that were trained without keypoints predict them with low confidence
            kpt_conf = keypoints.conf[i]
            if float(kpt_conf.mean()) < self.keypoints_threshold:
                continue

            results.append(
                {
                    "confidence": str(float(kpt_conf.mean())),
                    "label": self.keypoints_label,
                    "points": keypoints.xy[i].flatten().tolist(),
                    "type": "points",
                }
            )
        return results
