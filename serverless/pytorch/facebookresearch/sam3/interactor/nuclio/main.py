# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import base64
import io
import json

import numpy as np
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    # requests forwarded by the SAM 3 text detector function, which shares this model
    if "text_labels" in data:
        results = context.user_data.model.detect(
            image, data["text_labels"], float(data.get("threshold", 0.5))
        )
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    pos_points = data.get("pos_points") or []
    neg_points = data.get("neg_points") or []
    obj_bbox = data.get("obj_bbox")

    # CVAT 2.37 expects the legacy interactor response: a 0/1 mask cropped to
    # its inclusive bounds [xtl, ytl, xbr, ybr] in image coordinates
    mask = None
    if pos_points or obj_bbox:
        mask = context.user_data.model.handle(image, pos_points, neg_points, obj_bbox)

    ys, xs = np.nonzero(mask) if mask is not None else ([], [])
    if len(xs):
        bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        cropped = mask[bounds[1] : bounds[3] + 1, bounds[0] : bounds[2] + 1]
    else:
        bounds = [0, 0, 0, 0]
        cropped = np.zeros((1, 1), dtype=np.uint8)

    return context.Response(
        body=json.dumps({"mask": cropped.tolist(), "bounds": bounds}),
        headers={},
        content_type="application/json",
        status_code=200,
    )
