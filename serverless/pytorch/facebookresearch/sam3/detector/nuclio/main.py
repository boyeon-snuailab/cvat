# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

# SAM 3 does not fit into GPU memory twice on small GPUs, so this function
# holds no model and forwards requests to the SAM 3 interactor function instead.

import json
import os
import urllib.request

import yaml


def init_context(context):
    context.logger.info("Init context...  0%")

    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig["metadata"]["annotations"]["spec"]
    context.user_data.labels = [item["name"] for item in json.loads(labels_spec)]
    context.user_data.backend_url = os.environ["SAM3_BACKEND_URL"]

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Forward SAM 3 text-prompted segmentation")
    data = event.body
    payload = {
        "image": data["image"],
        "threshold": float(data.get("threshold", 0.5)),
        "text_labels": context.user_data.labels,
    }
    request = urllib.request.Request(
        context.user_data.backend_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read()

    return context.Response(
        body=body, headers={}, content_type="application/json", status_code=200
    )
