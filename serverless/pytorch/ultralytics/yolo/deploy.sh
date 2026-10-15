#!/bin/bash
# Deploys a custom Ultralytics YOLO detect / pose model (.pt) as a CVAT detector.
# Pose keypoints are returned as one "Point" points shape per object.
# The function name and labels are taken from the weights file, so several
# models can be deployed side by side.
# Usage: ./deploy.sh <path to weights .pt> [display name]

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WEIGHTS="$(realpath "$1")"
MODEL_ID="$(basename "$WEIGHTS" .pt | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
DISPLAY_NAME="${2:-YOLO $(basename "$WEIGHTS" .pt)}"
FUNC_NAME="ultralytics-yolo-$MODEL_ID"
IMAGE="cvat.ultralytics.yolo:latest-gpu"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

nuctl create project cvat --platform local || true
nuctl build "$FUNC_NAME" --project-name cvat --path "$SCRIPT_DIR/nuclio" \
    --file "$SCRIPT_DIR/nuclio/function-gpu.yaml" --platform local

# read the class names stored in the checkpoint
SPEC="$(docker run --rm -v "$WEIGHTS:/w/model.pt:ro" --entrypoint python3 "$IMAGE" -c '
import json
from ultralytics import YOLO
model = YOLO("/w/model.pt")
spec = [{"id": int(i), "name": n, "type": "rectangle"} for i, n in model.names.items()]
if model.task == "pose":
    spec.append({"id": len(spec), "name": "Point", "type": "points"})
print(json.dumps(spec))
' | tail -1)"

cp -r "$SCRIPT_DIR/nuclio/." "$WORK_DIR/"
SPEC="$SPEC" FUNC_NAME="$FUNC_NAME" DISPLAY_NAME="$DISPLAY_NAME" python3 - "$WORK_DIR/function-gpu.yaml" <<'PY'
import json, os, sys
path = sys.argv[1]
text = open(path).read()
spec = "\n".join("      " + line for line in json.dumps(json.loads(os.environ["SPEC"]), indent=2).splitlines())
text = text.replace("  name: ultralytics-yolo\n", f"  name: {os.environ['FUNC_NAME']}\n", 1)
text = text.replace("    name: Ultralytics YOLO\n", f"    name: {os.environ['DISPLAY_NAME']}\n", 1)
text = text.replace("    spec: |\n      []\n", "    spec: |\n" + spec + "\n", 1)
open(path, "w").write(text)
PY

nuctl deploy "$FUNC_NAME" --project-name cvat --path "$WORK_DIR" \
    --file "$WORK_DIR/function-gpu.yaml" --platform local \
    --volume "$WEIGHTS:/opt/nuclio/weights/model.pt" \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

nuctl get function --platform local
