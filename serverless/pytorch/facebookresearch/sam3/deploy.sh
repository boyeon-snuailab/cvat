#!/bin/bash
# Deploys the SAM 3 interactor (GPU, holds the model) and the SAM 3 text detector
# (CPU, forwards to the interactor). The checkpoint is mounted from the host;
# download it from https://huggingface.co/facebook/sam3 (gated) first.
# Usage: ./deploy.sh [path to the directory containing sam3.pt]

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WEIGHTS_DIR="$(realpath "${1:-$HOME/models/sam3}")"

if [ ! -f "$WEIGHTS_DIR/sam3.pt" ]; then
    echo "sam3.pt was not found in $WEIGHTS_DIR" >&2
    exit 1
fi

nuctl create project cvat --platform local || true

nuctl deploy --project-name cvat --path "$SCRIPT_DIR/interactor/nuclio" \
    --file "$SCRIPT_DIR/interactor/nuclio/function-gpu.yaml" --platform local \
    --volume "$WEIGHTS_DIR:/opt/nuclio/weights" \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

nuctl deploy --project-name cvat --path "$SCRIPT_DIR/detector/nuclio" \
    --file "$SCRIPT_DIR/detector/nuclio/function.yaml" --platform local \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

nuctl get function --platform local
