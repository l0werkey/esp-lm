#!/usr/bin/env bash
# export model weights and flash to ESP32
#
# Usage:
#   ./regenerate.sh                # export + build (ESP32-C3, default)
#   ./regenerate.sh --flash        # export + build + flash ESP32-C3
#   ./regenerate.sh --flash-wroom  # export + build + flash ESP32-WROOM-32
#
# Optional flag (combinable):
#   --fan   enable fan on GPIO3 during inference

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv/bin/python"
V1="model"

ENV="esp32-c3-devkitm-1"
DO_FLASH=0
USE_FAN=0

for arg in "$@"; do
    case "$arg" in
        --flash)       DO_FLASH=1; ENV="esp32-c3-devkitm-1" ;;
        --flash-wroom) DO_FLASH=1; ENV="esp32-wroom-32";;
        --fan)         USE_FAN=1;;
        *) echo "Usage: $0 [--flash | --flash-wroom] [--fan]" >&2; exit 1 ;;
    esac
done

export EXTRA_BUILD_FLAGS=""
[[ "$USE_FAN" == "1" ]] && EXTRA_BUILD_FLAGS="-DFAN_PIN=3"

# Pick checkpoint
if [[ -f "$V1/checkpoints/checkpoint.pt" ]]; then
    CHECKPOINT="$V1/checkpoints/checkpoint.pt"
elif [[ -f "$V1/checkpoints/pretrain_checkpoint.pt" ]]; then
    CHECKPOINT="$V1/checkpoints/pretrain_checkpoint.pt"
else
    echo "ERROR: no checkpoint in $V1/checkpoints/" >&2; exit 1
fi

echo "=== export  $CHECKPOINT → $V1/model.c ==="
$VENV - <<EOF
import sys, os
sys.path.insert(0, os.path.abspath('$V1'))
sys.path.insert(0, os.path.abspath('.'))
import torch
from model import LM
from exporter import export

state = torch.load("$CHECKPOINT", map_location="cpu")
m = LM(dim=128, n_layers=6, vocab=2048, mem_dim=512, up_dim=256)
sd = state["model"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
m.load_state_dict(sd)
export(m.cpu(), out_path="$V1/model.c")
print(f"  epoch={state.get('epoch')}  loss={state.get('loss', float('nan')):.4f}")
EOF

echo ""
echo "=== copy  $V1/model.c → $V1/esp-src/src/model.c ==="
cp "$V1/model.c" "$V1/esp-src/src/model.c"

echo ""
echo "=== build  [$ENV] ==="
cd "$V1/esp-src"
pio run -e "$ENV"

if [[ "$DO_FLASH" == "1" ]]; then
    echo ""
    echo "=== flash  [$ENV] ==="
    pio run -e "$ENV" --target upload
    echo "=== monitor (Ctrl+C to stop) ==="
    pio device monitor -e "$ENV"
fi

echo ""
echo "Done.  Connect to WiFi: esp-lm  →  http://192.168.4.1"
