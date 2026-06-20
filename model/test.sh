#!/usr/bin/env bash
# run from anywhere - paths are relative to this script
set -euo pipefail
cd "$(dirname "$0")"

VENV="../.venv/bin/python"
MODEL_C="model.c"
TEST_C="test.c"
CHAT_C="chat.c"
BIN_TEST="test"
BIN_CHAT="chat"

# Pick checkpoint: fine-tuned > pretrain
if [[ -f checkpoints/checkpoint.pt ]]; then
    CHECKPOINT="checkpoints/checkpoint.pt"
elif [[ -f checkpoints/pretrain_checkpoint.pt ]]; then
    CHECKPOINT="checkpoints/pretrain_checkpoint.pt"
else
    echo "ERROR: no checkpoint in checkpoints/" >&2; exit 1
fi
echo "=== export $CHECKPOINT → $MODEL_C ==="

$VENV - <<EOF
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("$0")), '..'))
import torch
from model import LM
from exporter import export

state = torch.load("$CHECKPOINT", map_location="cpu")
m = LM(dim=128, n_layers=6, vocab=2048, mem_dim=512, up_dim=256)
sd = state["model"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
m.load_state_dict(sd)
export(m.cpu())
print(f"epoch={state.get('epoch')}  loss={state.get('loss', float('nan')):.4f}")
EOF

echo ""
echo "=== compile ==="
gcc -O2 -std=c11 "$TEST_C" -o "$BIN_TEST" -lm
echo "built → $BIN_TEST"
gcc -O2 -std=c11 "$CHAT_C" -o "$BIN_CHAT" -lm
echo "built → $BIN_CHAT"

echo ""
echo "=== generation benchmark ==="
./"$BIN_TEST"

echo ""
echo "=== to chat ==="
echo "  ./$BIN_CHAT            # default temp 0.8"
echo "  ./$BIN_CHAT 1.0        # custom temp"
