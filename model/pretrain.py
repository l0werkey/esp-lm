import math
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch

sys.setrecursionlimit(32768)
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import load, encode
from model import LM

_HERE        = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_MB  = 200
VAL_MB       = 10
BLOCK_SIZE   = 256
CHUNK_SIZE   = BLOCK_SIZE + 1
TBPTT_K      = 64
BATCH_SIZE   = 1024
EPOCHS       = 500
LR           = 1e-4
MIN_LR       = 1e-5
WARMUP_STEPS = 400
CHECKPOINT   = os.path.join(_HERE, "checkpoints", "pretrain_checkpoint.pt")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = load(os.path.join(_HERE, '..', 'tokenizer', 'tokenizer.json'))
torch.set_float32_matmul_precision('high')


def preprocess(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("'", "")
    text = re.sub(r' +', ' ', text)
    return text


TARGET_BYTES = PRETRAIN_MB * 1024 * 1024

def stream_tokens(target_bytes: int, desc: str, ds_iter) -> list[int]:
    tokens = []
    seen   = 0
    with tqdm(total=target_bytes, desc=desc, unit="B", unit_scale=True) as pbar:
        for example in ds_iter:
            text   = preprocess(example["text"])
            tokens.extend(encode(tok, "[BOS]" + text))
            nb     = len(text.encode())
            pbar.update(nb)
            seen  += nb
            if seen >= target_bytes:
                break
    return tokens

print(f"Streaming FineWeb  (train {PRETRAIN_MB} MB + val {VAL_MB} MB) …")
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)

train_tokens = stream_tokens(TARGET_BYTES,             "train", ds_iter)
val_tokens   = stream_tokens(VAL_MB * 1024 * 1024,     "val  ", ds_iter)

print(f"Train: {len(train_tokens):,} tokens  |  Val: {len(val_tokens):,} tokens")


def pack(tokens):
    n = len(tokens) // CHUNK_SIZE
    return [tokens[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(n)]

train_chunks = pack(train_tokens)
val_chunks   = pack(val_tokens)
n = len(train_chunks)
print(f"Train: {n:,} chunks  |  Val: {len(val_chunks):,} chunks")


class PackedDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


_pin = DEVICE.type == "cuda"
loader     = DataLoader(PackedDataset(train_chunks), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=_pin)
val_loader = DataLoader(PackedDataset(val_chunks),   batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=_pin)

model     = LM(dim=128, n_layers=6, vocab=2048, mem_dim=512, up_dim=256, dropout=0.05).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

N_CHUNKS    = BLOCK_SIZE // TBPTT_K   # optimizer steps per TBPTT batch
TOTAL_STEPS = len(loader) * N_CHUNKS * EPOCHS

def lr_lambda(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR / LR + (1.0 - MIN_LR / LR) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

start_epoch = 0
global_step = 0

if os.path.exists(CHECKPOINT):
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    sd = state["model"]
    if any(k.startswith("_orig_mod.") for k in sd):  # torch.compile artifact
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(state["optimizer"])
    if "lr_lambdas" in state["scheduler"]:  # only load if same scheduler type
        scheduler.load_state_dict(state["scheduler"])
    start_epoch = state["epoch"] + 1
    global_step = state.get("global_step", 0)
    print(f"Resumed from epoch {start_epoch}  (step {global_step})")

if DEVICE.type == "cuda":
    _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    _amp_dtype = torch.float32
_use_amp = _amp_dtype != torch.float32
_scaler  = torch.amp.GradScaler("cuda", enabled=(_amp_dtype == torch.float16))

print(f"Using {DEVICE}  |  {model.num_params:,} params  |  amp={_amp_dtype}")
print(f"{n:,} chunks  |  {len(loader)} batches/epoch  |  {N_CHUNKS} TBPTT chunks/batch  |  {EPOCHS} epochs")


epoch_bar = tqdm(range(start_epoch, EPOCHS), desc="Pretrain", unit="epoch",
                 initial=start_epoch, total=EPOCHS)

for epoch in epoch_bar:
    model.train()
    total_loss = 0.0
    step_bar   = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

    for x, y in step_bar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        mem        = None
        batch_loss = 0.0

        for k in range(N_CHUNKS):
            x_c = x[:, k * TBPTT_K : (k + 1) * TBPTT_K]
            y_c = y[:, k * TBPTT_K : (k + 1) * TBPTT_K]

            with torch.autocast(device_type=DEVICE.type, dtype=_amp_dtype, enabled=_use_amp):
                logits_c, mem = model(x_c, mem)
                loss = criterion(logits_c.reshape(-1, logits_c.size(-1)), y_c.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            _scaler.scale(loss).backward()
            _scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _scaler.step(optimizer)
            _scaler.update()
            scheduler.step()

            mem = mem.detach()
            batch_loss  += loss.item()
            global_step += 1

        batch_loss /= N_CHUNKS
        total_loss += batch_loss
        step_bar.set_postfix(
            loss=f"{batch_loss:.4f}",
            ppl=f"{math.exp(min(batch_loss, 20)):.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    avg = total_loss / len(loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=_amp_dtype, enabled=_use_amp):
                logits, _ = model(x)
            val_loss += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    val_avg = val_loss / len(val_loader)
    tqdm.write(f"[val] epoch={epoch}  loss={val_avg:.4f}  ppl={math.exp(min(val_avg, 20)):.2f}")

    epoch_bar.set_postfix(avg_loss=f"{avg:.4f}", avg_ppl=f"{math.exp(avg):.2f}",
                          val_ppl=f"{math.exp(min(val_avg, 20)):.2f}")

    torch.save({
        "epoch":       epoch,
        "global_step": global_step,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "loss":        avg,
    }, CHECKPOINT)

print(f"Done. Saved → {CHECKPOINT}")
