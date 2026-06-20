import re
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import load, encode
from model import LM
from exporter import export

_HERE         = os.path.dirname(os.path.abspath(__file__))
tok = load(os.path.join(_HERE, '..', 'tokenizer', 'tokenizer.json'))
torch.set_float32_matmul_precision('high')

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE     = 128
CHUNK_SIZE     = BLOCK_SIZE + 1
BATCH_SIZE     = 512
EPOCHS         = 50
WARMUP_EPOCHS  = 10   # epochs 0–9: core dialog only, before mixing SODA/UltraChat
COOLDOWN_EPOCH = 49   # last epoch: core dialog only
LR             = 2e-5
MIN_LR         = 3e-6
CHECKPOINT     = os.path.join(_HERE, "checkpoints", "checkpoint.pt")
FINEWEB_RATIO  = 100
FINEWEB_MB     = 20

SEP_ID = tok.token_to_id("[SEP]")


def preprocess(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("'", "")
    text = re.sub(r' +', ' ', text)
    return text


def fix_dialog_text(text: str) -> str:
    text = preprocess(text)
    text = re.sub(r'\s+([,\.!?;:\)])', r'\1', text)
    text = re.sub(r'([\(])\s+', r'\1', text)
    text = re.sub(r'([\.!?])([a-z])', r'\1 \2', text)
    return text


def format_dialog(dialog):
    speakers = ["[A]", "[B]"]
    turns = [speakers[i % 2] + t for i, t in enumerate(dialog)]
    return "[BOS]" + "[SEP]".join(turns) + "[SEP]"


def parse_synthetic_turns(text: str) -> list:
    turns = re.split(r'User \d+:\s*', text)
    return [t.strip() for t in turns if t.strip()]


def _collect_tokens(ds, desc: str) -> list:
    tokens = []
    for example in tqdm(ds, desc=desc):
        tokens.extend(example["dialog"])
    return tokens


def _pack(tokens: list) -> list:
    n = len(tokens) // CHUNK_SIZE
    return [tokens[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(n)]


def load_dialog(split: str) -> list:
    dd = load_dataset("OpenRL/daily_dialog", split=split)
    dd = dd.map(lambda ex: {"dialog": [fix_dialog_text(t) for t in ex["dialog"]]})
    dd = dd.map(lambda ex: {"dialog": format_dialog(ex["dialog"])})
    dd = dd.map(lambda ex: {"dialog": encode(tok, ex["dialog"])})

    sp = load_dataset("google/Synthetic-Persona-Chat", split=split)
    sp = sp.map(lambda ex: {"dialog": parse_synthetic_turns(ex["Best Generated Conversation"])})
    sp = sp.map(lambda ex: {"dialog": [fix_dialog_text(t) for t in ex["dialog"]]})
    sp = sp.map(lambda ex: {"dialog": format_dialog(ex["dialog"])})
    sp = sp.map(lambda ex: {"dialog": encode(tok, ex["dialog"])})

    tokens  = _collect_tokens(dd, f"daily_dialog/{split}")
    tokens += _collect_tokens(sp, f"synthetic-persona/{split}")
    return _pack(tokens)


def load_soda(split: str, max_n: int = 80_000) -> list:
    ds = load_dataset("allenai/soda", split=split)
    if max_n and len(ds) > max_n:
        ds = ds.select(range(max_n))
    tokens = []
    spk = ["[A]", "[B]"]
    for row in tqdm(ds, desc=f"soda/{split}"):
        turns = row["dialogue"]
        if len(turns) < 2:
            continue
        formatted = "[BOS]" + "[SEP]".join(
            spk[i % 2] + fix_dialog_text(t) for i, t in enumerate(turns)
        ) + "[SEP]"
        tokens.extend(encode(tok, formatted))
    return _pack(tokens)


def load_ultrachat(split: str, max_n: int = 25_000, max_turns: int = 6) -> list:
    hf_split = "train_sft" if split == "train" else "test_sft"
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=hf_split)
    if max_n and len(ds) > max_n:
        ds = ds.select(range(max_n))
    tokens = []
    role_to_spk = {"user": "[A]", "assistant": "[B]"}
    for row in tqdm(ds, desc=f"ultrachat/{split}"):
        msgs = row["messages"][:max_turns]
        if len(msgs) < 2:
            continue
        turns = [
            role_to_spk.get(m["role"], "[A]") + fix_dialog_text(m["content"])
            for m in msgs if m.get("content", "").strip()
        ]
        if len(turns) < 2:
            continue
        tokens.extend(encode(tok, "[BOS]" + "[SEP]".join(turns) + "[SEP]"))
    return _pack(tokens)


def load_fineweb(mb: int) -> list:
    fw = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    target = mb * 1024 * 1024
    seen, tokens = 0, []
    for ex in tqdm(fw, desc=f"FineWeb mix ({mb} MB)"):
        text = preprocess(ex["text"])
        tokens.extend(encode(tok, "[BOS]" + text))
        seen += len(text.encode())
        if seen >= target:
            break
    return _pack(tokens)


class PackedDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long, device=DEVICE)
        return chunk[:-1], chunk[1:]


print("Loading core dialog (DailyDialog + Persona) …")
core_train = load_dialog("train")
core_val   = load_dialog("validation")
print(f"  core train: {len(core_train):,} chunks  |  val: {len(core_val):,} chunks")

print("Loading SODA …")
soda_train = load_soda("train", max_n=80_000)
print(f"  soda train: {len(soda_train):,} chunks")

print("Loading UltraChat …")
uc_train = load_ultrachat("train", max_n=25_000)
print(f"  ultrachat train: {len(uc_train):,} chunks")

full_train = core_train + soda_train + uc_train
print(f"  FULL train: {len(full_train):,} chunks  |  val (core only): {len(core_val):,} chunks")

print("Loading FineWeb for mixing …")
fw_chunks = load_fineweb(FINEWEB_MB)
print(f"  fineweb: {len(fw_chunks):,} chunks")

loader_core = DataLoader(PackedDataset(core_train), batch_size=BATCH_SIZE, shuffle=True)
loader_full = DataLoader(PackedDataset(full_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader  = DataLoader(PackedDataset(core_val),   batch_size=BATCH_SIZE, shuffle=False)
fw_loader   = DataLoader(PackedDataset(fw_chunks),  batch_size=BATCH_SIZE, shuffle=True)


def epoch_loader(epoch: int):
    if epoch < WARMUP_EPOCHS or epoch == COOLDOWN_EPOCH:
        return loader_core
    return loader_full


model     = LM(dim=128, n_layers=6, vocab=2048,
               mem_dim=512, up_dim=256, dropout=0.1).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
sep_criterion = torch.nn.CrossEntropyLoss()

if DEVICE.type == "cuda":
    _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    _amp_dtype = torch.float32
_use_amp = _amp_dtype != torch.float32
_scaler  = torch.amp.GradScaler("cuda", enabled=(_amp_dtype == torch.float16))

start_epoch = 0
global_step = 0

if os.path.exists(CHECKPOINT):
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    sd = state["model"]
    if any(k.startswith("_orig_mod.") for k in sd):  # torch.compile artifact
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(state["optimizer"])
    for pg in optimizer.param_groups:  # reset LR - don't restore scheduler (stale T_max)
        pg['lr'] = LR
    start_epoch = state["epoch"] + 1
    global_step = state.get("global_step", 0)
    print(f"Resumed from epoch {start_epoch}  LR reset → {LR:.1e}")
elif os.path.exists(os.path.join(_HERE, "checkpoints", "pretrain_checkpoint.pt")):
    state = torch.load(os.path.join(_HERE, "checkpoints", "pretrain_checkpoint.pt"), map_location=DEVICE)
    sd = state["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print("Initialized from checkpoints/pretrain_checkpoint.pt (fresh optimizer + scheduler)")


def _remaining_steps(from_epoch: int) -> int:
    total = 0
    for e in range(from_epoch, EPOCHS):
        total += len(loader_core) if (e < WARMUP_EPOCHS or e == COOLDOWN_EPOCH) else len(loader_full)
    return max(total, 1)

remaining_steps = _remaining_steps(start_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=remaining_steps, eta_min=MIN_LR,
)

print(f"Using {DEVICE}  |  {model.num_params:,} params  |  amp={_amp_dtype}")
print(f"core: {len(loader_core)} batches/epoch  |  full: {len(loader_full)} batches/epoch  |  FineWeb ratio 1:{FINEWEB_RATIO}")


def train_step(x, y, boost_sep: bool = False) -> float:
    model.train()
    with torch.autocast(device_type=DEVICE.type, dtype=_amp_dtype, enabled=_use_amp):
        logits, _ = model(x)
        y_flat  = y.view(-1)
        lg_flat = logits.view(-1, logits.size(-1))
        loss = criterion(lg_flat, y_flat)
        if boost_sep:
            sep_mask = (y_flat == SEP_ID)
            if sep_mask.any():
                loss = loss + sep_criterion(lg_flat[sep_mask], y_flat[sep_mask])

    optimizer.zero_grad(set_to_none=True)
    _scaler.scale(loss).backward()
    _scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    _scaler.step(optimizer)
    _scaler.update()
    scheduler.step()
    return loss.item()


fw_iter = iter(fw_loader)

epoch_bar = tqdm(range(start_epoch, EPOCHS), desc="Finetune", unit="epoch",
                 initial=start_epoch, total=EPOCHS)

for epoch in epoch_bar:
    total_loss = 0.0
    cur_loader = epoch_loader(epoch)
    phase      = "warmup" if epoch < WARMUP_EPOCHS else ("cooldown" if epoch == COOLDOWN_EPOCH else "full")
    step_bar   = tqdm(cur_loader, desc=f"Epoch {epoch} [{phase}]", unit="batch", leave=False)

    for batch_idx, (x, y) in enumerate(step_bar):
        if batch_idx % FINEWEB_RATIO == 0:  # mix FineWeb to prevent catastrophic forgetting
            try:
                x_fw, y_fw = next(fw_iter)
            except StopIteration:
                fw_iter = iter(fw_loader)
                x_fw, y_fw = next(fw_iter)
            train_step(x_fw, y_fw, boost_sep=False)

        loss = train_step(x, y, boost_sep=True)
        total_loss  += loss
        global_step += 1

        step_bar.set_postfix(
            loss=f"{loss:.4f}",
            ppl=f"{math.exp(min(loss, 20)):.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    avg = total_loss / len(cur_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            with torch.autocast(device_type=DEVICE.type, dtype=_amp_dtype, enabled=_use_amp):
                logits, _ = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    val_avg = val_loss / len(val_loader)
    tqdm.write(f"[val] epoch={epoch+1:3d}  loss={val_avg:.4f}  ppl={math.exp(min(val_avg, 20)):.2f}")

    epoch_bar.set_postfix(avg_loss=f"{avg:.4f}", avg_ppl=f"{math.exp(avg):.2f}")

    torch.save({
        "epoch":       epoch,
        "global_step": global_step,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "loss":        avg,
    }, CHECKPOINT)

export(model.cpu())
