import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from tokenizer import load, encode, decode
from model import LM

_HERE        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT   = os.path.join(_HERE, "checkpoints", "checkpoint.pt")
TOKENIZER    = os.path.join(_HERE, "..", "tokenizer", "tokenizer.json")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_KWARGS = dict(dim=128, n_layers=6, vocab=2048, mem_dim=512, up_dim=256)


def load_model() -> LM:
    model = LM(**MODEL_KWARGS).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def generate(model: LM, tok, prompt_ids: list[int],
             max_new: int = 100, temperature: float = 0.8,
             top_k: int = 40, min_p: float = 0.05,
             repetition_penalty: float = 1.3):
    sep_id = tok.token_to_id("[SEP]")

    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logits, mem = model(x)

    next_logits = logits[0, -1, :]
    generated: list[int] = []
    t0 = time.perf_counter()

    for _ in range(max_new):
        scaled = next_logits.clone()

        if repetition_penalty != 1.0 and generated:
            for tid in set(generated):
                scaled[tid] = scaled[tid] / repetition_penalty if scaled[tid] > 0 \
                              else scaled[tid] * repetition_penalty

        scaled = scaled / temperature

        if top_k > 0:
            top_vals, _ = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < top_vals[-1], float("-inf"))

        if min_p > 0:
            probs_raw = torch.softmax(scaled, dim=-1)
            scaled = scaled.masked_fill(probs_raw < min_p * probs_raw.max(), float("-inf"))

        probs = torch.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, 1).item()

        if token == sep_id:
            break

        generated.append(token)

        x_next = torch.tensor([[token]], dtype=torch.long, device=DEVICE)
        logits_next, mem = model(x_next, mem)
        next_logits = logits_next[0, -1, :]

    elapsed = time.perf_counter() - t0
    n_tok   = len(generated)
    return tok.decode(generated).strip(), elapsed, n_tok


def chat(auto: bool = False) -> None:
    tok   = load(TOKENIZER)
    model = load_model()
    print(f"Loaded on {DEVICE}. Type 'quit' to exit.\n")

    history: list[str] = []

    if auto:
        history.append("hello")
        print("A: hello")

    while True:
        try:
            if auto:
                input()
            else:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                history.append(user_input.lower().replace("'", ""))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        speakers = ["[A]", "[B]"]
        tagged   = [speakers[i % 2] + t for i, t in enumerate(history)]
        next_spk = speakers[len(history) % 2]
        prompt_ids = encode(tok, "[BOS]" + "[SEP]".join(tagged) + "[SEP]" + next_spk)
        response, elapsed, n_tok = generate(model, tok, prompt_ids)

        for tag in ("[A]", "[B]"):
            if response.startswith(tag):
                response = response[len(tag):].strip()

        label = next_spk if auto else "Bot"
        print(f"{label}: {response}")
        print(f"    [{n_tok} tokens · {elapsed:.2f}s · {n_tok/elapsed:.1f} tok/s]")
        history.append(response)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="bot talks to itself")
    args = ap.parse_args()
    chat(auto=args.auto)
