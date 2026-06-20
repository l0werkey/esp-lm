import argparse
import json
import re
import string
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFD, Sequence as NormSeq, StripAccents
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

_METASPACE  = "▁"
_SPECIAL_RE = re.compile(r'(\[[A-Z]+\])')

DEFAULT_OUT        = "tokenizer.json"
DEFAULT_VOCAB_SIZE = 2048

# IDs must be stable (0–6): C code and model weights hard-code this order
BASE_SPECIAL = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]", "[A]", "[B]"]

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
SPK_A     = "[A]"
SPK_B     = "[B]"

FINEWEB_MB = 100


def _iter_utterances():
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("frankdarkluo/DailyDialog", split=split)
        except Exception:
            continue
        for row in ds:
            for field in ("context", "response"):
                text = row.get(field, "").strip()
                if text:
                    text = re.sub(r"\s+([.,!?;:'\")\]])", r"\1", text)
                    text = text.replace("'", "")
                    text = text.encode("ascii", errors="ignore").decode("ascii")
                    yield text

    print(f"\nStreaming FineWeb for tokenizer ({FINEWEB_MB} MB) …")
    fw = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    target = FINEWEB_MB * 1024 * 1024
    seen   = 0
    for example in fw:
        text = example["text"].lower().strip().replace("'", "")
        text = text.encode("ascii", errors="ignore").decode("ascii")
        seen += len(text.encode())
        yield text
        if seen >= target:
            break


def train(
    vocab_size:    int       = DEFAULT_VOCAB_SIZE,
    extra_tokens:  list[str] = (),
    out_path:      Path      = Path(DEFAULT_OUT),
) -> Tokenizer:
    special_tokens = BASE_SPECIAL + [t for t in extra_tokens if t not in BASE_SPECIAL]

    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.normalizer    = NormSeq([NFD(), Lowercase(), StripAccents()])
    # Metaspace marks word-starts with ▁ - "me" and " me" both tokenize as ▁me
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder       = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    _alphabet = list("▁" + string.ascii_lowercase + string.digits + string.punctuation)
    trainer = BpeTrainer(
        vocab_size       = vocab_size,
        special_tokens   = special_tokens,
        initial_alphabet = _alphabet,
        min_frequency    = 2,
        show_progress    = True,
    )

    print("Loading DailyDialog …")
    utterances = list(_iter_utterances())
    print(f"  {len(utterances):,} utterances across train / val / test")

    print(f"\nTraining BPE  vocab={vocab_size}  special={special_tokens}")
    tok.train_from_iterator(utterances, trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    _print_stats(tok, special_tokens)
    print(f"\nSaved → {out_path}")
    return tok


def load(path: str | Path = DEFAULT_OUT) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def _normalize(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    text = text.replace("'", "")
    return text


def _maxmatch_word(word_str: str, vocab: dict) -> list[int]:
    """Minimum-token DP segmentation - fewest tokens wins over greedy BPE."""
    n = len(word_str)
    INF = n + 1
    cost   = [INF] * (n + 1)
    prev   = [-1]  * (n + 1)
    tok_at = [0]   * (n + 1)
    cost[0] = 0

    for i in range(n):
        if cost[i] == INF:
            continue
        for j in range(i + 1, n + 1):
            s = word_str[i:j]
            if s not in vocab:
                continue
            c = cost[i] + 1
            if c < cost[j]:
                cost[j]   = c
                prev[j]   = i
                tok_at[j] = vocab[s]

    if cost[n] == INF:
        return [vocab.get(c, 0) for c in word_str]

    ids, p = [], n
    while p > 0:
        ids.append(tok_at[p])
        p = prev[p]
    ids.reverse()
    return ids


def encode(tok: Tokenizer, text: str) -> list[int]:
    """encode with special-token passthrough then MaxMatch DP on normal words

    special tokens (e.g. [BOS]) are matched before normalization
    normalization: ASCII-only, lowercase, apostrophes stripped
    """
    vocab = {k: v for k, v in tok.get_vocab().items() if v >= len(BASE_SPECIAL)}
    ids   = []
    for part in _SPECIAL_RE.split(text):
        if _SPECIAL_RE.fullmatch(part):
            tid = tok.token_to_id(part)
            if tid is not None:
                ids.append(tid)
        else:
            for word in _normalize(part).split():
                ids.extend(_maxmatch_word(_METASPACE + word, vocab))
    return ids


def decode(tok: Tokenizer, ids: list[int], skip_special: bool = True) -> str:
    special = {tok.token_to_id(t) for t in BASE_SPECIAL
               if tok.token_to_id(t) is not None}
    if skip_special:
        ids = [i for i in ids if i not in special]
    return tok.decode(ids).lstrip(" ")


def token_id(tok: Tokenizer, token: str) -> int | None:
    return tok.token_to_id(token)


def _print_stats(tok: Tokenizer, special_tokens: list[str]) -> None:
    vocab = tok.get_vocab()
    print(f"\n── Tokenizer stats ───────────────────────────────")
    print(f"  vocab size : {len(vocab)}")
    print(f"  special    : {[(t, vocab[t]) for t in special_tokens if t in vocab]}")

    samples = [
        "hey how are you doing today",
        "i had a rough day at work",
        "want to grab some food later",
        "that sounds good to me",
        "my battery just died mid call",
        "i can not believe it is already friday",
    ]
    print(f"\n── Sample encodings ──────────────────────────────")
    for s in samples:
        enc = tok.encode(s)
        print(f"  {s!r}")
        print(f"    {enc.tokens}  →  {len(enc.ids)} tokens")

    test_words = ["the", "and", "you", "is", "it", "that", "have",
                  "was", "for", "on", "are", "with", "he", "as",
                  "what", "just", "good", "okay", "yeah", "know"]
    single = [w for w in test_words if len(tok.encode(w).ids) == 1]
    print(f"\n── Single-token common words ({len(single)}/{len(test_words)}) ──")
    print(f"  {single}")


def sanity_check(path: str | Path = DEFAULT_OUT) -> None:
    tok = load(path)
    print(f"Loaded tokenizer from {path}")
    _print_stats(tok, BASE_SPECIAL)

    texts = [
        "hey what are you up to",
        "not much just chilling at home",
        "want to grab food later",
    ]
    print("\n── Round-trip ────────────────────────────────────")
    for t in texts:
        ids  = encode(tok, t)
        back = decode(tok, ids)
        ok   = "✓" if back.strip() == t else "✗"
        print(f"  {ok}  {t!r}  →  {ids}  →  {back!r}")


def main():
    p = argparse.ArgumentParser(
        description="Train a BPE tokenizer on DailyDialog + FineWeb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vocab-size",   type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--out",          default=DEFAULT_OUT,
                   help="Output path for the tokenizer JSON")
    p.add_argument("--extra-tokens", nargs="*", default=[],
                   help='Custom special tokens e.g. --extra-tokens "[USR]" "[SYS]"')
    p.add_argument("--test",         action="store_true",
                   help="Load an existing tokenizer and run a sanity check")
    args = p.parse_args()

    if args.test:
        sanity_check(args.out)
    else:
        train(
            vocab_size   = args.vocab_size,
            extra_tokens = args.extra_tokens,
            out_path     = Path(args.out),
        )


if __name__ == "__main__":
    main()
