import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from model import LM
from exporter import export
from tokenizer import load, encode, decode, token_id
import torch

model = LM(dim=128, n_layers=6, vocab=2048, mem_dim=512, up_dim=256)

dummy_data = torch.randint(0, 2048, (1, 32))
pred, mem = model(dummy_data)

print(pred)
print(mem)
print(model.num_params)

# export(model)

tok = load(os.path.join(_HERE, '..', 'tokenizer', 'tokenizer.json'))
ids = encode(tok, "want to grab food later?[EOS]")
print(ids)
text = decode(tok, ids)
print(text)

eos_id = token_id(tok, "[EOS]")