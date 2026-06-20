import argparse
from graphviz import Digraph

DIM      = 128
N_LAYERS = 6
VOCAB    = 2048
MEM_DIM  = 512
UP_DIM   = 256
DROPOUT  = 0.1


def node(g, name, label, color="#313244", shape="box"):
    g.node(name, label, shape=shape, fillcolor=color,
           style="filled", fontname="Helvetica", fontsize="11",
           fontcolor="#cdd6f4", penwidth="0")


def edge(g, a, b, label="", color="#6c7086", style="solid"):
    g.edge(a, b, label=label, color=color, fontcolor="#585b70",
           fontname="Helvetica", fontsize="9", style=style)


def build_graph() -> Digraph:
    g = Digraph("LM", graph_attr={
        "rankdir": "TB",
        "splines": "ortho",
        "nodesep": "0.5",
        "ranksep": "0.7",
        "fontname": "Helvetica",
        "bgcolor": "#1e1e2e",
    })

    # ── Inputs ────────────────────────────────────────────────────────────────
    node(g, "tokens", "Token IDs\n[B, T]",         color="#45475a", shape="parallelogram")
    node(g, "mem_in", f"mem₀  [{MEM_DIM}]  = 0",  color="#45475a", shape="parallelogram")

    # ── Embedding ─────────────────────────────────────────────────────────────
    node(g, "emb",  f"Embedding\n{VOCAB} × {DIM}", color="#1e66f5")
    node(g, "drop", f"Dropout  p={DROPOUT}",        color="#313244")
    edge(g, "tokens", "emb")
    edge(g, "emb",    "drop", f"[B,T,{DIM}]")

    # ── Per-token step cluster ────────────────────────────────────────────────
    with g.subgraph(name="cluster_step") as s:
        s.attr(label=f"Unrolled over T tokens",
               style="dashed", color="#585b70",
               fontcolor="#a6adc8", fontname="Helvetica", fontsize="11",
               bgcolor="#181825")

        s.node("tok_proj",
               f"token_proj\n({DIM}+{MEM_DIM}) → {DIM}",
               shape="box", fillcolor="#40a02b", style="filled",
               fontname="Helvetica", fontsize="11",
               fontcolor="#1e1e2e", penwidth="0")

        edge(g, "drop",   "tok_proj", f"h_t [{DIM}]")
        edge(g, "mem_in", "tok_proj", f"mem [{MEM_DIM}]", color="#fab387", style="dashed")

        prev_h   = "tok_proj"
        prev_mem = "mem_in"

        for i in range(N_LAYERS):
            b = f"b{i}"

            with g.subgraph(name=f"cluster_b{i}") as sb:
                sb.attr(label=f"LMBlock {i}",
                        style="rounded,filled", bgcolor="#24273a",
                        color="#313244", fontcolor="#cba6f7",
                        fontname="Helvetica", fontsize="10")

                for nid, lbl, col, fc in [
                    (f"{b}_norm",  "RMSNorm",                              "#89b4fa", "#1e1e2e"),
                    (f"{b}_up",    f"up_proj\n{DIM}→{UP_DIM}",            "#313244", "#cdd6f4"),
                    (f"{b}_gate",  f"gate_proj\n({DIM}+{MEM_DIM})→{UP_DIM}", "#313244", "#cdd6f4"),
                    (f"{b}_act",   "SiLU(gate) × up\nDropout",            "#f38ba8", "#1e1e2e"),
                    (f"{b}_down",  f"down_proj\n{UP_DIM}→{DIM}",          "#313244", "#cdd6f4"),
                    (f"{b}_mproj", f"mem_proj\n{UP_DIM}→{MEM_DIM}",       "#fab387", "#1e1e2e"),
                    (f"{b}_hout",  "+ residual",                           "#a6e3a1", "#1e1e2e"),
                    (f"{b}_mout",  "mem += mem_proj(latent)",              "#fab387", "#1e1e2e"),
                ]:
                    sb.node(nid, lbl, shape="box", fillcolor=col, style="filled",
                            fontname="Helvetica", fontsize="10",
                            fontcolor=fc, penwidth="0")

            c = "#6c7086"
            edge(g, prev_h,       f"{b}_norm",  color=c)
            edge(g, f"{b}_norm",  f"{b}_up",    color=c)
            edge(g, f"{b}_norm",  f"{b}_gate",  color=c)
            edge(g, prev_mem,     f"{b}_gate",  color="#fab387", style="dashed")
            edge(g, f"{b}_up",    f"{b}_act",   color=c)
            edge(g, f"{b}_gate",  f"{b}_act",   color=c)
            edge(g, f"{b}_act",   f"{b}_down",  color=c)
            edge(g, f"{b}_act",   f"{b}_mproj", color="#fab387")
            edge(g, f"{b}_down",  f"{b}_hout",  color=c)
            edge(g, prev_h,       f"{b}_hout",  label="residual", color="#a6e3a1", style="dashed")
            edge(g, prev_mem,     f"{b}_mout",  color="#fab387", style="dashed")
            edge(g, f"{b}_mproj", f"{b}_mout",  color="#fab387")

            prev_h   = f"{b}_hout"
            prev_mem = f"{b}_mout"

        # ── Memory gating & decay ─────────────────────────────────────────────
        s.node("mem_norm",  f"mem_norm  RMSNorm({MEM_DIM})",
               shape="box", fillcolor="#89b4fa", style="filled",
               fontname="Helvetica", fontsize="10", fontcolor="#1e1e2e", penwidth="0")
        s.node("mem_gate_n", f"mem_gate\n(2×{MEM_DIM})→{MEM_DIM}  SiLU",
               shape="box", fillcolor="#89b4fa", style="filled",
               fontname="Helvetica", fontsize="10", fontcolor="#1e1e2e", penwidth="0")
        s.node("mem_decay_n", f"mem = σ(decay)·mem\n     + (1-σ)·gate·n_mem",
               shape="box", fillcolor="#cba6f7", style="filled",
               fontname="Helvetica", fontsize="10", fontcolor="#1e1e2e", penwidth="0")

        edge(g, prev_mem,       "mem_norm",   color="#fab387")
        edge(g, "mem_norm",     "mem_gate_n", color="#fab387")
        edge(g, "mem_gate_n",   "mem_decay_n",color="#fab387")

    # ── Output ────────────────────────────────────────────────────────────────
    node(g, "mem_out",  f"mem  [{MEM_DIM}]",    color="#45475a", shape="parallelogram")
    node(g, "out_norm", f"out_norm  RMSNorm({DIM})", color="#89b4fa")
    node(g, "head",     f"pred_head\n{DIM}→{VOCAB}  (tied to emb)", color="#1e66f5")
    node(g, "logits",   f"logits  [B, T, {VOCAB}]", color="#45475a", shape="parallelogram")

    edge(g, "mem_decay_n", "mem_out",  color="#fab387")
    edge(g, prev_h,        "out_norm")
    edge(g, "out_norm",    "head")
    edge(g, "head",        "logits")

    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_arch", help="Output basename")
    ap.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"])
    args = ap.parse_args()

    g = build_graph()
    path = g.render(args.out, format=args.fmt, cleanup=True)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
