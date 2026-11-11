from __future__ import annotations

import argparse
from typing import Dict, Tuple


def norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


# Hyperparameters transcribed from the official PolyFormer shell scripts included
# in the uploaded GitHub source. node_classification.sh contains Table 5 configs;
# node_classification_large_graph.sh contains Table 6 configs.
OFFICIAL: Dict[Tuple[str, str], Dict[str, object]] = {
    ("citeseer", "mono"): dict(lr=0.0005, weight_decay=0.001, attn_lr=0.001, attn_wd=1e-5, hidden=256, dropout=0.9, dprate=0.6, n_head=4, d_ffn=128, q=2.0, multi=2.0, K=6, nlayer=1, epochs=2000, patience=250),
    ("coauthor_cs", "mono"): dict(lr=0.001, weight_decay=1e-7, attn_lr=0.005, attn_wd=0.0001, hidden=128, dropout=0.0, dprate=0.8, n_head=8, d_ffn=128, q=1.0, multi=1.0, K=2, nlayer=1, epochs=2000, patience=250),
    ("cs", "mono"): dict(lr=0.001, weight_decay=1e-7, attn_lr=0.005, attn_wd=0.0001, hidden=128, dropout=0.0, dprate=0.8, n_head=8, d_ffn=128, q=1.0, multi=1.0, K=2, nlayer=1, epochs=2000, patience=250),
    ("pubmed", "mono"): dict(lr=0.005, weight_decay=0.001, attn_lr=0.0005, attn_wd=1e-8, hidden=256, dropout=0.5, dprate=0.8, n_head=8, d_ffn=32, q=1.6, multi=2.0, K=2, nlayer=2, epochs=2000, patience=250),
    ("coauthor_physics", "mono"): dict(lr=0.001, weight_decay=1e-5, attn_lr=0.0001, attn_wd=0.001, hidden=128, dropout=0.9, dprate=0.8, n_head=2, d_ffn=256, q=1.2, multi=0.5, K=4, nlayer=1, epochs=2000, patience=250),
    ("physics", "mono"): dict(lr=0.001, weight_decay=1e-5, attn_lr=0.0001, attn_wd=0.001, hidden=128, dropout=0.9, dprate=0.8, n_head=2, d_ffn=256, q=1.2, multi=0.5, K=4, nlayer=1, epochs=2000, patience=250),
    ("chameleon_filtered", "mono"): dict(lr=0.01, weight_decay=1e-6, attn_lr=5e-5, attn_wd=1e-5, hidden=256, dropout=0.2, dprate=0.4, n_head=8, d_ffn=32, q=2.0, multi=2.0, K=6, nlayer=4, epochs=2000, patience=250),
    ("squirrel_filtered", "mono"): dict(lr=0.0001, weight_decay=0.0, attn_lr=0.001, attn_wd=1e-7, hidden=256, dropout=0.3, dprate=0.8, n_head=4, d_ffn=128, q=1.4, multi=1.0, K=12, nlayer=2, epochs=2000, patience=250),
    ("minesweeper", "mono"): dict(lr=0.01, weight_decay=1e-5, attn_lr=0.0005, attn_wd=0.0, hidden=16, dropout=0.2, dprate=0.3, n_head=8, d_ffn=32, q=1.6, multi=2.0, K=10, nlayer=4, epochs=2000, patience=250, metric="rocauc", force_binary=1),
    ("tolokers", "mono"): dict(lr=0.0005, weight_decay=1e-8, attn_lr=0.0001, attn_wd=0.0, hidden=64, dropout=0.2, dprate=0.0, n_head=16, d_ffn=128, q=1.0, multi=1.0, K=10, nlayer=1, epochs=2000, patience=250, metric="rocauc", force_binary=1),
    ("roman_empire", "mono"): dict(lr=0.0001, weight_decay=0.001, attn_lr=0.001, attn_wd=0.0001, hidden=256, dropout=0.5, dprate=0.1, n_head=16, d_ffn=64, q=1.0, multi=2.0, K=14, nlayer=3, epochs=2000, patience=250),
    ("questions", "mono"): dict(lr=0.0005, weight_decay=0.001, attn_lr=0.0001, attn_wd=0.0, hidden=128, dropout=0.2, dprate=0.3, n_head=4, d_ffn=256, q=1.0, multi=1.0, K=12, nlayer=1, epochs=2000, patience=250, metric="rocauc", force_binary=1),

    ("citeseer", "cheb"): dict(lr=0.0005, weight_decay=1e-5, attn_lr=0.001, attn_wd=1e-5, hidden=256, dropout=0.7, dprate=0.4, n_head=1, d_ffn=64, q=1.6, multi=0.5, K=2, nlayer=2, epochs=2000, patience=250),
    ("coauthor_cs", "cheb"): dict(lr=0.0005, weight_decay=0.001, attn_lr=0.01, attn_wd=0.001, hidden=64, dropout=0.8, dprate=0.0, n_head=8, d_ffn=32, q=1.0, multi=0.5, K=4, nlayer=1, epochs=2000, patience=250),
    ("cs", "cheb"): dict(lr=0.0005, weight_decay=0.001, attn_lr=0.01, attn_wd=0.001, hidden=64, dropout=0.8, dprate=0.0, n_head=8, d_ffn=32, q=1.0, multi=0.5, K=4, nlayer=1, epochs=2000, patience=250),
    ("pubmed", "cheb"): dict(lr=0.0005, weight_decay=0.001, attn_lr=0.0005, attn_wd=0.0001, hidden=128, dropout=0.3, dprate=0.0, n_head=16, d_ffn=64, q=2.0, multi=2.0, K=6, nlayer=1, epochs=2000, patience=250),
    ("coauthor_physics", "cheb"): dict(lr=0.001, weight_decay=1e-5, attn_lr=0.005, attn_wd=0.001, hidden=128, dropout=0.9, dprate=0.1, n_head=8, d_ffn=64, q=1.6, multi=0.5, K=2, nlayer=2, epochs=2000, patience=250),
    ("physics", "cheb"): dict(lr=0.001, weight_decay=1e-5, attn_lr=0.005, attn_wd=0.001, hidden=128, dropout=0.9, dprate=0.1, n_head=8, d_ffn=64, q=1.6, multi=0.5, K=2, nlayer=2, epochs=2000, patience=250),
    ("chameleon_filtered", "cheb"): dict(lr=0.01, weight_decay=1e-7, attn_lr=0.005, attn_wd=1e-6, hidden=128, dropout=0.6, dprate=0.9, n_head=16, d_ffn=256, q=1.4, multi=0.5, K=2, nlayer=1, epochs=2000, patience=30),
    ("squirrel_filtered", "cheb"): dict(lr=0.0005, weight_decay=1e-7, attn_lr=0.0005, attn_wd=0.001, hidden=256, dropout=0.1, dprate=0.8, n_head=8, d_ffn=256, q=1.4, multi=0.5, K=12, nlayer=1, epochs=2000, patience=250),
    ("minesweeper", "cheb"): dict(lr=0.0005, weight_decay=1e-7, attn_lr=0.0001, attn_wd=0.0001, hidden=128, dropout=0.9, dprate=0.3, n_head=16, d_ffn=128, q=1.6, multi=2.0, K=8, nlayer=2, epochs=2000, patience=250, metric="rocauc", force_binary=1),
    ("tolokers", "cheb"): dict(lr=0.001, weight_decay=0.0001, attn_lr=0.0001, attn_wd=0.0001, hidden=256, dropout=0.0, dprate=0.9, n_head=16, d_ffn=32, q=1.6, multi=2.0, K=8, nlayer=1, epochs=2000, patience=250, metric="rocauc", force_binary=1),
    ("roman_empire", "cheb"): dict(lr=0.005, weight_decay=1e-5, attn_lr=0.0005, attn_wd=1e-8, hidden=256, dropout=0.5, dprate=0.3, n_head=16, d_ffn=64, q=1.4, multi=2.0, K=2, nlayer=3, epochs=2000, patience=250),
    ("questions", "cheb"): dict(lr=0.0001, weight_decay=0.0, attn_lr=0.0005, attn_wd=1e-5, hidden=256, dropout=0.4, dprate=0.7, n_head=16, d_ffn=128, q=1.6, multi=2.0, K=10, nlayer=2, epochs=2000, patience=250, metric="rocauc", force_binary=1),

    ("twitch_gamer", "mono"): dict(lr=5e-5, weight_decay=0.0005, attn_lr=0.0002, attn_wd=5e-5, hidden=128, dropout=0.0, dprate=0.6, n_head=4, d_ffn=512, q=1.0, multi=0.5, K=8, nlayer=2, epochs=2000, patience=250, batch_size=10000),
    ("pokec", "mono"): dict(lr=0.0002, weight_decay=5e-5, attn_lr=0.0002, attn_wd=0.0005, hidden=512, dropout=0.25, dprate=0.0, n_head=1, d_ffn=1024, q=2.0, multi=1.0, K=4, nlayer=1, epochs=2000, patience=250, batch_size=20000),
    ("ogbn_arxiv", "mono"): dict(lr=0.0002, weight_decay=0.0, attn_lr=0.01, attn_wd=0.001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=10000),
    ("ogbn_papers100m", "mono"): dict(lr=0.0002, weight_decay=0.0, attn_lr=5e-5, attn_wd=0.0, hidden=1024, dropout=0.5, dprate=0.8, n_head=1, d_ffn=512, q=1.0, multi=1.0, K=4, nlayer=1, epochs=2000, patience=250, batch_size=50000),
    ("paper100m", "mono"): dict(lr=0.0002, weight_decay=0.0, attn_lr=5e-5, attn_wd=0.0, hidden=1024, dropout=0.5, dprate=0.8, n_head=1, d_ffn=512, q=1.0, multi=1.0, K=4, nlayer=1, epochs=2000, patience=250, batch_size=50000),
}

# Self-set configs for datasets not covered by official PolyFormer scripts.
# They follow the nearest official family: citation/medium homophilic datasets use
# Citation/Coauthor-like configs; large graph benchmarks use the ogbn-arxiv large config.
FALLBACK: Dict[str, Dict[str, object]] = {
    "cora": dict(lr=0.0005, weight_decay=0.001, attn_lr=0.001, attn_wd=1e-5, hidden=256, dropout=0.8, dprate=0.5, n_head=4, d_ffn=128, q=2.0, multi=2.0, K=6, nlayer=1, epochs=2000, patience=250),
    "wikics": dict(lr=0.001, weight_decay=1e-5, attn_lr=0.001, attn_wd=1e-5, hidden=256, dropout=0.5, dprate=0.5, n_head=8, d_ffn=128, q=1.6, multi=1.0, K=6, nlayer=2, epochs=2000, patience=250),
    "amazon_computers": dict(lr=0.001, weight_decay=1e-5, attn_lr=0.001, attn_wd=1e-5, hidden=128, dropout=0.3, dprate=0.5, n_head=8, d_ffn=128, q=1.4, multi=1.0, K=4, nlayer=2, epochs=2000, patience=250),
    "amazon_photo": dict(lr=0.001, weight_decay=1e-5, attn_lr=0.001, attn_wd=1e-5, hidden=128, dropout=0.3, dprate=0.5, n_head=8, d_ffn=128, q=1.4, multi=1.0, K=4, nlayer=2, epochs=2000, patience=250),
    "ethanol": dict(lr=0.0005, weight_decay=0.001, attn_lr=0.0005, attn_wd=1e-5, hidden=128, dropout=0.3, dprate=0.3, n_head=4, d_ffn=256, q=1.0, multi=1.0, K=8, nlayer=2, epochs=2000, patience=250),
    "facebook_large": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=10000),
    "flickr": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=10000),
    "reddit": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=20000),
    "ogbn_products": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=20000),
    "ogbn_mag": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=20000),
    "github": dict(lr=0.0002, weight_decay=0.0, attn_lr=0.001, attn_wd=0.0001, hidden=128, dropout=0.4, dprate=0.6, n_head=8, d_ffn=512, q=2.0, multi=0.5, K=10, nlayer=2, epochs=2000, patience=250, batch_size=10000),
}

DEFAULT = dict(lr=0.001, weight_decay=5e-4, attn_lr=0.001, attn_wd=1e-5, hidden=128, dropout=0.5, dprate=0.5, n_head=4, d_ffn=256, q=1.0, multi=1.0, K=6, nlayer=2, epochs=2000, patience=250, batch_size=10000)

# Datasets without an official PolyFormer command that reuse the official command of the closest
# dataset (config_source "fallback:<dataset>").
OFFICIAL_ALIAS: Dict[str, str] = {
    # PolyFormer only reports the filtered squirrel of Platonov et al. (2,223 nodes); our squirrel is
    # the original 5,201-node graph
    "squirrel": "squirrel_filtered",
    # heterophilous multi-class graph of the same benchmark (Platonov et al.), similar size
    "amazon_ratings": "roman_empire",
    # same graph as ogbn-arxiv with year labels (LINKX)
    "arxiv_year": "ogbn_arxiv",
    # citation graphs of PubMed's size (CitationFull DBLP 17.7k, CoraFull 19.8k nodes)
    "dblp": "pubmed",
    "corafull": "pubmed",
    # social network with binary node attributes (people as nodes, as in Coauthor)
    "lastfm_asia": "coauthor_cs",
}


def get_config(dataset: str, base: str = "auto") -> Dict[str, object]:
    name = norm_name(dataset)
    base_l = base.lower()
    if base_l == "auto":
        base_l = "mono"
    cfg = dict(DEFAULT)
    source = "fallback-default"
    if (name, base_l) in OFFICIAL:
        cfg.update(OFFICIAL[(name, base_l)])
        source = "official"
    elif name in OFFICIAL_ALIAS and (OFFICIAL_ALIAS[name], base_l) in OFFICIAL:
        cfg.update(OFFICIAL[(OFFICIAL_ALIAS[name], base_l)])
        source = f"fallback:{OFFICIAL_ALIAS[name]}"
    elif name in FALLBACK:
        cfg.update(FALLBACK[name])
        source = "self-set"
    cfg["base"] = base_l
    cfg["config_source"] = source
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", default="auto")
    args = p.parse_args()
    cfg = get_config(args.dataset, args.base)
    mapping = {
        "base": "CFG_BASE", "lr": "CFG_LR", "weight_decay": "CFG_WEIGHT_DECAY",
        "attn_lr": "CFG_ATTN_LR", "attn_wd": "CFG_ATTN_WD", "hidden": "CFG_HIDDEN",
        "dropout": "CFG_DROPOUT", "dprate": "CFG_DPRATE", "n_head": "CFG_N_HEAD",
        "d_ffn": "CFG_D_FFN", "q": "CFG_Q", "multi": "CFG_MULTI", "K": "CFG_K",
        "nlayer": "CFG_NLAYER", "epochs": "CFG_EPOCHS", "patience": "CFG_PATIENCE",
        "batch_size": "CFG_BATCH_SIZE", "metric": "CFG_METRIC", "force_binary": "CFG_FORCE_BINARY",
        "config_source": "CFG_CONFIG_SOURCE",
    }
    for key, env_key in mapping.items():
        if key in cfg:
            print(f"{env_key}='{cfg[key]}'")


if __name__ == "__main__":
    main()
