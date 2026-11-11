# Official configuration notes

The official PolyFormer source has two node-classification folders.

## Small/medium graphs: Table 5

`node_classification/node_classification.sh` lists two official variants:

- `base=mono`
- `base=cheb`

This package transcribes both groups into `polyformer_node/configs.py`.

## Large graphs: Table 6

`node_classification_large_graph/node_classification_large_graph.sh` lists official configs for:

- `twitch-gamer`
- `pokec`
- `ogbn-arxiv`
- `ogbn-papers100M`

This package transcribes those configs as `twitch_gamer`, `pokec`, `ogbn_arxiv`, and `ogbn_papers100m` / `Paper100m` aliases.

## Dataset names in our project

The runner normalizes names internally by lowercasing and replacing hyphens with underscores. Therefore, these forms are accepted:

```text
PubMed -> pubmed
Questions -> questions
Flickr -> flickr
ogbn-arxiv -> ogbn_arxiv
roman-empire -> roman_empire
coauthor_cs -> coauthor_cs
coauthor_physics -> coauthor_physics
```

## Missing official configs

The official PolyFormer scripts do not contain configs for several datasets commonly used in our project, including Cora, Flickr, Reddit, ogbn-products, ogbn-mag, Amazon Computers, Amazon Photo, WikiCS, GitHub, Facebook-large, and Ethanol. For these datasets, `config_source` becomes `self-set` or `fallback-default` in the output summaries.
