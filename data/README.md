# Datasets

All methods (SCOPE-GT and every baseline) read the same files. Each dataset is one folder under `DATA_BASE`:

```
DATA_BASE/<name>/
  adj.npz     scipy.sparse CSR adjacency (N x N); symmetrized, binarized and stripped of self loops when loaded
  feat.npy    float32 node features (N x F)
  label.npy   int64 node labels (N,), classes 0..C-1
  train.npy   int64 training node ids  (boolean masks of length N are also accepted)
  val.npy     int64 validation node ids
  test.npy    int64 test node ids
```

The three splits must be disjoint. `label_mapping.json` is optional. Official conversions include
`metadata.json`; the trainer checks selected split hashes and records their source and split identity.
SCOPE-GT reads `adj.npz` a second time to recover edge directions for the topology relation; directions are
ignored automatically when the stored graph is symmetric.

| Paper name | Folder | Source | Split (train/val/test) |
|---|---|---|---|
| LastFM | `lastfm_asia` | SNAP LastFM-Asia (`prepare_new_datasets.py`) | 60/20/20 |
| Minesweeper | `minesweeper` | Platonov et al. (2023), official ten fixed masks | 50/25/25 |
| WikiCS | `wikics` | Mernyei & Cangea (2020), official split 0 | ≈5/15/50 |
| Tolokers | `tolokers` | Platonov et al. (2023), official ten fixed masks | 50/25/25 |
| Computers | `amazon_computers` | Shchur et al. (2018) | 60/20/20 |
| DBLP | `dblp` | Bojchevski & Günnemann (2018) (`prepare_new_datasets.py`) | 60/20/20 |
| CS | `coauthor_cs` | Shchur et al. (2018) | 60/20/20 |
| PubMed | `pubmed` | Sen et al. (2008) | 60/20/20 |
| CoraFull | `corafull` | Bojchevski & Günnemann (2018) | 60/20/20 |
| Facebook | `facebook_large` | Rozemberczki et al. (2021) | 60/20/20 |
| Roman | `roman_empire` | Platonov et al. (2023), official ten fixed masks | 50/25/25 |
| Ratings | `amazon_ratings` | Platonov et al. (2023), official ten fixed masks | 50/25/25 |
| GitHub | `github` | Rozemberczki et al. (2021) | 60/20/20 |
| Arxiv-year | `arxiv_year` | Lim et al. (2021), LINKX split 0 (`prepare_linkx.py`) | 50/25/25 |
| Pokec | `pokec` | Lim et al. (2021), LINKX split 0 (`prepare_linkx.py`) | 50/25/25 |
| Products | `ogbn_products` | Hu et al. (2020), OGB split | 8/2/90 |

All methods in a matched comparison must use the same processed graph, features and split arrays.
For the four Platonov datasets, `prepare_official.py` preserves all ten provided masks as
`train_split0.npy`/`val_split0.npy`/`test_split0.npy` through split 9; it does not generate replacement masks.
The unsuffixed files are exact aliases of official split 0. The trainer and classification launcher
default to numbered official split 0 for these four datasets. Select another fixed mask with
`--split-index <N>` (0..9) or `--split-suffix _split<N>`. The shell main launcher accepts
`SPLIT_INDEX=<N>`. Training seeds control initialization and sampling; they do not select or regenerate
a data split. Each selected split is run with five seeds (0..4), and its mean and sample standard
deviation are computed across these five runs. Different split IDs are kept separate.

Products uses the OGB sales-ranking split returned by `NodePropPredDataset.get_idx_split()`.
`prepare_new_datasets.py` draws stratified 60/20/20 splits with seed 0 for LastFM and DBLP only. For reconstructed Pokec,
the LINKX feature file is no longer available; `prepare_linkx.py` rebuilds the graph and labels from SNAP
soc-Pokec with the original LINKX splits and a 19-dimensional numeric profile encoding, which all methods receive.

Conversion scripts (no network access needed once the raw files are downloaded; run with `--help`):

```bash
python data/prepare_new_datasets.py --lastfm-zip lastfm_asia.zip --dblp-npz dblp.npz --out-root $DATA_BASE
python data/prepare_linkx.py --list-urls
python data/prepare_linkx.py arxiv-year --raw-dir /path/to/raw --out-dir $DATA_BASE/arxiv_year
python data/prepare_linkx.py pokec      --raw-dir /path/to/raw --out-dir $DATA_BASE/pokec
python data/prepare_official.py platonov minesweeper --raw-npz /path/to/minesweeper.npz --out-dir $DATA_BASE/minesweeper
python data/prepare_official.py platonov roman_empire --raw-npz /path/to/roman_empire.npz --out-dir $DATA_BASE/roman_empire
python data/prepare_official.py platonov amazon_ratings --raw-npz /path/to/amazon_ratings.npz --out-dir $DATA_BASE/amazon_ratings
python data/prepare_official.py platonov tolokers --raw-npz /path/to/tolokers.npz --out-dir $DATA_BASE/tolokers
pip install ogb  # optional, needed only for the Products converter
python data/prepare_official.py products --ogb-root /path/to/downloaded/ogb --out-dir $DATA_BASE/ogbn_products
SPLIT_INDEX=0 bash scripts/run_main.sh minesweeper roman_empire amazon_ratings tolokers
python -m SCOPE_GT.train --preset roman_empire --data-root "$DATA_BASE/roman_empire" --split-index 1 --seeds 0 1 2 3 4
```

The official converters require new or empty output directories. They preserve existing historical
datasets and results; converting inputs or changing a default does not rerun experiments or update table values.
An existing legacy single-split dataset can still be selected explicitly with `--split-suffix ""`;
its split identity must be described using its actual arrays rather than the official-mask policy.
