"""Shared comparison protocol for the baselines adapted after 2026-09-30.

Every adapter (GCFormer, SwapGT, HubGT, NeuralWalker, GQT) goes through this module so that the
parts that must be identical across methods are implemented once:

  data      adj.npz / feat.npy / label.npy / train|val|test<suffix>.npy, read with common/data.py,
            which is byte-identical to ACE_GTv6/data.py (same symmetrisation, self-loop removal and
            split checks as our model; features are left raw unless --feature-normalize l1)
  runs      run r uses seed base_seed + r and split suffix split_suffixes[r % len]
  training  at most --epochs epochs; the validation metric is checked at epoch 1, every
            --eval-every epochs and at the last epoch; training stops after --patience checks
            without improvement (same rule as Baselines/Gophormer/gophormer_npz.py)
  selection checkpoint with the best validation metric (accuracy, or ROC-AUC for the binary,
            imbalanced datasets); the test set is evaluated once, with that checkpoint
  cost      Tepoch = mean over all epochs of (training time + validation time on check epochs),
            Mem = peak torch.cuda.max_memory_allocated during training and evaluation (the two
            numbers reported by run_gophormer.sh); preprocessing and test inference are timed
            separately
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import torch


def _load_common_data():
    # Loaded by path under a private name: several official repos ship their own data.py.
    path = Path(__file__).resolve().parent / "data.py"
    spec = importlib.util.spec_from_file_location("ace_baselines_common_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_graph = _load_common_data().load_graph  # common/data.py, byte-identical to ACE_GTv6/data.py

# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh)
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def metric_name(metric: str) -> str:
    """Label printed in logs and summaries: 'ROC_AUC' for the ROC-AUC datasets, 'Acc' otherwise."""
    return "ROC_AUC" if metric == "auc" else "Acc"


# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------

def add_protocol_args(parser):
    g = parser.add_argument_group("comparison protocol (shared by all baselines)")
    g.add_argument("--data-root", required=True, help="directory with adj.npz, feat.npy, label.npy and splits")
    g.add_argument("--dataset", default=None, help="dataset name used for presets/logging (default: basename of data-root)")
    g.add_argument("--out-dir", default=None, help="where summary.json / runs.csv are written")
    g.add_argument("--device", default="auto")
    g.add_argument("--runs", type=int, default=5)
    g.add_argument("--base-seed", type=int, default=42)
    g.add_argument("--split-suffixes", nargs="*", default=[""],
                   help="run r reads train<s>.npy/val<s>.npy/test<s>.npy with s = suffixes[r %% len]")
    g.add_argument("--epochs", type=int, default=1000)
    g.add_argument("--patience", type=int, default=30, help="validation checks without improvement")
    g.add_argument("--eval-every", type=int, default=5)
    g.add_argument("--metric", choices=["acc", "auc", "rocauc", "roc_auc"], default=None,
                   help="default: auc (ROC-AUC) for %s, acc otherwise" % ", ".join(AUC_DATASETS))
    g.add_argument("--feature-normalize", choices=["none", "l1"], default="none",
                   help="none = raw feat.npy (the official preprocessing of each method is applied on top)")
    return parser


def finalize_args(args):
    args.data_root = str(Path(args.data_root))
    if args.dataset is None:
        args.dataset = Path(args.data_root).name
    args.dataset_key = args.dataset.lower().replace("-", "_")
    if args.metric is None:
        args.metric = "auc" if args.dataset_key in AUC_DATASETS else "acc"
    elif args.metric in ("rocauc", "roc_auc"):
        args.metric = "auc"
    if not args.split_suffixes:
        args.split_suffixes = [""]
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        # no silent CPU fallback: the timing and memory columns would no longer be comparable
        try:
            torch.cuda.init()  # raises with the actual reason (CPU-only build, no driver, driver too old, ...)
            reason = "torch.cuda.init() succeeded but is_available() is False"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        raise SystemExit(
            f"{device_arg} requested but CUDA is unavailable (use DEVICE=cpu explicitly to run on CPU)\n"
            f"  python               {sys.executable}\n"
            f"  torch                {torch.__version__} (built for CUDA {torch.version.cuda})\n"
            f"  CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
            f"  reason               {reason}")
    if device.type == "cuda" and device.index is not None:
        # [ACE-baseline] make it the current CUDA device: DGL (GQT's tokenizer) and other extension kernels
        # launch on the current device, not on the device of their tensors; with --device cuda:1 and the
        # default current device 0 they read cuda:1 memory from cuda:0 -> "illegal memory access"
        torch.cuda.set_device(device)
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

class ProtocolData:
    """Graph loaded once; splits loaded per suffix with the checks of common/data.py."""

    def __init__(self, args):
        self.args = args
        suffix0 = args.split_suffixes[0]
        graph = load_graph(args.data_root, suffix0, normalize=args.feature_normalize)
        self.adj = graph.adj                      # symmetric CSR, no self loops, sorted indices
        self.features = graph.features            # float32 [N, F]
        self.labels = graph.labels                # int64 [N], -1 allowed for unlabeled nodes
        self.num_nodes = int(self.labels.shape[0])
        self.num_features = int(self.features.shape[1])
        self.num_classes = int(self.labels[self.labels >= 0].max()) + 1
        self.fingerprint = graph.fingerprint
        self._splits = {suffix0: graph.splits}

    def splits(self, suffix: str) -> Dict[str, np.ndarray]:
        if suffix not in self._splits:
            # load_graph re-validates everything; features/adjacency are identical across suffixes
            graph = load_graph(self.args.data_root, suffix, normalize=self.args.feature_normalize)
            self._splits[suffix] = graph.splits
        return self._splits[suffix]

    def edge_index(self) -> np.ndarray:
        """[2, nnz] int64 of the symmetric adjacency (both directions, no self loops)."""
        coo = self.adj.tocoo()
        return np.vstack([coo.row, coo.col]).astype(np.int64)

    def summary(self) -> str:
        return (f"nodes={self.num_nodes} undirected_edges={self.adj.nnz // 2} "
                f"features={self.num_features} classes={self.num_classes}")


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

def compute_metric(metric: str, scores, labels, kind: str = "logits") -> float:
    """scores: [n, C] logits, log-probabilities or probabilities (kind), or [n] / [n, 1] binary logits.

    Accuracy uses the arg-max class (threshold 0 for a single binary logit). ROC-AUC uses a score
    that is rank-equivalent to P(class 1): s1 - s0 for two columns, the logit itself for one column.
    """
    from sklearn.metrics import roc_auc_score

    s = scores.detach().float().cpu() if torch.is_tensor(scores) else torch.as_tensor(scores).float()
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    y = y.reshape(-1).astype(np.int64)
    single = s.dim() == 1 or s.shape[-1] == 1
    if single:
        s = s.reshape(-1)
    if metric == "acc":
        if single:
            pred = (s > (0.5 if kind == "probs" else 0.0)).long().numpy()
        else:
            pred = s.argmax(dim=-1).numpy()
        return float((pred == y).mean())
    if metric == "auc":
        if single:
            return float(roc_auc_score(y, s.numpy()))
        if s.shape[-1] == 2:
            return float(roc_auc_score(y, (s[:, 1] - s[:, 0]).numpy()))
        prob = s if kind == "probs" else torch.softmax(s, dim=-1)
        return float(roc_auc_score(y, prob.numpy(), multi_class="ovr"))
    raise ValueError(metric)


# ----------------------------------------------------------------------------------------------
# Timing / memory
# ----------------------------------------------------------------------------------------------

class CostMeter:
    """with CostMeter(device) as m: ...  ->  m.seconds, m.peak_mb (torch.cuda.max_memory_allocated)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = 0.0
        self.peak_mb = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        self.seconds = time.perf_counter() - self._t0
        return False


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ----------------------------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = -math.inf
        self.best_epoch = 0
        self.best_state = None
        self.bad_checks = 0

    def step(self, score: float, epoch: int, model: torch.nn.Module) -> bool:
        if score > self.best:
            self.best = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.bad_checks >= self.patience


@dataclass
class RunResult:
    run: int
    seed: int
    split_suffix: str
    best_epoch: int
    stop_epoch: int
    best_val: float
    test: float
    epoch_time_s: float          # Tepoch: mean over all epochs of train + validation-check time
    train_time_s: float          # mean training time per epoch
    val_time_s: float            # mean time of one validation check
    peak_mem_mb: float           # max over training and validation
    infer_time_s: float = 0.0    # final test evaluation with the best checkpoint
    infer_peak_mem_mb: float = 0.0
    pre_time_s: float = 0.0      # method-specific preprocessing (token/index/tokenizer construction)
    pre_peak_mem_mb: float = 0.0
    params: int = 0
    time_to_best_s: float = 0.0  # training + validation time from epoch 1 to the end of the best epoch
    extra: Dict[str, float] = field(default_factory=dict)


def fit(
    model: torch.nn.Module,
    train_epoch: Callable[[int], float],
    evaluate: Callable[[str], float],
    args,
    device: torch.device,
    tag: str,
    evaluate_test: Optional[Callable[[], float]] = None,
) -> Dict[str, float]:
    """Runs the shared loop.

    train_epoch(epoch) -> mean training loss (the adapter owns optimizer/scheduler steps)
    evaluate(split)    -> metric on 'val' (called on check epochs) or 'test'
    evaluate_test()    -> optional replacement for evaluate('test') after restoring the best
                          checkpoint (e.g. to average several stochastic inference passes)
    """
    stopper = EarlyStopper(args.patience)
    epoch_times, train_times, val_times, peaks = [], [], [], []
    stop_epoch = 0
    for epoch in range(1, args.epochs + 1):
        stop_epoch = epoch
        with CostMeter(device) as tm:
            loss = train_epoch(epoch)
        check = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        val_s, val_peak, stop = 0.0, 0.0, False
        if check:
            with CostMeter(device) as vm:
                val = float(evaluate("val"))
            val_s, val_peak = vm.seconds, vm.peak_mb
            val_times.append(val_s)
            improved = val > stopper.best
            stop = stopper.step(val, epoch, model)
            print(f"[{tag}] epoch={epoch:04d} loss={loss:.4f} val_{metric_name(args.metric)}={val:.4f}"
                  f"{' *' if improved else ''} best={stopper.best:.4f}@{stopper.best_epoch} "
                  f"time={tm.seconds + val_s:.3f}s(train={tm.seconds:.3f},val={val_s:.3f}) "
                  f"peak_mem={max(tm.peak_mb, val_peak):.1f}MB", flush=True)
        epoch_times.append(tm.seconds + val_s)
        train_times.append(tm.seconds)
        peaks.append(max(tm.peak_mb, val_peak))
        if stop:
            print(f"[{tag}] early stop at epoch {epoch}: no improvement in {args.patience} checks", flush=True)
            break
    if stopper.best_state is None:
        raise RuntimeError("training finished without a validation check")
    model.load_state_dict(stopper.best_state)
    with CostMeter(device) as im:
        test = float(evaluate_test() if evaluate_test is not None else evaluate("test"))
    print(f"[{tag}] best_epoch={stopper.best_epoch} stop_epoch={stop_epoch} best_val={stopper.best:.4f} "
          f"test_{metric_name(args.metric)}={test:.4f} infer={im.seconds:.3f}s", flush=True)
    return dict(
        best_epoch=stopper.best_epoch,
        stop_epoch=stop_epoch,
        best_val=stopper.best,
        test=test,
        epoch_time_s=float(np.mean(epoch_times)),
        train_time_s=float(np.mean(train_times)),
        val_time_s=float(np.mean(val_times)) if val_times else 0.0,
        peak_mem_mb=float(max(peaks) if peaks else 0.0),
        infer_time_s=im.seconds,
        infer_peak_mem_mb=im.peak_mb,
        # epoch e is epoch_times[e - 1]; the best epoch is always a validation-check epoch
        time_to_best_s=float(np.sum(epoch_times[:stopper.best_epoch])),
    )


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------

def _mean_std(values: Sequence[float]):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def print_efficiency(params: int, train_epoch_s: Sequence[float], time_to_best_s: Sequence[float],
                     peak_mem_mb: float, pre_s: Optional[Sequence[float]] = None,
                     best_epoch: Optional[Sequence[float]] = None, pre_peak_mem_mb: Optional[float] = None,
                     extra_params: Optional[Dict[str, int]] = None) -> None:
    """Efficiency block printed with the same labels and definitions by every baseline:
      Params                trainable parameters of the model trained in the epoch loop (models
                            trained during preprocessing, e.g. GQT's tokenizer, are listed apart)
      Train time/epoch (s)  training only (forward, backward, optimizer, batch construction),
                            mean over epochs and runs; evaluation is excluded
      Time to best (s)      training + validation-check time from epoch 1 to the end of the epoch
                            with the best validation score (mean ± std over runs); test
                            evaluation and one-off preprocessing are excluded
      Preprocessing (s)     one-off, method-specific preprocessing (only if the method has any)
      Peak GPU mem (MB)     torch.cuda.max_memory_allocated during training and validation, max
                            over runs (with preprocessing: printed separately as well)
    """
    ttb = np.asarray(time_to_best_s, dtype=np.float64)
    ttb_std = float(ttb.std(ddof=1)) if ttb.size > 1 else 0.0
    print("==== Efficiency ====")
    print(f"Params: {int(params):,} ({int(params) / 1e6:.3f}M)")
    if extra_params:
        for name, n in extra_params.items():
            print(f"Params ({name}): {int(n):,} ({int(n) / 1e6:.3f}M)")
        total = int(params) + sum(int(n) for n in extra_params.values())
        print(f"Params (total): {total:,} ({total / 1e6:.3f}M)")
    print(f"Train time/epoch (s): {float(np.mean(train_epoch_s)):.4f}")
    at = f" (best epoch {float(np.mean(best_epoch)):.1f})" if best_epoch is not None else ""
    print(f"Time to best (s): {ttb.mean():.2f} ± {ttb_std:.2f}{at}")
    if pre_s is not None:
        pre = np.asarray(pre_s, dtype=np.float64)
        print(f"Preprocessing (s): {pre.mean():.2f}")
        print(f"Time to best incl. preprocessing (s): {(ttb + pre).mean():.2f}")
    print(f"Peak GPU mem (MB): {peak_mem_mb:.2f}")
    if pre_peak_mem_mb:
        print(f"Peak GPU mem incl. preprocessing (MB): {max(peak_mem_mb, pre_peak_mem_mb):.2f}")


def report(method: str, results: List[RunResult], args, extra_config: Optional[dict] = None) -> dict:
    name = metric_name(args.metric)
    test_m, test_s = _mean_std([r.test for r in results])
    val_m, val_s = _mean_std([r.best_val for r in results])
    summary = {
        "method": method,
        "dataset": args.dataset,
        "metric": args.metric,
        "metric_name": name,
        "runs": len(results),
        "test_mean": test_m, "test_std": test_s,
        "val_mean": val_m, "val_std": val_s,
        "epoch_time_s_mean": _mean_std([r.epoch_time_s for r in results])[0],
        "train_time_s_mean": _mean_std([r.train_time_s for r in results])[0],
        "val_time_s_mean": _mean_std([r.val_time_s for r in results])[0],
        "peak_mem_mb_max": float(max(r.peak_mem_mb for r in results)),
        "infer_time_s_mean": _mean_std([r.infer_time_s for r in results])[0],
        "pre_time_s_mean": _mean_std([r.pre_time_s for r in results])[0],
        "pre_peak_mem_mb_max": float(max(r.pre_peak_mem_mb for r in results)),
        "best_epoch_mean": _mean_std([r.best_epoch for r in results])[0],
        "stop_epoch_mean": _mean_std([r.stop_epoch for r in results])[0],
        "time_to_best_s_mean": _mean_std([r.time_to_best_s for r in results])[0],
        "time_to_best_s_std": _mean_std([r.time_to_best_s for r in results])[1],
        "params": results[0].params if results else 0,
    }
    print(f"\n==== Summary ({method}, {args.dataset}) ====")
    print(f"seeds={[r.seed for r in results]} splits={[r.split_suffix or 'train.npy' for r in results]}")
    print(f"Test {name}: {test_m * 100:.2f} ± {test_s * 100:.2f}")
    print(f"Tepoch (s): {summary['epoch_time_s_mean']:.2f}")
    print(f"Mem. (MB): {summary['peak_mem_mb_max']:.2f}")
    print(f"Val {name}: {val_m * 100:.2f} ± {val_s * 100:.2f}")
    print(f"Train time/epoch (s): {summary['train_time_s_mean']:.3f}  Val check (s): {summary['val_time_s_mean']:.3f}  "
          f"Test inference (s): {summary['infer_time_s_mean']:.3f}")
    print(f"Preprocessing (s): {summary['pre_time_s_mean']:.2f}  Pre peak mem (MB): {summary['pre_peak_mem_mb_max']:.2f}  "
          f"Params: {summary['params'] / 1e6:.3f}M")
    print(f"Best epoch: {summary['best_epoch_mean']:.1f}  Stop epoch: {summary['stop_epoch_mean']:.1f}")
    has_pre = any(r.pre_time_s > 0 for r in results)
    extra_params = {k[:-len("_params")]: int(v) for k, v in results[0].extra.items() if k.endswith("_params")}
    print_efficiency(summary["params"], [r.train_time_s for r in results], [r.time_to_best_s for r in results],
                     summary["peak_mem_mb_max"], pre_s=[r.pre_time_s for r in results] if has_pre else None,
                     best_epoch=[r.best_epoch for r in results],
                     pre_peak_mem_mb=summary["pre_peak_mem_mb_max"] if has_pre else None,
                     extra_params=extra_params or None)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))}
        payload = {"summary": summary, "config": config, "extra_config": extra_config or {},
                   "runs": [asdict(r) for r in results], "command": " ".join(sys.argv)}
        (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [k for k in asdict(results[0]).keys() if k != "extra"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row.pop("extra")
                writer.writerow(row)
        print(f"Results saved to: {out}")
    return summary


def default_out_dir(script_dir: Path, method: str, dataset: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(script_dir / "results" / f"{dataset}_{method}_{stamp}_{os.getpid()}")
