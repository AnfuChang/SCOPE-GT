"""One real method/run per process; epoch timers include online data preparation."""
import argparse
import json
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from SCOPE_GT_budgets.common import Meter, dataset_manifest, sha, sync, write_json
from . import PROTOCOL
from .adapters import make_adapter
from .config import CODE, identity_note
from SCOPE_GT.external_backbones import source_path


def runtime(device):
    import scipy
    import sklearn
    # Keep all methods in full FP32; no method-specific TF32 acceleration.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = torch.device(device)
    if d.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
    if d.type == 'cuda':
        torch.cuda.set_device(d)
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,precision='float32',tf32=False,platform=platform.platform(),
                cuda=torch.version.cuda, device=str(d), gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,
                executable=sys.executable)


def run(spec):
    out = Path(spec['directory'])
    out.mkdir(parents=True,exist_ok=True)
    result = dict(protocol=PROTOCOL,job_id=spec['job_id'],method=spec['method'],dataset=spec['dataset'],
                  study=spec['study'],seed=spec['seed'],n=spec['n'],status='failed',
                  implementation=identity_note(spec['method']))
    phase_name = 'preflight'
    history = []
    def phase(name,epoch=None):
        nonlocal phase_name
        phase_name = name
        value=dict(phase=name,epoch=epoch,started_monotonic=time.monotonic())
        # Windows can deny replace while the parent briefly has the old file open.
        # Bounded retries preserve atomic phase changes; persistent permission errors remain failures.
        for attempt in range(20):
            try:
                write_json(out/'progress.json',value)
                break
            except PermissionError:
                if attempt==19: raise
                time.sleep(.005)
    try:
        result['runtime'] = runtime(spec['device'])
        if result['runtime'] != spec['runtime']:
            raise ValueError('Worker runtime differs from planned interpreter/hardware')
        for name,value in spec['source_sha256'].items():
            if sha(source_path(name)) != value:
                raise ValueError(f'Frozen source changed: {name}')
        root = Path(spec['data_root'])
        if dataset_manifest(root) != spec['data_sha256']:
            raise ValueError('Frozen input changed')
        torch.set_num_threads(spec['threads'])
        device = torch.device(spec['device'])
        cfg = spec['config']
        adapter = make_adapter(spec['method'],cfg,root,spec['seed'],device,out)
        # Imports, integrity preflight and runtime initialization precede this clock.
        start = time.perf_counter()
        phase('load')
        with Meter(device) as load:
            adapter.load()
        phase('preprocess')
        with Meter(device) as prep:
            adapter.prepare()
            optimizer = adapter.optimizer()
        result.update(load_s=load.seconds,preprocess_s=prep.seconds,setup_complete_s=time.perf_counter()-start,
                      train_centers=len(adapter.splits['train']),feature_location=adapter.feature_location,
                      parameters=sum(p.numel() for p in adapter.model.parameters()))
        scaling = spec['study']=='scaling'
        epochs = spec['warmup_epochs']+spec['measured_epochs'] if scaling else cfg['epochs']
        best, best_epoch, bad = -float('inf'), 0, 0
        targets = {str(x):None for x in spec.get('targets',[])}
        def elapsed():
            return time.perf_counter()-start
        with (out/'history.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,epochs+1):
                phase('train',epoch)
                with Meter(device) as train_meter:
                    metrics = adapter.epoch(epoch,optimizer)
                if not np.isfinite(metrics['loss']):
                    raise ValueError('Nonfinite training loss')
                if metrics.get('skipped_batches',0):
                    raise ValueError('Training skipped nonfinite batches; resource results would have incomplete coverage')
                if metrics['supervised_centers'] != len(adapter.splits['train']):
                    raise ValueError('Efficiency epochs must cover every training center')
                row = dict(epoch=epoch,warmup_epoch=epoch<=spec['warmup_epochs'],train=train_meter.result,
                           epoch_scope='complete_training_split',training_centers_available=len(adapter.splits['train']),
                           **metrics,elapsed_train_s=elapsed())
                if not scaling:
                    phase('validation',epoch)
                    with Meter(device) as val_meter:
                        predictions, _ = adapter.predict(adapter.splits['val'],epoch,validation=True)
                        labels = adapter.labels[adapter.splits['val']]
                        score = float(np.mean(np.argmax(predictions,axis=1)==labels))
                    row.update(validation=val_meter.result,val_accuracy=score,elapsed_val_s=elapsed())
                    if not np.isfinite(score):
                        raise ValueError('Nonfinite validation quality')
                    for target in targets:
                        if targets[target] is None and score >= float(target):
                            targets[target] = row['elapsed_val_s']
                    if hasattr(adapter,'after_validation'):
                        adapter.after_validation(score)
                    if score > best + (1e-6 if spec['method']=='scope_gt' else 0):
                        best,best_epoch,bad = score,epoch,0
                        phase('checkpoint',epoch)
                        temp = out/'checkpoint.tmp'
                        torch.save(dict(job_id=spec['job_id'],config=cfg,best_epoch=epoch,best_val=best,
                            state_dict={k:v.detach().cpu().clone() for k,v in adapter.model.state_dict().items()}),temp)
                        temp.replace(out/'checkpoint.pt')
                    else:
                        bad += 1
                row['elapsed_wall_s'] = elapsed()
                history.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(f"epoch={epoch} train_s={train_meter.seconds:.4f} val={row.get('val_accuracy')} elapsed={elapsed():.3f}",flush=True)
                if not scaling and cfg['patience_epochs'] and bad>=cfg['patience_epochs']:
                    break
        measured = [r for r in history if not r['warmup_epoch']]
        train_peaks = [r['train']['gpu_peak_allocated_bytes'] for r in (measured if scaling else history)]
        train_peaks = [x for x in train_peaks if x is not None]
        selection_peaks = train_peaks + [r['validation']['gpu_peak_allocated_bytes'] for r in history
            if r.get('validation',{}).get('gpu_peak_allocated_bytes') is not None]
        result.update(status='complete',epochs_run=len(history),
            epoch_seconds_mean=float(np.mean([r['train']['seconds'] for r in measured])) if measured else None,
            train_peak_allocated_bytes=max(train_peaks) if train_peaks else None,
            selection_peak_allocated_bytes=max(selection_peaks) if selection_peaks else None,
            cpu_rss_peak_bytes=max([load.result['rss_sampled_peak_bytes'],prep.result['rss_sampled_peak_bytes']]+
                [r['train']['rss_sampled_peak_bytes'] for r in history]+
                [r['validation']['rss_sampled_peak_bytes'] for r in history if 'validation' in r]),
            time_to_targets=targets,observed_until_s=elapsed(),
            preprocessing_peak_allocated_bytes=prep.result['gpu_peak_allocated_bytes'])
        if not scaling:
            if not best_epoch:
                raise RuntimeError('No validation-selected checkpoint')
            phase('test')
            checkpoint = torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
            adapter.model.load_state_dict(checkpoint['state_dict'])
            with Meter(device) as testing:
                predictions,_ = adapter.predict(adapter.splits['test'],best_epoch,validation=False)
                result['test_accuracy'] = float(np.mean(np.argmax(predictions,axis=1)==adapter.labels[adapter.splits['test']]))
            np.savez_compressed(out/'test_predictions.npz',ids=adapter.splits['test'],predictions=predictions)
            result.update(best_val_accuracy=best,best_epoch=best_epoch,test_seconds=testing.seconds)
        result.update(job_seconds=elapsed(),timing_scope='data loading through required export; excludes imports and integrity preflight')
        result['complete_gpu_peak_allocated_bytes'] = max([x for x in [load.result['gpu_peak_allocated_bytes'],
            prep.result['gpu_peak_allocated_bytes'],result['selection_peak_allocated_bytes'],
            testing.result['gpu_peak_allocated_bytes'] if not scaling else None] if x is not None],default=None)
        phase('complete')
    except Exception as error:
        result.update(status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed',
                      failed_phase=phase_name,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    write_json(out/'metrics.json',result)
    return result['status']=='complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path)
    p.add_argument('--probe')
    args=p.parse_args()
    if args.probe:
        print(json.dumps(runtime(args.probe)))
    else:
        if not args.spec:
            p.error('--spec is required')
        if not run(json.loads(args.spec.read_text(encoding='utf-8'))):
            raise SystemExit(1)


if __name__=='__main__':
    main()
