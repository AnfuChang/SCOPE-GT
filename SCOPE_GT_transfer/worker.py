"""An isolated complete training run, validation checkpoint selection and one final test."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from SCOPE_GT_budgets.common import Meter,dataset_manifest,sha,write_json
from SCOPE_GT_efficiency.worker import runtime
from . import PROTOCOL
from .config import CODE,implementation,quality
from .adapters import TransferAdapter
from SCOPE_GT.external_backbones import source_path


def run(spec):
    out=Path(spec['directory']); out.mkdir(parents=True,exist_ok=True)
    result={k:spec[k] for k in ('job_id','method','variant','dataset','seed','n')}
    result.update(protocol=PROTOCOL,status='failed',implementation=implementation(spec['method']),
                  metric='roc_auc' if spec['dataset']=='minesweeper' else 'accuracy')
    phase_name='preflight'; history=[]
    def phase(name,epoch=None):
        nonlocal phase_name
        phase_name=name
        for attempt in range(20):
            try:
                write_json(out/'progress.json',dict(phase=name,epoch=epoch,started_monotonic=time.monotonic()))
                break
            except PermissionError:
                if attempt==19: raise
                time.sleep(.005)
    try:
        result['runtime']=runtime(spec['device'])
        if result['runtime']!=spec['runtime']: raise ValueError('Runtime differs from frozen job')
        for name,value in spec['source_sha256'].items():
            if sha(source_path(name))!=value: raise ValueError(f'Frozen source changed: {name}')
        root=Path(spec['data_root'])
        if dataset_manifest(root)!=spec['data_sha256']: raise ValueError('Frozen data changed')
        torch.set_num_threads(spec['threads'])
        device=torch.device(spec['device']); cfg=spec['config']
        adapter=TransferAdapter(spec['method'],spec['variant'],cfg,root,spec['seed'],device,out)
        start=time.perf_counter()
        elapsed=lambda:time.perf_counter()-start
        phase('load')
        with Meter(device) as load: adapter.load()
        phase('preprocess')
        with Meter(device) as prep:
            adapter.prepare(); opt=adapter.optimizer()
        result.update(initial_state_sha256=adapter.initial_state_sha256,parameters=sum(p.numel() for p in adapter.model.parameters()),
                      load_s=load.seconds,preprocess_s=prep.seconds,train_centers=len(adapter.splits['train']))
        best=-float('inf'); best_epoch=0; bad=0
        with (out/'history.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,cfg['epochs']+1):
                phase('train',epoch)
                with Meter(device) as train: metrics=adapter.epoch(epoch,opt)
                if metrics['supervised_centers']!=len(adapter.splits['train']): raise ValueError('Incomplete epoch coverage')
                row=dict(epoch=epoch,train=train.result,warmup_epoch=epoch<=spec['warmup_epochs'],**metrics)
                if adapter.eval_due(epoch):
                    phase('validation',epoch)
                    with Meter(device) as val:
                        pred,extra=adapter.predict(adapter.splits['val'],epoch,validation=True)
                        score=quality(pred,adapter.labels[adapter.splits['val']],spec['dataset'])
                    row.update(validation=val.result,val_accuracy=score,validation_inputs=extra,elapsed_val_s=elapsed())
                    adapter.after_validation(score)
                    if not np.isfinite(score): raise ValueError('Nonfinite validation quality')
                    if score>best:
                        best,best_epoch,bad=score,epoch,0
                        phase('checkpoint',epoch)
                        temp=out/'checkpoint.tmp'
                        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in adapter.model.state_dict().items()},
                            job_id=spec['job_id'],config=cfg,best_epoch=epoch,best_val=best),temp)
                        temp.replace(out/'checkpoint.pt')
                        result['time_to_best_validation_s']=row['elapsed_val_s']
                    else:
                        # Native NLGT patience is in epochs; native Gophormer in validation checks.
                        bad+=cfg['eval_every'] if spec['method']=='nlgt' else 1
                row['elapsed_wall_s']=elapsed(); history.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(f"epoch={epoch} train_s={train.seconds:.4f} val={row.get('val_accuracy')} elapsed={elapsed():.3f}",flush=True)
                if cfg['patience'] and bad>=cfg['patience']: break
        if not best_epoch: raise ValueError('No validation-selected checkpoint')
        phase('checkpoint_load')
        checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        adapter.model.load_state_dict(checkpoint['state_dict'])
        phase('test')
        with Meter(device) as testing:
            pred,extra=adapter.predict(adapter.splits['test'],best_epoch,validation=False)
            accuracy=quality(pred,adapter.labels[adapter.splits['test']],spec['dataset'])
        phase('export')
        np.savez_compressed(out/'test_predictions.npz',ids=adapter.splits['test'],labels=adapter.labels[adapter.splits['test']],predictions=pred)
        measured=[r for r in history if not r['warmup_epoch']]
        peaks=[r['train']['gpu_peak_allocated_bytes'] for r in history]
        complete_peaks=peaks+[load.result['gpu_peak_allocated_bytes'],prep.result['gpu_peak_allocated_bytes'],testing.result['gpu_peak_allocated_bytes']]
        complete_peaks += [r['validation']['gpu_peak_allocated_bytes'] for r in history if 'validation' in r]
        result.update(status='complete',epochs_run=len(history),measured_epochs=len(measured),best_epoch=best_epoch,
            best_val_accuracy=best,test_accuracy=accuracy,test_seconds=testing.seconds,
            test_centers_per_second=len(adapter.splits['test'])/testing.seconds,test_inputs=extra,
            epoch_seconds_mean=float(np.mean([r['train']['seconds'] for r in measured])) if measured else None,
            epoch_seconds_within_run_sd=float(np.std([r['train']['seconds'] for r in measured],ddof=1)) if len(measured)>1 else None,
            train_peak_allocated_bytes=max([x for x in peaks if x is not None],default=None),
            complete_gpu_peak_allocated_bytes=max([x for x in complete_peaks if x is not None],default=None),
            test_gpu_peak_allocated_bytes=testing.result['gpu_peak_allocated_bytes'],
            validation_total_seconds=sum(r['validation']['seconds'] for r in history if 'validation' in r),
            cpu_rss_peak_bytes=max([load.result['rss_sampled_peak_bytes'],prep.result['rss_sampled_peak_bytes'],testing.result['rss_sampled_peak_bytes']]+
                [r['train']['rss_sampled_peak_bytes'] for r in history]+[r['validation']['rss_sampled_peak_bytes'] for r in history if 'validation' in r]),
            timing_scope='load, preprocess, train, validation, checkpoint, one final test and prediction export; imports/integrity preflight excluded')
        result['job_seconds']=elapsed()
        phase('complete')
    except Exception as error:
        result.update(status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed',
                      failed_phase=phase_name,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    write_json(out/'metrics.json',result)
    return result['status']=='complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path); p.add_argument('--probe')
    a=p.parse_args()
    if a.probe: print(json.dumps(runtime(a.probe)))
    elif not a.spec: p.error('--spec required')
    elif not run(json.loads(a.spec.read_text(encoding='utf-8'))): raise SystemExit(1)


if __name__=='__main__': main()
