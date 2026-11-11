"""Launch Native/+SCMB paired complete runs; optional center_padded diagnostic."""
import argparse
import json
import os
import time
from pathlib import Path
import subprocess
import sys
import numpy as np

from SCOPE_GT_budgets.common import dataset_manifest,identity,lock,sha,write_json
from SCOPE_GT_efficiency.run import source_manifest as baseline_sources
from . import PROTOCOL
from .config import CODE,METHODS,DEFAULT_METHODS,VARIANTS,DATASETS,canonical,config_for
from .report import summarize

HERE=Path(__file__).resolve().parent


def sources(method):
    if method in ('hubgt','swapgt'):
        from .extra_backbones import extra_sources
        result=extra_sources(method)
    else:
        result=baseline_sources(method)
    for name in ('__init__.py','config.py','batches.py','model.py','adapters.py','worker.py','run.py'):
        path=HERE/name; result[path.relative_to(CODE).as_posix()]=sha(path)
    path=CODE/'SCOPE_GT/packed_ops.py'; result[path.relative_to(CODE).as_posix()]=sha(path)
    return result


def launch(spec,python,args):
    out=Path(spec['directory']); out.mkdir(parents=True,exist_ok=True)
    with lock(out/'.lock'):
        marker=out/'spec.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['job_id']!=spec['job_id']:
            raise ValueError(f'Existing output has another source/config/data/runtime identity: {out}; use a fresh --out')
        metrics=out/'metrics.json'
        if metrics.exists():
            old=json.loads(metrics.read_text(encoding='utf-8'))
            if old.get('status')=='complete' and old.get('job_id')==spec['job_id']:
                needed=['history.jsonl']+(['checkpoint.pt','test_predictions.npz'] if spec['study']!='scaling' else [])
                if not all((out/name).exists() for name in needed):
                    raise ValueError(f'Missing completed artifacts: {out}')
                print(f'[reuse] {spec["study"]} {spec["method"]} N={spec["n"]} seed={spec["seed"]}',flush=True)
                return old
        write_json(marker,spec)
        (out/'progress.json').unlink(missing_ok=True)
        metrics.unlink(missing_ok=True)
        (out/'history.jsonl').unlink(missing_ok=True)
        command=[python,'-u','-m','SCOPE_GT_transfer.worker','--spec',str(marker)]
        started=time.monotonic(); timeout_status=None; progress={}
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1',OMP_NUM_THREADS=str(args.threads),
                 MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
        with (out/'stdout.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(.25)
                try: progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
                except (OSError,json.JSONDecodeError): pass
                now=time.monotonic()
                phase=progress.get('phase')
                phase_limit=args.preprocess_timeout if phase in ('load','preprocess') else (
                    args.epoch_timeout if phase in ('train','validation') else 0)
                if phase_limit and now-progress.get('started_monotonic',now)>phase_limit:
                    timeout_status='epoch_timeout' if phase=='train' else f'{phase}_timeout'
                if args.job_timeout and now-started>args.job_timeout:
                    timeout_status=timeout_status or 'job_timeout'
                if timeout_status:
                    process.kill(); process.wait(); break
        if timeout_status or not metrics.exists():
            write_json(metrics,dict(protocol=spec.get('protocol',PROTOCOL),job_id=spec['job_id'],study=spec['study'],
                method=spec['method'],dataset=spec['dataset'],n=spec['n'],seed=spec['seed'],
                status=timeout_status or 'failed_no_metrics',failed_phase=progress.get('phase'),
                observed_process_s=time.monotonic()-started,exit_code=process.returncode))
        result=json.loads(metrics.read_text(encoding='utf-8'))
        write_json(out/'process.json',dict(command=command,exit_code=process.returncode,wall_s=time.monotonic()-started))
        print(f'[{result["status"]}] {spec["study"]} {spec["method"]} N={spec["n"]} seed={spec["seed"]}',flush=True)
        return result



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,epilog=(
        'Execution uses bundled NPZ adapters; SCOPE_GT_BASELINES_ROOT optionally overrides them. '
        'unmodified upstream repositories are not adapter replacements. '
        '--dry-run does not need external sources.'))
    p.add_argument('--data-base',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--datasets',nargs='+',type=canonical,choices=DATASETS,default=['lastfm_asia','minesweeper'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(DEFAULT_METHODS))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['native','scmb'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--device',default='cuda:0'); p.add_argument('--threads',type=int,default=6)
    p.add_argument('--warmup-epochs',type=int,default=3)
    p.add_argument('--epoch-timeout',type=float,default=0,help='train/validation watchdog seconds, 0 disables')
    p.add_argument('--preprocess-timeout',type=float,default=0,help='load/preprocessing watchdog seconds, 0 disables')
    p.add_argument('--job-timeout',type=float,default=0,help='whole-run watchdog seconds, 0 disables')
    p.add_argument('--accuracy-margin-pp',type=float,default=.5,help='predeclared descriptive accuracy-drop margin in percentage points')
    p.add_argument('--overrides',type=Path,help='JSON {all:{method:{...}},datasets:{dataset:{method:{...}}}}; identical in all arms')
    p.add_argument('--dataset-map',type=Path,help='JSON canonical dataset -> exact directory')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if (a.threads<1 or a.warmup_epochs<0 or any(s<0 for s in a.seeds) or
        min(a.epoch_timeout,a.preprocess_timeout,a.job_timeout,a.accuracy_margin_pp)<0 or
        not np.isfinite([a.epoch_timeout,a.preprocess_timeout,a.job_timeout,a.accuracy_margin_pp]).all()):
        p.error('Invalid seeds, counts, margin or time budgets')
    for name in ('datasets','methods','variants','seeds'):
        if len(set(getattr(a,name)))!=len(getattr(a,name)): p.error(f'Duplicate {name}')
    if not {'native','scmb'}.issubset(a.variants): p.error('Main experiment requires both native and scmb')
    overrides=json.loads(a.overrides.read_text(encoding='utf-8')) if a.overrides else {}
    roots=json.loads(a.dataset_map.read_text(encoding='utf-8')) if a.dataset_map else {}
    configs={f'{d}/{m}':config_for(m,d,{**overrides.get('all',{}).get(m,{}),
              **overrides.get('datasets',{}).get(d,{}).get(m,{})}) for d in a.datasets for m in a.methods}
    description=dict(protocol=PROTOCOL,datasets=a.datasets,methods=a.methods,variants=a.variants,seeds=a.seeds,
        jobs=len(a.datasets)*len(a.methods)*len(a.variants)*len(a.seeds),configurations=configs,
        warmup_epochs=a.warmup_epochs,accuracy_margin_pp=a.accuracy_margin_pp,
        adaptation='Fixed differentiable input memory; original attention domains; all SwapGT readout centers retained')
    if a.dry_run:
        print(json.dumps(description,ensure_ascii=False,indent=2)); return
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    description.update(source_sha256={m:sources(m) for m in a.methods},device=a.device,threads=a.threads,
        data_base=str(a.data_base.resolve()),dataset_map=roots,python=sys.executable,
        time_limits=dict(epoch=a.epoch_timeout,preprocess=a.preprocess_timeout,job=a.job_timeout))
    with lock(out/'.study.lock'):
        frozen=out/'study_identity.json'
        if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=description:
            raise ValueError('Frozen study/source/config changed; use a fresh --out')
        write_json(frozen,description)
        probe=subprocess.run([sys.executable,'-m','SCOPE_GT_transfer.worker','--probe',a.device],cwd=CODE,
            capture_output=True,text=True,encoding='utf-8',timeout=60)
        if probe.returncode: raise RuntimeError(probe.stderr or probe.stdout)
        runtime=json.loads(probe.stdout)
        specs=[]
        for dataset in a.datasets:
            root=Path(roots.get(dataset,a.data_base/dataset)).resolve()
            fingerprints=dataset_manifest(root)
            n=len(np.load(root/'label.npy',mmap_mode='r',allow_pickle=False))
            for method in a.methods:
                for seed in a.seeds:
                    order=a.variants[seed%len(a.variants):]+a.variants[:seed%len(a.variants)]
                    for variant in order:
                        spec=dict(protocol=PROTOCOL,study='transfer',method=method,variant=variant,dataset=dataset,
                            seed=seed,requested_seeds=a.seeds,n=n,config=configs[f'{dataset}/{method}'],
                            data_root=str(root),data_sha256=fingerprints,source_sha256=description['source_sha256'][method],
                            runtime=runtime,device=a.device,threads=a.threads,warmup_epochs=a.warmup_epochs,
                            accuracy_margin_pp=a.accuracy_margin_pp,time_limits=description['time_limits'])
                        spec['job_id']=identity(spec)
                        spec['directory']=str(out/dataset/method/f'seed{seed}'/variant)
                        specs.append(spec)
        write_json(out/'plan.json',specs)
        for spec in specs:
            launch(spec,sys.executable,a)
            summarize(out)
        rows=summarize(out)
        import csv
        with (out/'paired.csv').open(encoding='utf-8',newline='') as f:
            paired=list(csv.DictReader(f))
        if any(r['status']!='complete' for r in rows+paired):
            raise SystemExit('Incomplete runs; inspect failures.csv and per-run stdout.log')


if __name__=='__main__': main()
