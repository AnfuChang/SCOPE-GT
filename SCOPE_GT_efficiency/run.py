"""Run induced-graph cost curves and full-dataset convergence/resource summaries."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from SCOPE_GT_budgets.common import dataset_manifest, identity, lock, sha, write_json
from . import PROTOCOL
from .config import CODE, METHODS, FOLDERS, canonical, config_for, identity_note
from .prepare_subgraphs import DATASET_SIZES, prepare
from SCOPE_GT.external_backbones import baseline_path, source_key

HERE = Path(__file__).resolve().parent


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def source_manifest(method):
    files=[HERE/name for name in ['__init__.py','config.py','adapters.py','worker.py','run.py','prepare_subgraphs.py']]
    files += [CODE/'SCOPE_GT_budgets'/name for name in ['__init__.py','adapters.py','common.py','profiles.json']]
    files.append(CODE/'SCOPE_GT/external_backbones.py')
    folders = [CODE/FOLDERS[method]] if method == 'scope_gt' else [
        baseline_path(FOLDERS[method]), baseline_path('common')]
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            rel=p.relative_to(folder)
            if any(x in ('results','logs','figures','__pycache__','.git','checkpoints','.pytest_cache') for x in rel.parts):
                continue
            if p.is_file() and p.suffix in ('.py','.json','.sh') and not p.name.startswith('test_'):
                files.append(p)
    return {source_key(p):sha(p) for p in files}


def read_history(folder):
    result=[]
    p=Path(folder)/'history.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try: result.append(json.loads(line))
            except json.JSONDecodeError: break  # interrupted final write; keep preceding complete epochs
    return result


def launch(spec,python,args,worker_module='SCOPE_GT_efficiency.worker'):
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
        command=[python,'-u','-m',worker_module,'--spec',str(marker)]
        started=time.monotonic(); timeout_status=None; progress={}
        observation_limit=(spec.get('observation_seconds',0)
            if spec['study']=='convergence' and spec['dataset']=='amazon_computers' else 0)
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
                if observation_limit and now-started>observation_limit:
                    timeout_status=timeout_status or 'observation_limit'
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


def summarize(specs,results,out):
    per_seed=[]
    for spec,result in zip(specs,results):
        row={k:spec[k] for k in ['study','method','dataset','n','seed','job_id','directory']}
        row.update({k:v for k,v in result.items() if k not in ('traceback','runtime','time_to_targets') and not isinstance(v,(dict,list))})
        # A completed validation crossing remains observable even if a later phase times out.
        hist=read_history(spec['directory'])
        for target in spec.get('targets',[]):
            hits=[r['elapsed_val_s'] for r in hist if r.get('val_accuracy',-1)>=target]
            row[f'target_{target}_s']=hits[0] if hits else None
        per_seed.append(row)
    grouped=[]
    for key in dict.fromkeys((s['study'],s['dataset'],s['method'],s['n']) for s in specs):
        items=[r for r in per_seed if (r['study'],r['dataset'],r['method'],r['n'])==key]
        ok=[r for r in items if r['status']=='complete']
        expected=next(s.get('requested_seeds',[x['seed'] for x in items]) for s in specs
                      if (s['study'],s['dataset'],s['method'],s['n'])==key)
        complete={r['seed'] for r in ok}==set(expected)
        count=len(expected)
        row=dict(study=key[0],dataset=key[1],method=key[2],n=key[3],requested_runs=count,completed_runs=len(ok),
                 status='complete' if complete else 'incomplete',failures=';'.join(sorted({r['status'] for r in items if r['status']!='complete'})))
        fields=['epoch_seconds_mean','train_peak_allocated_bytes','selection_peak_allocated_bytes',
                'preprocess_s','test_seconds','test_accuracy','job_seconds','cpu_rss_peak_bytes']
        fields+=sorted({k for r in items for k in r if k.startswith('target_')})
        for field in fields:
            values=[r[field] for r in ok if r.get(field) is not None]
            if field.startswith('target_'):
                row[field+'_reached']=sum(r.get(field) is not None for r in items)
            row[field+'_mean']=float(np.mean(values)) if complete and len(values)==count else None
            row[field+'_std']=float(np.std(values,ddof=1)) if complete and len(values)==count and count>1 else None
        grouped.append(row)
    write_csv(out/'per_seed.csv',per_seed); write_csv(out/'summary.csv',grouped)
    write_json(out/'summary.json',grouped)
    write_csv(out/'failures.csv',[r for r in per_seed if r['status']!='complete'])
    return grouped


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,epilog=(
        'Backbones use the bundled Baselines tree; SCOPE_GT_BASELINES_ROOT overrides it. '
        'Use --methods scope_gt for SCOPE-GT alone.'))
    p.add_argument('study',choices=['all','scaling','pilot','convergence'])
    p.add_argument('--data-base',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--subgraphs',type=Path)
    p.add_argument('--scaling-dataset',type=canonical,choices=['pubmed','arxiv_year'],default='pubmed')
    p.add_argument('--scaling-dir',type=Path,help='exact source graph directory; default data-base/scaling-dataset')
    p.add_argument('--computers-dir',type=Path)
    p.add_argument('--convergence-dataset',type=canonical,choices=['amazon_computers','pokec','ogbn_products'],default='amazon_computers')
    p.add_argument('--convergence-dir',type=Path,help='exact dataset directory; default data-base/convergence-dataset')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4],help='formal training seeds (default: 0 1 2 3 4)')
    p.add_argument('--pilot-seeds',nargs='+',type=int,default=[100,101,102,103,104],
                   help='independent pilot seeds, disjoint from formal runs (default: 100 101 102 103 104)')
    p.add_argument('--sizes',nargs='+',help='node counts or full; defaults depend on scaling dataset')
    p.add_argument('--graph-seed',type=int,default=2027)
    p.add_argument('--warmup-epochs',type=int,default=3)
    p.add_argument('--measured-epochs',type=int,default=10)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--epoch-timeout',type=float,default=0,help='train/validation watchdog seconds; 0 disables (default)')
    p.add_argument('--preprocess-timeout',type=float,default=0,help='load/preparation watchdog seconds; 0 disables (default)')
    p.add_argument('--job-timeout',type=float,default=0,help='whole-job watchdog seconds; 0 disables (default)')
    p.add_argument('--observation-seconds',type=float,default=7200,
                   help='Computers convergence observation window from subprocess launch; 0 disables; does not limit other datasets/studies')
    p.add_argument('--python-map',type=Path,help='JSON method -> Python executable, optional per-method environments')
    p.add_argument('--overrides',type=Path,help='JSON {all:{method:{...}},datasets:{dataset:{method:{...}}}}')
    p.add_argument('--targets-json',type=Path)
    target_flags=p.add_mutually_exclusive_group()
    target_flags.add_argument('--without-targets',action='store_true',help='explicitly skip pilot targets (the default)')
    target_flags.add_argument('--with-targets',action='store_true',help='opt in to pilot/time-to-target comparison')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.computers_dir and (args.convergence_dataset!='amazon_computers' or args.convergence_dir):
        p.error('--computers-dir is only a legacy alias for the Computers convergence directory')
    if args.without_targets and (args.study=='pilot' or args.targets_json):
        p.error('--without-targets cannot be combined with pilot or --targets-json')
    args.use_targets=bool(args.with_targets or args.targets_json)
    args.sizes=args.sizes or list(DATASET_SIZES[args.scaling_dataset])
    if (args.warmup_epochs<0 or min(args.measured_epochs,args.threads)<1
            or any(x<0 for x in args.seeds+args.pilot_seeds)
            or len(set(args.methods))!=len(args.methods) or len(set(args.seeds))!=len(args.seeds)
            or len(set(args.pilot_seeds))!=len(args.pilot_seeds) or set(args.seeds)&set(args.pilot_seeds)
            or min(args.epoch_timeout,args.preprocess_timeout,args.job_timeout,args.observation_seconds)<0):
        p.error('invalid counts, duplicate methods/seeds, overlapping pilot seeds, or negative time limits')
    overrides=json.loads(args.overrides.read_text(encoding='utf-8')) if args.overrides else {}
    py_map=json.loads(args.python_map.read_text(encoding='utf-8')) if args.python_map else {}
    out=args.out.resolve()
    groups=(['scaling']+(['pilot'] if args.use_targets else [])+['convergence']) if args.study=='all' else [args.study]
    configs={}
    datasets=list(dict.fromkeys(([args.scaling_dataset] if 'scaling' in groups else []) +
                               ([args.convergence_dataset] if any(s!='scaling' for s in groups) else [])))
    for dataset in datasets:
        for method in args.methods:
            update={**overrides.get('all',{}).get(method,{}),**overrides.get('datasets',{}).get(dataset,{}).get(method,{})}
            configs[dataset,method]=config_for(method,dataset,update,scaling=dataset==args.scaling_dataset)
    if args.dry_run:
        print(json.dumps(dict(protocol=PROTOCOL,studies=groups,methods=args.methods,
            scaling_dataset=args.scaling_dataset,scaling_sizes=args.sizes,convergence_dataset=args.convergence_dataset,
            uses_targets=args.use_targets,
            jobs=len(args.methods)*(len(args.sizes)*('scaling' in groups)*len(args.seeds)+
                                   ('pilot' in groups)*len(args.pilot_seeds)+('convergence' in groups)*len(args.seeds)),
            training_seeds=args.seeds,pilot_seeds=args.pilot_seeds,
            warmup_epochs=args.warmup_epochs,measured_epochs=args.measured_epochs,
            watchdog_seconds=dict(epoch=args.epoch_timeout,preprocess=args.preprocess_timeout,job=args.job_timeout),
            observation_seconds=(args.observation_seconds
                if args.convergence_dataset=='amazon_computers' and 'convergence' in groups else 0),
            configurations={f'{d}/{m}':cfg for (d,m),cfg in configs.items()}),indent=2))
        return
    out.mkdir(parents=True,exist_ok=True)
    sources={m:source_manifest(m) for m in args.methods}
    binding=dict(protocol=PROTOCOL,methods=args.methods,configurations={f'{d}/{m}':c for (d,m),c in configs.items()},
        source_sha256=sources,seeds=args.seeds,pilot_seeds=args.pilot_seeds,sizes=args.sizes,graph_seed=args.graph_seed,
        device=args.device,threads=args.threads,warmup_epochs=args.warmup_epochs,measured_epochs=args.measured_epochs,
        time_limits=[args.epoch_timeout,args.preprocess_timeout,args.job_timeout],
        observation_seconds=args.observation_seconds,python_map=py_map,
        data_base=str(args.data_base.resolve()),computers_dir=str(args.computers_dir.resolve()) if args.computers_dir else None,
        convergence_dataset=args.convergence_dataset,convergence_dir=str(args.convergence_dir.resolve()) if args.convergence_dir else None,
        scaling_dataset=args.scaling_dataset,scaling_dir=str(args.scaling_dir.resolve()) if args.scaling_dir else None,
        subgraphs=str(args.subgraphs.resolve()) if args.subgraphs else None,use_targets=args.use_targets)
    binding_file=out/'study_identity.json'
    if binding_file.exists() and json.loads(binding_file.read_text(encoding='utf-8'))!=binding:
        raise ValueError('Frozen study identity changed; use a fresh --out rather than combining versions')
    write_json(binding_file,binding)
    specs,results,runtimes,data_cache=[],[],{},{}
    for study in groups:
        if study=='scaling':
            dataset=args.scaling_dataset
            source=args.scaling_dir or args.data_base/dataset
            if not source.exists() and dataset=='arxiv_year' and (args.data_base/'arxiv-year').exists():
                source=args.data_base/'arxiv-year'
            graph_items=prepare(source,args.subgraphs or out/f'{dataset}_subgraphs',args.sizes,args.graph_seed)
        else:
            dataset=args.convergence_dataset
            root=args.convergence_dir or args.computers_dir or args.data_base/dataset
            if not root.exists() and dataset=='amazon_computers' and (args.data_base/'computers').exists():
                root=args.data_base/'computers'
            if not root.exists() and dataset=='ogbn_products' and (args.data_base/'products').exists():
                root=args.data_base/'products'
            n=len(np.load(root/'label.npy',mmap_mode='r',allow_pickle=False))
            graph_items=[dict(n=n,root=str(root.resolve()))]
        target_file=args.targets_json or out/'targets.json'
        targets=[]
        if study=='convergence' and args.use_targets:
            if not target_file.exists():
                raise ValueError('Run pilot first to freeze targets.json, or provide --targets-json')
            target_spec=json.loads(target_file.read_text(encoding='utf-8'))
            targets=target_spec['targets']
            if (target_spec['dataset']!=dataset or target_spec['data_sha256']!=dataset_manifest(Path(graph_items[0]['root']))
                    or set(target_spec['methods'])!=set(args.methods)
                    or target_spec['configurations']!={m:configs[dataset,m] for m in args.methods}):
                raise ValueError('Targets must match the same dataset, method set and frozen configurations')
        seeds=args.pilot_seeds if study=='pilot' else args.seeds
        study_specs,study_results=[],[]
        for item in graph_items:
            root=Path(item['root'])
            if str(root) not in data_cache:
                data_cache[str(root)]=dataset_manifest(root)
            for seed in seeds:
                ordered=args.methods[seed%len(args.methods):]+args.methods[:seed%len(args.methods)]
                for method in ordered:
                    python=py_map.get(method,sys.executable)
                    runtime_key=(python,args.device)
                    if runtime_key not in runtimes:
                        probe=subprocess.run([python,'-m','SCOPE_GT_efficiency.worker','--probe',args.device],
                            cwd=CODE,capture_output=True,text=True,encoding='utf-8',timeout=60)
                        if probe.returncode:
                            raise RuntimeError(probe.stderr or probe.stdout)
                        runtimes[runtime_key]=json.loads(probe.stdout)
                    spec=dict(protocol=PROTOCOL,study=study,method=method,dataset=dataset,n=item['n'],seed=seed,
                        requested_seeds=seeds,
                        config=configs[dataset,method],data_root=str(root),data_sha256=data_cache[str(root)],
                        source_sha256=sources[method],runtime=runtimes[runtime_key],device=args.device,threads=args.threads,
                        warmup_epochs=args.warmup_epochs,measured_epochs=args.measured_epochs,targets=targets,
                        observation_seconds=(args.observation_seconds
                            if study=='convergence' and dataset=='amazon_computers' else 0),
                        time_limits=dict(epoch=args.epoch_timeout,preprocess=args.preprocess_timeout,job=args.job_timeout))
                    spec['job_id']=identity(spec)
                    folder=out/(f'{dataset}_scaling' if study=='scaling' else ('computers' if dataset=='amazon_computers' else dataset)+'_'+study)
                    if study=='scaling': folder=folder/f'N{item["n"]}'
                    spec['directory']=str(folder/method/f'seed{seed}')
                    write_json(out/'plan.json',specs+[spec])
                    result=launch(spec,python,args)
                    specs.append(spec); results.append(result); study_specs.append(spec); study_results.append(result)
                    summarize(specs,results,out)
        if study=='pilot':
            if any(r['status']!='complete' for r in study_results):
                raise RuntimeError('Incomplete pilot; cannot select targets from a subset of the methods')
            vals=[r['best_val_accuracy'] for r in study_results]
            frozen=dict(dataset=dataset,methods=args.methods,data_sha256=data_cache[str(root)],
                configurations={m:configs[dataset,m] for m in args.methods},pilot_seeds=seeds,
                targets=sorted(set([max(0.,float(np.floor((min(vals)-.01)*1000)/1000)),
                                   float(np.floor(np.median(vals)*1000)/1000)])),
                rule='common: worst pilot best minus one percentage point; high: median pilot best, rounded down to 0.1 points',
                pilot_job_ids=[s['job_id'] for s in study_specs])
            if target_file.exists() and json.loads(target_file.read_text(encoding='utf-8'))!=frozen:
                raise ValueError('Existing frozen targets differ; use a fresh output root')
            write_json(target_file,frozen)
    # Include all matching runs already in this output root when a stage is run separately.
    for marker in sorted(out.rglob('spec.json')):
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old.get('protocol')!=PROTOCOL or old['job_id'] in {s['job_id'] for s in specs}:
            continue
        met=marker.with_name('metrics.json')
        if met.exists():
            specs.append(old); results.append(json.loads(met.read_text(encoding='utf-8')))
    write_json(out/'plan.json',specs)
    summarize(specs,results,out)
    if any(r['status']!='complete' for r in results):
        raise SystemExit('Some runs failed or exceeded a budget; see failures.csv and per-run logs')


if __name__=='__main__':
    main()
