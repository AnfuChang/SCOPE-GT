"""Export measured accuracy/cost and paired deltas; never fill absent GPU data."""
import argparse
import json
from pathlib import Path
import numpy as np
from SCOPE_GT_efficiency.run import write_csv,read_history
from SCOPE_GT_budgets.common import write_json

FIELDS=('test_accuracy','epoch_seconds_mean','train_peak_allocated_bytes','complete_gpu_peak_allocated_bytes',
        'test_gpu_peak_allocated_bytes','job_seconds','test_seconds','test_centers_per_second',
        'preprocess_s','validation_total_seconds','time_to_best_validation_s','parameters','epochs_run')


def moments(row,name,values,expected):
    values=[v for v in values if v is not None]
    row[name+'_mean']=float(np.mean(values)) if len(values)==expected else None
    row[name+'_sd']=float(np.std(values,ddof=1)) if len(values)==expected and expected>1 else None


def summarize(out):
    out=Path(out); specs=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    rows=[]; mapping={}
    for s in specs:
        folder=Path(s['directory']); path=folder/'metrics.json'
        r=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(status='pending')
        row={k:s[k] for k in ('dataset','method','variant','seed','job_id','directory')}
        if path.exists() and r.get('job_id')!=s['job_id']: raise ValueError(f'Metrics identity mismatch: {path}')
        row.update({k:v for k,v in r.items() if not isinstance(v,(dict,list)) and k!='traceback'})
        mapping[s['dataset'],s['method'],s['seed'],s['variant']]=(s,r)
        rows.append(row)
    grouped=[]
    for d,m,v in dict.fromkeys((s['dataset'],s['method'],s['variant']) for s in specs):
        items=[r for r in rows if (r['dataset'],r['method'],r['variant'])==(d,m,v)]
        good=[r for r in items if r['status']=='complete']
        expected=len(items)
        row=dict(dataset=d,method=m,variant=v,metric='roc_auc' if d=='minesweeper' else 'accuracy',requested_runs=expected,completed_runs=len(good),
                 status='complete' if len(good)==expected else 'incomplete')
        for field in FIELDS: moments(row,field,[r.get(field) for r in good],expected)
        grouped.append(row)
    pairs=[]
    for d,m,seed in dict.fromkeys((s['dataset'],s['method'],s['seed']) for s in specs):
        s,native=mapping[d,m,seed,'native']
        for variant in ('scmb','center_padded'):
            if (d,m,seed,variant) not in mapping: continue
            target_s,target=mapping[d,m,seed,variant]
            pair=dict(dataset=d,method=m,seed=seed,variant=variant,status='incomplete')
            if native.get('status')=='complete' and target.get('status')=='complete':
                hn={r['epoch']:r for r in read_history(s['directory'])}
                ht={r['epoch']:r for r in read_history(target_s['directory'])}
                common=sorted(set(hn)&set(ht))
                matched=(native['initial_state_sha256']==target['initial_state_sha256'] and
                    native['parameters']==target['parameters'] and s['config']==target_s['config'] and
                    s['data_sha256']==target_s['data_sha256'] and
                    all(hn[e]['input_sha256']==ht[e]['input_sha256'] for e in common))
                pair.update(status='complete' if matched and common else 'pair_validation_failed',
                            matched_epochs=len(common),paired_inputs_and_initialization=bool(matched and common))
                if pair['status']=='complete':
                    delta=100*(target['test_accuracy']-native['test_accuracy'])
                    pair.update(accuracy_delta_pp=delta,accuracy_drop_within_margin=delta>=-s['accuracy_margin_pp'],
                        accuracy_margin_pp=s['accuracy_margin_pp'],
                        epoch_speedup=native['epoch_seconds_mean']/target['epoch_seconds_mean']
                            if native.get('epoch_seconds_mean') is not None and target.get('epoch_seconds_mean') else None,
                        job_speedup=native['job_seconds']/target['job_seconds'],
                        inference_speedup=native['test_seconds']/target['test_seconds'],
                        gpu_reduction_percent=100*(1-target['train_peak_allocated_bytes']/native['train_peak_allocated_bytes'])
                            if native.get('train_peak_allocated_bytes') and target.get('train_peak_allocated_bytes') is not None else None)
                    timing=[e for e in common if not hn[e]['warmup_epoch'] and not ht[e]['warmup_epoch']]
                    pair['matched_measured_epochs']=len(timing)
                    pair['matched_epoch_speedup']=(sum(hn[e]['train']['seconds'] for e in timing)/
                        sum(ht[e]['train']['seconds'] for e in timing)) if timing else None
            pairs.append(pair)
    pair_summary=[]
    for d,m,v in dict.fromkeys((r['dataset'],r['method'],r['variant']) for r in pairs):
        items=[r for r in pairs if (r['dataset'],r['method'],r['variant'])==(d,m,v)]
        good=[r for r in items if r['status']=='complete']; n=len(items)
        row=dict(dataset=d,method=m,variant=v,requested_pairs=n,completed_pairs=len(good),
                 status='complete' if len(good)==n else 'incomplete')
        for field in ('accuracy_delta_pp','epoch_speedup','matched_epoch_speedup','job_speedup','inference_speedup','gpu_reduction_percent'):
            moments(row,field,[r.get(field) for r in good],n)
        row['interpretation']='single-seed descriptive comparison' if n==1 else 'paired runs; margin check is descriptive, not a statistical equivalence test'
        pair_summary.append(row)
    write_csv(out/'per_run.csv',rows); write_csv(out/'summary.csv',grouped); write_json(out/'summary.json',grouped)
    write_csv(out/'paired.csv',pairs); write_csv(out/'paired_summary.csv',pair_summary)
    write_csv(out/'failures.csv',[r for r in rows if r['status'] not in ('complete','pending')]+[r for r in pairs if r['status']=='pair_validation_failed'])
    def cell(r,field,scale=1.):
        x=r.get(field+'_mean'); sd=r.get(field+'_sd')
        if x is None: return '--'
        return f'{x*scale:.2f}' if sd is None else f'{x*scale:.2f} $\\pm$ {sd*scale:.2f}'
    tex=['% Measured values only. CPU runs have -- in GPU columns. Units: %, seconds, MiB.',
         r'\begin{tabular}{lllrrrr}',r'\toprule',
         r'Dataset & Backbone & Variant & Score & Epoch (s) & GPU (MiB) & Job (s) \\',r'\midrule']
    names={'native':'Native','scmb':'+SCMB','center_padded':'Center-query padded'}
    for r in grouped:
        values=[r['dataset'].replace('_',r'\_'),r['method'].upper(),names[r['variant']],
                cell(r,'test_accuracy',100),cell(r,'epoch_seconds_mean'),cell(r,'train_peak_allocated_bytes',1/2**20),cell(r,'job_seconds')]
        tex.append(' & '.join(values)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'transfer_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    write_json(out/'report_metadata.json',dict(gpu_unit='MiB (2^20 bytes)',accuracy_unit='fraction in CSV; percent in LaTeX',
        across_seed_sd='sample SD; absent for one seed',epoch_scope='complete training epoch; excludes first warmup epochs',
        complete_job='load through prediction export',noninferiority='descriptive margin, not statistical proof'))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--results',type=Path,required=True)
    summarize(p.parse_args().results)
