"""Robustness, baseline and scaling reports using shared metric definitions."""

from experiments.shared.paths import observed_report, OBSERVED_RUNS
import argparse
import numpy as np
import pandas as pd
from experiments.shared.runtime import fresh
from experiments.model_accuracy.in_prior import fragments
from experiments.model_accuracy.in_prior import long_frame
from experiments.model_accuracy.in_prior import tables
from experiments.model_accuracy.in_prior import per_dgp
from experiments.model_accuracy.in_prior import prepare
from experiments.model_accuracy.in_prior import from_totals
from experiments.model_accuracy.in_prior import COLS
from experiments.model_accuracy.in_prior import summarize
from experiments.shared.evaluation import gmm_arrays,gmm_quantiles


def e9(smoke):
    name='e9_smoke' if smoke else 'e9';wide=fragments(name)
    out=fresh(observed_report('e9', smoke))
    rng=np.random.default_rng(20260797)
    summaries=[]
    for family in ['oscillatory_outcome','heavy_tailed_outcome','extreme_overlap','gamma_extrapolation']:
        w=wide[wide.family.eq(family)];long=long_frame(w,'E9_'+family)
        n=w.dataset_id.nunique();draws={side:rng.integers(0,n,size=(2000,n)) for side in ['lower','upper']}
        dest=fresh(out/family);summary=tables(long,dest,20260797,draws);summary['family']=family;summaries.append(summary)
    pd.concat(summaries).to_csv(out/'summary.csv',index=False)

def e7(study,smoke):
    name='e7_'+study+('_smoke' if smoke else '');wide=fragments(name)
    out=fresh(observed_report('e7_'+study, smoke))
    field='context_length' if study=='context' else 'covariate_dimension'
    long=long_frame(wide,'E7_'+study);summaries=[];pdgps=[]
    for condition,g in long.groupby(field):
        dest=fresh(out/str(condition));s=tables(g,dest,20311726 if study=='context' else 20311727,detail_only=True)
        s[field]=condition;summaries.append(s);d=per_dgp(g);d[field]=condition;pdgps.append(d)
    pd.concat(summaries).to_csv(out/'summary.csv',index=False)
    pd.concat(pdgps).to_csv(out/'per_dgp.csv',index=False)
    if study=='dimension':return  # Different DGP families; no paired cross-dimension contrast.
    baseline=1024
    records=[];ids=np.sort(long.dataset_id.unique())
    draws=np.random.default_rng(20311726).integers(0,len(ids),size=(2000,len(ids)))
    for (condition,head),g in long.groupby([field,'head']):
        b=long[(long[field]==baseline)&(long['head']==head)]
        sums=prepare(g).groupby('dataset_id')[COLS].sum().reindex(ids).to_numpy();base=prepare(b).groupby('dataset_id')[COLS].sum().reindex(ids).to_numpy()
        bm=from_totals(sums[draws].sum(axis=1));bb=from_totals(base[draws].sum(axis=1));point=from_totals(sums.sum(axis=0));bp=from_totals(base.sum(axis=0))
        for metric in ['rmse','coverage_90']:
            d=bm[metric]-bb[metric];records.append(dict(condition=condition,baseline=baseline,head=head,metric=metric,delta=point[metric]-bp[metric],ci_low=np.quantile(d,.025),ci_high=np.quantile(d,.975)))
    pd.DataFrame(records).to_csv(out/'paired_condition_differences.csv',index=False)

def e3_long(wide):
    keys=['dataset_id','source_row_index','a_query','sensitivity_model','gamma']
    ref=wide[wide.method.eq('pifm')].pivot(index=keys,columns='bound_type',values='theta_reference')
    ref['width']=ref.upper-ref.lower
    w=wide.merge(ref[['width']].reset_index(),on=keys,validate='many_to_one')
    w=w[w.reference_available].copy()
    outputs=[]
    for (sensitivity,method,head),g in w.groupby(['sensitivity_model','method','bound_type']):
        f=pd.DataFrame(dict(dataset_id=g.dataset_id,source_row_index=g.source_row_index,a_query=g.a_query,gamma=g.gamma,gamma_group=g.gamma,head=head,cohort='E3_'+sensitivity+'_'+method,method=method,sensitivity_model=sensitivity,truth=g.theta_reference,prediction=g.theta_prediction,sigma_obs=g.sigma_obs,width=g.width,counterfactual_weight=g.counterfactual_weight))
        if method=='pifm':
            quant=gmm_quantiles(*gmm_arrays(g,head),[.05,.95]);f['q05']=quant[:,0];f['q95']=quant[:,1]
        else:f['q05']=np.nan;f['q95']=np.nan
        outputs.append(f)
    return pd.concat(outputs,ignore_index=True)

def neuralcsa_diagnostics(raw, out):
    stage=pd.read_csv(out/'stage1_metrics.csv').sort_values('dataset_id')
    ids=np.sort(raw.dataset_id.unique());draws=np.random.default_rng(20333726).integers(0,len(ids),size=(2000,len(ids)))
    rows=[]
    for metric in ['validation_nll','validation_pit_ks','validation_pit_pvalue']:
        values=stage[metric].to_numpy();samples=values[draws].mean(axis=1)
        rows.append(dict(metric=metric,mean=values.mean(),ci_low=np.quantile(samples,.025),ci_high=np.quantile(samples,.975),bootstrap_seed=20333726))
    pd.DataFrame(rows).to_csv(out/'stage1_summary.csv',index=False)
    neural=raw[raw.method.ne('pifm')]
    group=['sensitivity_model','method','bound_type','gamma']
    summary=neural.groupby(group).agg(n_rows=('gamma','size'),n_dgps=('dataset_id','nunique'),feasible_rate=('feasible','mean'),target_attained_rate=('target_attained','mean'),materially_undersaturated_rate=('materially_undersaturated','mean'),mean_constraint_violation=('constraint_violation','mean'))
    summary.to_csv(out/'constraint_summary.csv')
    supported=neural[neural.reference_available]
    keys=['dataset_id','source_row_index','a_query','sensitivity_model','bound_type','gamma']
    full=supported[supported.method.eq('neuralcsa_full')];oracle=supported[supported.method.eq('neuralcsa_oracle_stage')]
    paired=full[keys+['theta_prediction']].merge(oracle[keys+['theta_prediction']],on=keys,validate='one_to_one',suffixes=('_full','_oracle'))
    paired['gap']=(paired.theta_prediction_full-paired.theta_prediction_oracle).abs();rows=[]
    for (model,head),g in paired.groupby(['sensitivity_model','bound_type']):
        totals=g.groupby('dataset_id').agg(gap=('gap','sum'),n=('gap','size')).reindex(ids,fill_value=0).to_numpy()
        sums=totals[draws].sum(axis=1);samples=sums[:,0]/sums[:,1]
        rows.append(dict(sensitivity_model=model,head=head,mean_abs_full_minus_oracle=totals[:,0].sum()/totals[:,1].sum(),ci_low=np.quantile(samples,.025),ci_high=np.quantile(samples,.975)))
    pd.DataFrame(rows).to_csv(out/'full_minus_oracle_summary.csv',index=False)


def e3(smoke):
    name='e3_smoke' if smoke else 'e3';wide=fragments(name);long=e3_long(wide)
    out=fresh(observed_report('e3', smoke))
    summaries=[];dgps=[]
    for (sensitivity,method),g in long.groupby(['sensitivity_model','method']):
        dest=fresh(out/(sensitivity+'_'+method));s=tables(g,dest,20333726,detail_only=True)
        s['sensitivity_model']=sensitivity;s['method']=method
        if method!='pifm':
            # NeuralCSA emits point estimates; posterior coverage is not defined.
            for file in dest.glob('*.csv'):
                table=pd.read_csv(file)
                for c in table:
                    if c.startswith('coverage_90'):table[c]=np.nan
                table.to_csv(file,index=False)
            for c in s:
                if c.startswith('coverage_90'):s[c]=np.nan
        summaries.append(s);d=per_dgp(g);d['sensitivity_model']=sensitivity;d['method']=method;dgps.append(d)
    pd.concat(summaries).to_csv(out/'summary.csv',index=False)
    pd.concat(dgps).to_csv(out/'per_dgp.csv',index=False)
    ids=np.sort(wide.dataset_id.unique());draws=np.random.default_rng(20333726).integers(0,len(ids),size=(2000,len(ids)))
    differences_rows=[];pooled=[]
    for sensitivity,sl in long.groupby('sensitivity_model'):
        for method,g in sl.groupby('method'):
            pooled.append(summarize(g,ids,draws,sensitivity_model=sensitivity,method=method,head='both'))
        for head in ['lower','upper','both']:
            local=sl if head=='both' else sl[sl['head'].eq(head)]
            values={};point={}
            for method,g in local.groupby('method'):
                total=prepare(g).groupby('dataset_id')[COLS].sum().reindex(ids,fill_value=0).to_numpy()
                values[method]=from_totals(total[draws].sum(axis=1));point[method]=from_totals(total.sum(axis=0))
            for baseline in ['neuralcsa_full','neuralcsa_oracle_stage']:
                for metric in ['rmse','rmse_original','mae']:
                    d=values['pifm'][metric]-values[baseline][metric]
                    differences_rows.append(dict(sensitivity_model=sensitivity,head=head,comparison='pifm_minus_'+baseline,metric=metric,delta=point['pifm'][metric]-point[baseline][metric],ci_low=np.quantile(d,.025),ci_high=np.quantile(d,.975)))
    pd.DataFrame(pooled).to_csv(out/'pooled_heads_summary.csv',index=False)
    pd.DataFrame(differences_rows).to_csv(out/'paired_method_differences.csv',index=False)
    # Preserve every method's reference support and constraint diagnostics for plotting.
    cols=['dataset_id','source_row_index','a_query','sensitivity_model','method','bound_type','gamma','theta_prediction','theta_reference','reference_available','gamma_realized','constraint_violation','undersaturation','materially_undersaturated','sigma_obs']
    wide[cols].to_csv(out/'accuracy_rows.csv',index=False)
    pd.read_csv(OBSERVED_RUNS/name/'stage1_metrics.csv').to_csv(out/'stage1_metrics.csv',index=False)
    neuralcsa_diagnostics(wide, out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['e9','e3','e7_context','e7_dimension'])
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.stage.startswith('e7_'):e7(a.stage.split('_')[1],a.smoke)
    else:{'e9':e9,'e3':e3}[a.stage](a.smoke)
