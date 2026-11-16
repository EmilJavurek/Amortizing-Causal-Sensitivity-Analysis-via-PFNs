"""In-prior MSM/KL inference and calibration on factual-context outcomes."""
from __future__ import annotations

from experiments.shared.paths import OBSERVED_RUNS, observed_report
import argparse
import time
import numpy as np
import pandas as pd
import torch
from experiments.shared.observed import observed_run
from experiments.shared.runtime import read, sha, fresh
from experiments.shared.inputs import cohort_ids, input_path, normalized_root, overlap_edges, kl_reference, cohort_artifacts
from experiments.shared.evaluation import (normalized_dataset, make_query_grid, load_pifm, pifm_predict_frame,
    add_gmm_diagnostics, gmm_arrays, gmm_cdf, MSM_CHECKPOINT, KL_CHECKPOINT)

def affine(i, values, root, source_root):
    old=read(source_root/'scalers'/f'scaler_{i}.json');new=read(root/'scalers'/f'scaler_{i}.json')
    return (old['y_std']*values+old['y_mean']-new['y_mean'])/new['y_std']

def evaluate_e2(smoke):
    run=observed_run('e2_smoke' if smoke else 'e2',smoke)
    root=normalized_root()
    artifacts=cohort_artifacts('msm')
    import experiments.shared.msm_reference as msm_reference
    ids=cohort_ids('msm')[:2] if smoke else cohort_ids('msm')
    model,_=load_pifm(MSM_CHECKPOINT,torch.device('cuda:0'))
    fresh(run.path/'fragments')
    try:
        for pos,i in enumerate(ids):

            frame,xcols=normalized_dataset(artifacts[i])
            selected=msm_reference.propensity_stratified_row_indices(frame,8,msm_reference.QUANTILES)
            query=make_query_grid(frame,xcols,selected,msm_reference.GAMMA_GRID)
            base=query.drop_duplicates(['source_row_index','a_query'])
            outcome=msm_reference.load_outcome_map(artifacts[i],torch.device('cuda:0'))
            latent=msm_reference.seeded_normal((len(base),16384),20260726+1000*pos+i,torch.device('cuda:0'))
            with torch.no_grad():
                samples=outcome.evaluate(torch.tensor(base[xcols].to_numpy(),dtype=torch.float32,device='cuda:0'),
                       torch.tensor(base.a_query.to_numpy(),dtype=torch.float32,device='cuda:0'),latent).cpu().numpy()
            low,up=msm_reference.msm_reference_from_samples(samples,base.pi_query.to_numpy(),msm_reference.GAMMA_GRID)
            query['theta_lower']=low.ravel(); query['theta_upper']=up.ravel()
            pred=pifm_predict_frame(model,frame,xcols,query,torch.device('cuda:0'))
            pred=add_gmm_diagnostics(pred,'theta_upper')
            pred['lower_pit']=gmm_cdf(pred.theta_lower.to_numpy(),*gmm_arrays(pred,'lower'))
            pred['dataset_id']=i
            pred['sigma_obs']=read(root/'scalers'/f'scaler_{i}.json')['y_std']
            pred.to_parquet(run.path/'fragments'/f'dgp_{i}.parquet',index=False)
            run.manifest['completed_ids'].append(i)
            if pos%16==0: run.save();print(f'E2 {pos+1}/{len(ids)} elapsed={time.time()-run.started:.1f}s',flush=True)
        run.manifest.update(checkpoint_sha256=sha(MSM_CHECKPOINT),oracle_samples=16384,seed=20260726)
        run.done()
    except BaseException:
        run.done('failed');raise

def evaluate_e1(smoke):
    run=observed_run('e1_smoke' if smoke else 'e1',smoke)
    root=normalized_root()
    ids=cohort_ids('kl')[:2] if smoke else cohort_ids('kl')
    artifacts=cohort_artifacts('kl',ids)
    references=kl_reference(smoke)
    reference_root=input_path('kl_reference_root')
    if smoke: reference_root=reference_root.with_name(reference_root.name+'_smoke')
    model,_=load_pifm(KL_CHECKPOINT,torch.device('cuda:0'))
    fresh(run.path/'fragments')
    try:
        for i in ids:
            frame,xcols=normalized_dataset(artifacts[i])
            ref=references[references.dataset_id.eq(i)].copy()
            ref['theta_reference']=affine(i,ref.theta_reference,root,reference_root)
            keys=['dataset_id','source_row_index','a_query','bound_type','lambda_index','gamma_reference','theta_reference','p_a1','pi_query','counterfactual_weight']+xcols
            query=ref[keys].copy();query['gamma']=query.gamma_reference
            np.testing.assert_allclose(query[xcols],frame.iloc[query.source_row_index.to_numpy()][xcols],atol=1e-7,rtol=1e-7)
            pred=pifm_predict_frame(model,frame,xcols,query,torch.device('cuda:0'))
            pred=add_gmm_diagnostics(pred,'theta_reference')
            pred['sigma_obs']=read(root/'scalers'/f'scaler_{i}.json')['y_std']
            pred.to_parquet(run.path/'fragments'/f'dgp_{i}.parquet',index=False)
            run.manifest['completed_ids'].append(i)
        run.manifest.update(checkpoint_sha256=sha(KL_CHECKPOINT),reference_sha256=sha(reference_root/'reference_selected.parquet'))
        run.done()
    except BaseException:
        run.done('failed');raise

"""In-prior metrics, DGP bootstrap, examples and structural projection.

RMSE pools row squared errors. Original-unit residuals multiply by each DGP's sigma_obs.
Relative error uses the same-query two-sided reference width, never posterior width.
Zero widths have undefined row ratios and are counted, not replaced by an arbitrary floor.
"""


def fragments(name):
    assert read(OBSERVED_RUNS/name/'manifest.json')['status']=='completed'
    return pd.concat([pd.read_parquet(p) for p in sorted((OBSERVED_RUNS/name/'fragments').glob('*.parquet'))],ignore_index=True)

def curve_interp(frame, gamma, side):
    curve=frame.groupby('gamma_reference').theta_reference.agg('max' if side=='upper' else 'min').sort_index()
    y=np.maximum.accumulate(curve.values) if side=='upper' else np.minimum.accumulate(curve.values)
    return np.interp(gamma,curve.index.to_numpy(),y,left=np.nan,right=np.nan)

def long_frame(wide,cohort,kl=False):
    outputs=[]
    for side in ['lower','upper']:
        w=wide[wide.bound_type.eq(side)].copy() if kl else wide.copy()
        r=pd.DataFrame(dict(dataset_id=w.dataset_id,head=side,gamma=w.gamma,source_row_index=w.source_row_index,a_query=w.a_query,
            truth=w.theta_reference if kl else w[f'theta_{side}'],prediction=w[f'{side}_mean'],sigma_obs=w.sigma_obs,
            q05=w[f'{side}_q05'],q95=w[f'{side}_q95'],counterfactual_weight=w.counterfactual_weight))
        if kl:
            widths=[]
            for key,g in w.groupby(['dataset_id','source_row_index','a_query'],sort=False):
                i,row,arm=key
                other=wide[(wide.dataset_id==i)&(wide.source_row_index==row)&(wide.a_query==arm)&(wide.bound_type!=side)]
                opposite=curve_interp(other,g.gamma.to_numpy(),'upper' if side=='lower' else 'lower')
                width=opposite-g.theta_reference.to_numpy() if side=='lower' else g.theta_reference.to_numpy()-opposite
                widths.append(pd.Series(width,index=g.index))
            r['width']=pd.concat(widths).reindex(w.index)
            r['gamma_group']=pd.qcut(w.gamma_reference,5,labels=False,duplicates='drop').to_numpy()
        else:
            r['width']=w.theta_upper-w.theta_lower;r['gamma_group']=w.gamma
        for c in ['family','context_length','covariate_dimension','replicate_index','lambda_index']:
            if c in w:r[c]=w[c]
        r['cohort']=cohort;outputs.append(r)
    return pd.concat(outputs,ignore_index=True)

def prepare(frame):
    f=frame.copy();e=f.prediction-f.truth
    f['n']=1.;f['se']=e**2;f['se_original']=(e*f.sigma_obs)**2
    f['covered']=((f.truth>=f.q05)&(f.truth<=f.q95)).astype(float) if 'q05' in f else np.nan
    f['ae']=e.abs();f['bias_sum']=e
    valid=f.width.notna()&f.width.ge(0)
    positive=f.width.gt(1e-12)&valid
    f['width_sum']=f.width.where(valid,0);f['width_ae']=f.ae.where(valid,0)
    f['ratio']=np.where(positive,f.ae/f.width,np.nan)
    f['width_valid']=valid.astype(float);f['ratio_valid']=positive.astype(float)
    return f

COLS=['n','se','se_original','covered','ae','bias_sum','width_sum','width_ae','width_valid','ratio_valid']

def from_totals(t):
    with np.errstate(divide='ignore',invalid='ignore'):
        return dict(rmse=np.sqrt(t[...,1]/t[...,0]),rmse_original=np.sqrt(t[...,2]/t[...,0]),coverage_90=t[...,3]/t[...,0],mae=t[...,4]/t[...,0],bias=t[...,5]/t[...,0],mean_abs_error_over_mean_width=np.where(t[...,6]>0,t[...,7]/t[...,6],np.nan))

def median_draws(frame,ids,draws):
    """Exact clustered-bootstrap row median via integer multiplicities; bounded memory."""
    rows=frame[frame.ratio.notna()].sort_values('ratio')
    if rows.empty:return np.full(len(draws),np.nan)
    positions=pd.Index(ids).get_indexer(rows.dataset_id)
    values=rows.ratio.to_numpy();out=[]
    for start in range(0,len(draws),16):
        chunk=draws[start:start+16]
        counts=np.zeros((len(chunk),len(ids)),dtype=np.int32)
        for j,d in enumerate(chunk):counts[j]=np.bincount(d,minlength=len(ids))
        weights=counts[:,positions];cs=np.cumsum(weights,axis=1)
        totals=cs[:,-1]
        for j,n in enumerate(totals):
            if n==0:out.append(np.nan);continue
            lo=np.searchsorted(cs[j],(n+1)//2,side='left');hi=np.searchsorted(cs[j],n//2+1,side='left')
            out.append((values[lo]+values[hi])/2)
    return np.array(out)

def summarize(frame,ids,draws,**labels):
    f=prepare(frame);stats=f.groupby('dataset_id')[COLS].sum(min_count=1).reindex(ids,fill_value=0)
    t=stats.to_numpy(float);point=from_totals(np.nansum(t,axis=0));boot=from_totals(np.nansum(t[draws],axis=1))
    if 'q05' not in f or not f.q05.notna().any():
        point['coverage_90']=np.nan;boot['coverage_90']=np.full(len(draws),np.nan)
    point['median_abs_error_over_width']=(f.ratio.median() if f.ratio.notna().any() else np.nan)
    boot['median_abs_error_over_width']=median_draws(f,ids,draws)
    result=dict(**labels,n_rows=len(f),n_dgps=f.dataset_id.nunique(),n_width_undefined=int(f.ratio.isna().sum()),bootstrap_replicates=2000)
    for metric,value in point.items():
        result[metric]=float(value)
        finite=boot[metric][np.isfinite(boot[metric])]
        result[metric+'_ci_low']=float(np.quantile(finite,.025)) if len(finite) else np.nan
        result[metric+'_ci_high']=float(np.quantile(finite,.975)) if len(finite) else np.nan
    return result

def per_dgp(frame):
    records=[]
    for (i,head),g in frame.groupby(['dataset_id','head']):
        f=prepare(g);m=from_totals(f[COLS].sum().to_numpy())
        if 'q05' not in f or not f.q05.notna().any():m['coverage_90']=np.nan
        records.append(dict(dataset_id=i,head=head,n_rows=len(f),n_width_undefined=int(f.ratio.isna().sum()),median_abs_error_over_width=(f.ratio.median() if f.ratio.notna().any() else np.nan),**m))
    return pd.DataFrame(records)

def tables(frame,dest,seed,draws_by_head=None,detail_only=False):
    ids=np.array(sorted(frame.dataset_id.unique()));draws=np.random.default_rng(seed).integers(0,len(ids),size=(2000,len(ids)))
    summary=[];gamma=[];strata=[]
    if frame.cohort.iloc[0]=='E2_fresh':
        edges=overlap_edges()
    else:
        edges=np.unique(np.quantile(frame.counterfactual_weight,[0,.25,.5,.75,1]));edges[0]=np.nextafter(edges[0],-np.inf);edges[-1]=np.nextafter(edges[-1],np.inf)
    frame=frame.copy();frame['overlap_quartile']=pd.cut(frame.counterfactual_weight,edges,labels=False,include_lowest=True)
    assert frame.overlap_quartile.notna().all()
    for head,f in frame.groupby('head'):
        d=draws_by_head[head] if draws_by_head is not None else draws
        labels=dict(cohort=f.cohort.iloc[0],head=head,bootstrap_seed=seed)
        summary.append(summarize(f,ids,d,**labels))
        for level,g in f.groupby('gamma_group'):
            gamma.append(summarize(g,ids,d,**labels,gamma=float(g.gamma.mean()),gamma_group=level,gamma_min=float(g.gamma.min()),gamma_max=float(g.gamma.max())))
        for (quartile,level),g in f.groupby(['overlap_quartile','gamma_group']):
            strata.append(summarize(g,ids,d,**labels,overlap_quartile=int(quartile),gamma=float(g.gamma.mean()),gamma_group=level))
    for name,rows in [('summary',summary),('by_gamma',gamma),('by_overlap_gamma',strata)]:
        if name != 'summary' or not detail_only:
            pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)
    if not detail_only:
        per_dgp(frame).to_csv(dest/'per_dgp.csv',index=False)
    return pd.DataFrame(summary)

def project(wide):
    import experiments.robustness.structural_constraints as projection
    output=wide.copy();records=[]
    for key,g in wide.groupby(['dataset_id','source_row_index','a_query'],sort=False):
        g=g.sort_values('gamma');assert not g.gamma.duplicated().any()
        result=projection.project_joint_bounds(g.lower_mean,g.upper_mean)
        output.loc[g.index,'lower_projected']=result.lower;output.loc[g.index,'upper_projected']=result.upper
        raw=projection.structural_violations(g.lower_mean,g.upper_mean);after=projection.structural_violations(result.lower,result.upper)
        record=dict(dataset_id=key[0],source_row_index=key[1],a_query=key[2],raw_lower_monotone=float(raw.lower_nonincreasing<=1e-10),raw_upper_monotone=float(raw.upper_nondecreasing<=1e-10),raw_never_crosses=float(raw.pointwise_order<=1e-10),projected_lower_monotone=float(after.lower_nonincreasing<=1e-10),projected_upper_monotone=float(after.upper_nondecreasing<=1e-10),projected_never_crosses=float(after.pointwise_order<=1e-10),max_residual=after.maximum,raw_max_crossing=raw.pointwise_order)
        if 'theta_lower' in g:
            old=((g.lower_mean-g.theta_lower)**2+(g.upper_mean-g.theta_upper)**2).sum()
            new=((result.lower-g.theta_lower)**2+(result.upper-g.theta_upper)**2).sum()
            assert new<=old+1e-8
            record['squared_error_change']=new-old
        records.append(record)
    return output,pd.DataFrame(records)

def examples(wide,metrics,dest,kl=False):
    ranking=metrics.groupby('dataset_id').rmse.mean().sort_values(kind='stable')
    n=len(ranking);positions=list(range(4))+list(range(n//2-2,n//2+2))+list(range(n-4,n))
    selected=ranking.index[positions]
    pd.DataFrame(dict(dataset_id=selected,rank=np.array(positions)+1,difficulty=['easy']*4+['median']*4+['hard']*4)).to_csv(dest/'example_selection.csv',index=False)
    records=[]
    for i in selected:
        local=wide[wide.dataset_id.eq(i)];row=int(local.source_row_index.min());local=local[local.source_row_index.eq(row)]
        if kl:
            # Both heads are evaluated at every native-head Gamma, with separate truth support.
            joint=local.sort_values(['gamma','bound_type']).drop_duplicates(['a_query','gamma']).copy()
            joint,_=project(joint)
        else:joint=local
        for _,r in joint.iterrows():
            for side in ['lower','upper']:
                truth=r.get(f'theta_{side}',np.nan);kind='native'
                if kl:
                    native=local[(local.a_query==r.a_query)&(local.bound_type==side)]
                    exact=native[native.gamma.eq(r.gamma)]
                    truth=float(exact.theta_reference.iloc[0]) if len(exact) else float(curve_interp(native,np.array([r.gamma]),side)[0])
                    kind='native' if len(exact) else ('interpolated' if np.isfinite(truth) else 'outside_opposite_head_support')
                record=dict(dataset_id=i,source_row_index=row,a_query=int(r.a_query),head=side,gamma=r.gamma,true_bound=truth,reference_kind=kind,predicted_mean=r[f'{side}_mean'],projected_mean=r[f'{side}_projected'],sigma_obs=r.sigma_obs)
                for q in ['05','25','50','75','95']:record['q'+q]=r[f'{side}_q'+q]
                records.append(record)
    pd.DataFrame(records).to_csv(dest/'example_curves.csv',index=False)

def core(stage,smoke=False):
    name=stage+('_smoke' if smoke else '')
    out=fresh(observed_report(stage, smoke))
    wide=fragments(name)
    if stage=='e2':
        wide,structure=project(wide)
        structure.to_csv(out/'structural_curves.csv',index=False)
        ids=np.sort(wide.dataset_id.unique());rng=np.random.default_rng(20260808)
        s=structure.groupby('dataset_id').mean(numeric_only=True).reindex(ids)
        records=[]
        for metric in ['raw_lower_monotone','raw_upper_monotone','raw_never_crosses','projected_lower_monotone','projected_upper_monotone','projected_never_crosses']:
            draws=rng.integers(0,len(ids),size=(2000,len(ids)))
            vals=s[metric].to_numpy();boot=vals[draws].mean(axis=1)
            records.append(dict(metric=metric,estimate=vals.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
        pd.DataFrame(records).to_csv(out/'structural_summary.csv',index=False)
        long=long_frame(wide,'E2_fresh');summary=tables(long,out,20260803)
        projected=wide.copy()
        for side in ['lower','upper']:projected[f'{side}_mean']=wide[f'{side}_projected']
        projected_long=long_frame(projected,'E8_projected');projrows=[]
        projection_rng=np.random.default_rng(20260808)
        for side,g in projected_long.groupby('head'):
            d=projection_rng.integers(0,len(ids),size=(2000,len(ids)))
            projrows.append(summarize(g,ids,d,cohort='E8_projected',head=side,bootstrap_seed=20260808))
        pd.DataFrame(projrows).to_csv(out/'projected_summary.csv',index=False)
        wide.to_parquet(OBSERVED_RUNS/name/'predictions_projected.parquet',index=False)
    else:
        long=long_frame(wide,'E1_fresh',kl=True);summary=tables(long,out,21311726)
    if not smoke:examples(wide,per_dgp(long),out,kl=stage=='e1')
    print(summary[['head','coverage_90','rmse','rmse_original']].to_string(index=False),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['msm', 'kl', 'report-msm', 'report-kl'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage.startswith('report-'):
        core('e2' if args.stage == 'report-msm' else 'e1', args.smoke)
    else:
        if args.stage=='msm':evaluate_e2(args.smoke)
        else:evaluate_e1(args.smoke)

if __name__ == '__main__':
    main()
