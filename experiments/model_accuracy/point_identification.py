"""Gamma=1 head collapse and a matched CausalFM point-CATE comparison."""

from experiments.shared.paths import OBSERVED_RUNS, observed_report
import argparse
import numpy as np
import pandas as pd
import torch
from experiments.shared.observed import observed_run
from experiments.shared.runtime import read, sha, fresh
from experiments.shared.inputs import normalized_root, cohort_artifacts
from experiments.shared.inputs import cohort_ids
from experiments.shared.evaluation import normalized_dataset, MSM_CHECKPOINT, gmm_quantiles

import experiments.comparisons.causalfm as causalfm_comparison
import experiments.shared.msm_reference as msm_reference

E2_SEED=20260726
PREDICTIVE_SEED=causalfm_comparison.SEED
BONUS_BOOT_SEED=causalfm_comparison.SEED+900_000
COLLAPSE_BOOT_SEED=20260804
REPS=2000
GAMMAS=[1.0,1.2]
MODEL_INPUTS=['treatment','outcome']

def cohort(smoke):
    return cohort_ids('msm')[:2] if smoke else cohort_ids('msm')

def e2_fragment(i):
    return pd.read_parquet(OBSERVED_RUNS / 'e2/fragments'/f'dgp_{i}.parquet')

def context_checks(frame, xcols, query, stats):
    assert len(frame)==1024 and len(xcols)==10
    assert np.isfinite(frame[xcols+MODEL_INPUTS].to_numpy()).all()
    assert set(frame.treatment)=={0,1}
    assert stats['y_fit_column']=='outcome'
    np.testing.assert_allclose(frame.outcome.mean(),0,atol=1e-4)
    np.testing.assert_allclose(frame.outcome.std(ddof=0),1,atol=1e-4)
    np.testing.assert_allclose(frame.outcome, np.where(frame.treatment==1,frame.y1,frame.y0), atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(frame.iloc[query.source_row_index.to_numpy()][xcols],query[xcols],atol=1e-6,rtol=1e-6)


def causalfm(smoke):
    run=observed_run('e4_causalfm_smoke' if smoke else 'e4_causalfm',smoke)
    try:
        device=torch.device('cuda:0');root=normalized_root()
        artifacts=cohort_artifacts('msm')
        assert read(OBSERVED_RUNS / 'e2/manifest.json')['status']=='completed'
        position={i:p for p,i in enumerate(cohort_ids('msm'))}
        model,cfm_provenance=causalfm_comparison.load_causalfm(device)
        collapse=[];cfm_rows=[]
        for n,i in enumerate(cohort(smoke)):
            frame,xcols=normalized_dataset(artifacts[i]);assert xcols==causalfm_comparison.X_COLUMNS
            stats=read(root/'scalers'/f'scaler_{i}.json')
            e2=e2_fragment(i);g1=e2[e2.gamma.eq(1.0)].reset_index(drop=True)
            assert len(g1)==16 and g1.source_row_index.drop_duplicates().tolist()==msm_reference.propensity_stratified_row_indices(frame,8,msm_reference.QUANTILES)
            assert g1.a_query.tolist()==[0,1]*8
            g1['base_query_index']=np.arange(16)
            context_checks(frame,xcols,g1,stats)
            # Independent arm-specific CAPO means define the oracle CATE.
            outcome=msm_reference.load_outcome_map(artifacts[i],device)
            latent=msm_reference.seeded_normal((16,16384),E2_SEED+1000*position[i]+i,device)
            with torch.no_grad():
                samples=outcome.evaluate(torch.tensor(g1[xcols].to_numpy(),dtype=torch.float32,device=device),
                    torch.tensor(g1.a_query.to_numpy(),dtype=torch.float32,device=device),latent).cpu().numpy()
            g1['q0']=samples.mean(axis=1)
            # Models see only factual context columns and query covariates (+arm, Gamma for PI-FM).
            X=torch.tensor(frame[xcols].to_numpy(),dtype=torch.float32)
            A=torch.tensor(frame[['treatment']].to_numpy(),dtype=torch.float32)
            Y=torch.tensor(frame[['outcome']].to_numpy(),dtype=torch.float32)
            query=g1[g1.a_query.eq(0)].reset_index(drop=True)
            call=lambda:model.estimate_cate(X,A,Y,torch.tensor(query[xcols].to_numpy(),dtype=torch.float32),return_gmm=True)
            out=call();causalfm_comparison.assert_gmm_output(out,8)
            pi,mu,sigma=(out[k].detach().cpu().numpy() for k in ('gmm_pi','gmm_mu','gmm_sigma'))
            q=gmm_quantiles(pi,mu,sigma,[0.05,0.95])
            local=pd.DataFrame(dict(dataset_id=i,query_position=np.arange(8),source_row_index=query.source_row_index,
                causalfm_cate=out['cate'].detach().cpu().numpy(),causalfm_q05=q[:,0],causalfm_q95=q[:,1]))
            for name,values in (('pi',pi),('mu',mu),('sigma',sigma)):
                for k in range(values.shape[1]):local[f'causalfm_{name}_{k}']=values[:,k]
            cfm_rows.append(local)
            g1['dataset_id']=i;g1['sigma_obs']=stats['y_std'];collapse.append(g1)
            run.manifest['completed_ids'].append(i)
            if n%64==0:run.save();print(f'E4 {n+1}/{len(cohort(smoke))}',flush=True)
        collapse=pd.concat(collapse,ignore_index=True)
        instances=causalfm_comparison.paired_e4_instances(collapse)
        instances=causalfm_comparison.attach_pifm_predictive_intervals(instances,collapse,4096,PREDICTIVE_SEED)
        predictions=instances.merge(pd.concat(cfm_rows,ignore_index=True),on=['dataset_id','query_position','source_row_index'],validate='one_to_one')
        predictions=predictions.merge(collapse.groupby('dataset_id').sigma_obs.first().reset_index(),on='dataset_id',validate='many_to_one')
        assert len(predictions)==8*len(cohort(smoke)) and np.isfinite(predictions.select_dtypes('number')).all().all()
        predictions.to_parquet(run.path/'predictions.parquet',index=False)
        run.manifest.update(causalfm=cfm_provenance,pifm_checkpoint_sha256=sha(MSM_CHECKPOINT),
            upstream_manifest_sha256=sha(OBSERVED_RUNS / 'e2/manifest.json'),
            oracle_samples=16384,oracle_seed_rule='20260726 + 1000*position + dataset_id',
            predictive_mc_samples=4096,predictive_seed=PREDICTIVE_SEED)
        run.done()
    except BaseException:
        run.done('failed');raise

def long_ci(point,boot,**labels):
    finite=boot[np.isfinite(boot)]
    return dict(**labels,estimate=float(point),ci_low=float(np.quantile(finite,.025)),ci_high=float(np.quantile(finite,.975)))

def collapse_tables(smoke):
    ids=np.array(cohort(smoke));frames=[e2_fragment(i) for i in ids]
    draws=np.random.default_rng(COLLAPSE_BOOT_SEED).integers(0,len(ids),size=(REPS,len(ids)))
    summary=[];per=[]
    for gamma in GAMMAS:
        g=[f[np.isclose(f.gamma,gamma,atol=1e-9,rtol=0)] for f in frames]
        assert all(len(x)==16 for x in g)
        arr=lambda c:np.stack([x[c].to_numpy(float) for x in g])  # (DGP, 16 query-arm rows)
        sigma=arr('sigma_obs');gap=arr('upper_mean')-arr('lower_mean');true_width=arr('theta_upper')-arr('theta_lower')
        if gamma==1.0:assert np.all(true_width==0)
        labels=dict(cohort='E4_fresh',gamma=gamma,n_rows=gap.size,n_dgps=len(ids),bootstrap_seed=COLLAPSE_BOOT_SEED,bootstrap_replicates=REPS)
        B=lambda m:m[draws].reshape(REPS,-1)
        for name,fn in [('mean_abs_gap',lambda m:np.abs(m).mean(-1)),('median_abs_gap',lambda m:np.median(np.abs(m),-1)),
                        ('p95_abs_gap',lambda m:np.quantile(np.abs(m),.95,axis=-1)),('mean_signed_gap',lambda m:m.mean(-1))]:
            summary.append(long_ci(fn(gap.ravel()),fn(B(gap)),head='between_heads',metric=name,**labels))
        summary.append(long_ci(true_width.mean(),B(true_width).mean(-1),head='between_heads',metric='mean_true_width',**labels))
        row=dict(dataset_id=ids,gamma=gamma,n_rows=16,sigma_obs=sigma[:,0],mean_abs_gap=np.abs(gap).mean(1),
                 median_abs_gap=np.median(np.abs(gap),1),max_abs_gap=np.abs(gap).max(1),mean_true_width=true_width.mean(1))
        for side in ['lower','upper']:
            e=arr(f'{side}_mean')-arr(f'theta_{side}')
            cov=((arr(f'theta_{side}')>=arr(f'{side}_q05'))&(arr(f'theta_{side}')<=arr(f'{side}_q95'))).astype(float)
            width=arr(f'{side}_q95')-arr(f'{side}_q05')
            metrics=[('rmse',e**2,np.sqrt),('rmse_original',(e*sigma)**2,np.sqrt),('mae',np.abs(e),None),('bias',e,None),
                     ('coverage_90',cov,None),('mean_predictive_width_90',width,None)]
            for name,m,post in metrics:
                f=post or (lambda v:v)
                summary.append(long_ci(f(m.mean()),f(B(m).mean(-1)),head=side,metric=name,**labels))
                row[f'{side}_{name}']=f(m.mean(1))
        per.append(pd.DataFrame(row))
    summary=pd.DataFrame(summary);first=['cohort','gamma','head','metric','estimate','ci_low','ci_high']
    return summary[first+[c for c in summary if c not in first]],pd.concat(per,ignore_index=True)

def causalfm_tables(smoke):
    name='e4_causalfm_smoke' if smoke else 'e4_causalfm'
    assert read(OBSERVED_RUNS/name/'manifest.json')['status']=='completed'
    p=pd.read_parquet(OBSERVED_RUNS/name/'predictions.parquet')
    assert p.dataset_id.drop_duplicates().tolist()==list(cohort(smoke))
    point=causalfm_comparison.point_metrics(p);predictive=causalfm_comparison.predictive_metrics(p)
    boot=causalfm_comparison.clustered_bootstrap(p,point,predictive,BONUS_BOOT_SEED,REPS)
    # Reuse cluster draws for paired uncertainty.
    ids=p.dataset_id.drop_duplicates().to_numpy();n=len(ids)
    draws=np.random.default_rng(BONUS_BOOT_SEED).integers(0,n,size=(REPS,n))
    per=p.groupby('dataset_id',sort=False)
    def dgp_sum(v):return pd.Series(v,index=p.index).groupby(p.dataset_id,sort=False).sum().to_numpy()
    se={m:(p[c]-p.oracle_cate)**2 for m,c in [('causalfm','causalfm_cate'),('pifm','pifm_cate')]}
    se_orig={m:v*p.sigma_obs**2 for m,v in se.items()}
    count=dgp_sum(np.ones(len(p)))
    extra={}
    for m in se:
        extra[m,'pehe_original']=(np.sqrt(se_orig[m].mean()),np.sqrt(dgp_sum(se_orig[m])[draws].sum(1)/count[draws].sum(1)))
    rm=lambda m:np.sqrt(dgp_sum(se[m])[draws].sum(1)/count[draws].sum(1))
    pehe_diff=(np.sqrt(se['causalfm'].mean())-np.sqrt(se['pifm'].mean()),rm('causalfm')-rm('pifm'))
    labels=dict(n_queries=len(p),n_dgps=n,bootstrap_seed=BONUS_BOOT_SEED,bootstrap_replicates=REPS)
    method={'causalfm':('causalfm','direct'),'pifm_midpoint':('pifm','midpoint'),'causalfm_direct':('causalfm','direct'),
            'pifm_lower_derived':('pifm','lower_derived'),'pifm_upper_derived':('pifm','upper_derived')}
    summary=[dict(method=method[r.method][0],distribution=method[r.method][1],metric=r.metric,estimate=r.point,ci_low=r.lower_95,ci_high=r.upper_95,**labels)
             for r in boot[boot.family.ne('paired')].itertuples() if r.metric!='rmse']  # rmse duplicates pehe
    summary+=[long_ci(v[0],v[1],method=m,distribution='direct' if m=='causalfm' else 'midpoint',metric=k,**labels) for (m,k),v in extra.items()]
    paired=[dict(contrast='causalfm_minus_pifm_midpoint',metric=r.metric,estimate=r.point,ci_low=r.lower_95,ci_high=r.upper_95,**labels)
            for r in boot[boot.family.eq('paired')].itertuples()]
    paired.append(long_ci(*pehe_diff,contrast='causalfm_minus_pifm_midpoint',metric='pehe_difference',**labels))
    rows=[]
    for i,g in per:
        ce=g.causalfm_cate-g.oracle_cate;pe=g.pifm_cate-g.oracle_cate
        cov=lambda lo,hi:float(((g.oracle_cate>=g[lo])&(g.oracle_cate<=g[hi])).mean())
        rows.append(dict(dataset_id=i,n_queries=len(g),sigma_obs=g.sigma_obs.iloc[0],causalfm_pehe=np.sqrt((ce**2).mean()),pifm_pehe=np.sqrt((pe**2).mean()),
            causalfm_mae=ce.abs().mean(),pifm_mae=pe.abs().mean(),causalfm_bias=ce.mean(),pifm_bias=pe.mean(),
            mean_absolute_error_difference=(ce.abs()-pe.abs()).mean(),causalfm_coverage_90=cov('causalfm_q05','causalfm_q95'),
            pifm_lower_derived_coverage_90=cov('pifm_lower_q05','pifm_lower_q95'),pifm_upper_derived_coverage_90=cov('pifm_upper_q05','pifm_upper_q95'),
            pifm_mean_abs_collapse_gap=g.pifm_cate_collapse_gap.abs().mean()))
    query=p[['dataset_id','query_position','source_row_index','sigma_obs','oracle_cate','causalfm_cate','pifm_cate','pifm_cate_lower','pifm_cate_upper',
             'causalfm_q05','causalfm_q95','pifm_lower_q05','pifm_lower_q95','pifm_upper_q05','pifm_upper_q95']]
    return pd.DataFrame(summary),pd.DataFrame(paired),pd.DataFrame(rows),query

def report(smoke):
    out=fresh(observed_report('e4', smoke))
    summary,per=collapse_tables(smoke)
    summary.to_csv(out/'collapse_summary.csv',index=False);per.to_csv(out/'collapse_per_dgp.csv',index=False)
    for name,frame in zip(['causalfm_summary','causalfm_paired','causalfm_per_dgp','causalfm_query_rows'],causalfm_tables(smoke)):
        frame.to_csv(out/f'{name}.csv',index=False)
    print(summary.to_string());print(pd.read_csv(out/'causalfm_summary.csv').to_string());print(pd.read_csv(out/'causalfm_paired.csv').to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['causalfm','report']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    {'causalfm':causalfm,'report':report}[args.stage](args.smoke)
