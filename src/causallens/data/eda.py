"""Training-only descriptive EDA; no oracle access or causal estimators."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .loader import load_ihdp, FEATURE_NAMES


def balance_table(X,T):
    rows=[]
    for j,name in enumerate(FEATURE_NAMES):
        a,b=X[T==1,j],X[T==0,j]
        levels=np.unique(X[:,j]); binary=len(levels)==2
        if binary:
            a,b=(a==levels[-1]).astype(float),(b==levels[-1]).astype(float)
            denom=np.sqrt((a.mean()*(1-a.mean())+b.mean()*(1-b.mean()))/2)
        else:
            denom=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        delta=a.mean()-b.mean()
        smd=delta/denom if denom else (0. if delta==0 else np.nan)
        rows.append(dict(feature=name,kind='binary' if binary else 'numeric',
                         treated_mean=a.mean(),control_mean=b.mean(),
                         treated_sd=a.std(ddof=1),control_sd=b.std(ddof=1),smd=smd))
    return pd.DataFrame(rows)


def run_eda(root='.',replication=0):
    root=Path(root); out=root/'reports/figures';out.mkdir(parents=True,exist_ok=True)
    d=load_ihdp(replication,'train',root/'data/raw');X,T,Y=d['X'],d['T'],d['Y']
    balance=balance_table(X,T); counts=[int((T==g).sum()) for g in (0,1)]
    colors=['#3876a8','#dc793b']
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Control','Treated'],counts,color=colors)
    for i,n in enumerate(counts):ax.text(i,n+4,f'{n} ({n/len(T):.1%})',ha='center')
    ax.set(ylim=(0,max(counts)*1.16),ylabel='Observations',title=f'Training treatment distribution · replication {replication}')
    save(fig,'01_treatment.png')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,ax in enumerate(axes.flat):
        bins=np.histogram_bin_edges(X[:,j],bins=15)
        for g,c,label in zip((0,1),colors,('Control','Treated')):ax.hist(X[T==g,j],bins=bins,density=True,histtype='step',linewidth=2,color=c,label=label)
        ax.set(title=FEATURE_NAMES[j],xlabel='Archived value',ylabel='Density')
    axes.flat[0].legend();fig.suptitle('Six multi-valued covariates · training only');save(fig,'02_numeric_distributions.png')
    ordered=balance.sort_values('smd',key=lambda s:s.abs())
    fig,ax=plt.subplots(figsize=(8,8));ax.barh(ordered.feature,ordered.smd,color='#3876a8');ax.axvline(0,color='black',lw=.7)
    for v in (-.1,.1):ax.axvline(v,color='gray',ls='--',lw=.8)
    ax.set(xlabel='Signed standardized mean difference (treated − control)',title='Covariate imbalance · ±0.1 is a descriptive reference');save(fig,'03_balance.png')
    fig,ax=plt.subplots(figsize=(8,4));bins=np.histogram_bin_edges(Y,bins=25)
    for g,c,label in zip((0,1),colors,('Control','Treated')):ax.hist(Y[T==g],bins=bins,density=True,histtype='step',linewidth=2,color=c,label=label)
    ax.legend();ax.set(xlabel='Simulated factual outcome (original archive scale)',ylabel='Density',title='Observed outcome distributions · association only');save(fig,'04_outcomes.png')
    # Equal-width marginal bins fixed from observed training covariates only.
    overlap=[];fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,ax in enumerate(axes.flat):
        edges=np.linspace(X[:,j].min(),X[:,j].max(),9)
        c=np.histogram(X[T==0,j],edges)[0];t=np.histogram(X[T==1,j],edges)[0]
        image=np.vstack([c,t]);ax.imshow(image,aspect='auto',cmap='Blues')
        for r in range(2):
            for k in range(8):ax.text(k,r,str(image[r,k]),ha='center',va='center',color='white' if image[r,k]>image.max()/2 else 'black')
        ax.set(yticks=[0,1],yticklabels=['Control','Treated'],xticks=range(8),xticklabels=range(1,9),xlabel='Equal-width training bin (low → high)',title=FEATURE_NAMES[j])
        overlap.append(dict(feature=FEATURE_NAMES[j],edges=edges.tolist(),control=c.tolist(),treated=t.tolist(),one_arm_bins=int(((c+t>0)&((c==0)|(t==0))).sum())))
    fig.suptitle('Marginal overlap: counts in 8 bins per feature');save(fig,'05_overlap.png')
    naive=float(Y[T==1].mean()-Y[T==0].mean());top=balance.loc[balance.smd.abs().idxmax()]
    fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
    ax.text(.03,.9,'CausalLens · IHDP training inspection',fontsize=22,weight='bold')
    ax.text(.03,.70,f'Replication {replication} · n={len(T)} · 25 covariates · treated {counts[1]/len(T):.1%}',fontsize=15)
    ax.text(.03,.49,f'Largest |SMD|: {top.feature} = {abs(top.smd):.3f}\nNaive outcome difference: {naive:.3f} (association)',fontsize=17,linespacing=1.6)
    ax.text(.03,.15,'Imbalance and sparse marginal cells motivate careful causal analysis.\nThey do not prove confounding or identify a treatment effect.\nSemi-synthetic benchmark; no clinical effectiveness claim.',fontsize=12,linespacing=1.5)
    save(fig,'06_summary.png')
    lines=['# Phase 3 EDA — IHDP-100', '',f'Replication {replication}, official training split only. All results describe this replication; no test outcomes or oracle fields were used. No model, propensity estimator or preprocessing was fitted.', '',
           f'**Sample:** {len(T)} observations, {counts[1]} treated ({counts[1]/len(T):.2%}) and {counts[0]} controls. Missing/nonfinite entries: {int((~np.isfinite(X)).sum())} in X and {int((~np.isfinite(Y)).sum())} in Y.', '',
           f'**Outcome association:** treated mean {Y[T==1].mean():.6f}, control mean {Y[T==0].mean():.6f}; difference **{naive:.6f}**. This comparison mixes different covariate distributions and is not a causal effect.', '',
           f'**Balance:** largest absolute SMD is {top.feature} ({top.smd:.6f}); {(balance.smd.abs()>.1).sum()} of 25 exceed 0.1. This conventional descriptive reference is not a significance test or proof of confounding.', '',
           '## Covariate summaries and balance', '', 'For numeric columns SMD divides the mean difference by sqrt((s1²+s0²)/2), using sample variances. For binary columns it uses upper-level proportions and sqrt((p1(1−p1)+p0(1−p0))/2). Binary means below are upper-level frequencies; x14 upper level is 2. No values are recoded in stored/model data. Zero denominators yield zero only for identical means, otherwise undefined.', '',
           '| Feature | Type | Treated mean/proportion | Control mean/proportion | Treated SD | Control SD | SMD |','|---|---|---|---|---|---|---|']
    for r in balance.itertuples():lines.append(f'| {r.feature} | {r.kind} | {r.treated_mean:.4f} | {r.control_mean:.4f} | {r.treated_sd:.4f} | {r.control_sd:.4f} | {r.smd:.4f} |')
    lines+=['','## Distributions and unusual values','','The six leading numeric columns have archived scaled values; large magnitudes are not automatically errors. No observations are removed. The remaining nineteen columns have two levels; x14 uses {1,2}, the others {0,1}. Feature names remain generic because clinical coding is unverified. Histograms share bins between arms and use density so unequal group sizes do not dominate visual comparison.','', '## Marginal overlap diagnostics','', 'Eight equal-width bins are defined separately for each numeric covariate using training values. Occupied cells with only one arm are sample sparsity warnings, not proof of population positivity failure. Marginal overlap cannot certify joint overlap. Bin counts are descriptive and sensitive to binning.','']
    for r in overlap:lines.append(f"- {r['feature']}: {r['one_arm_bins']} occupied one-arm bins; control counts {r['control']}, treated counts {r['treated']}.")
    lines+=['','For binary features, group proportions of zero or one in the table flag absent observed categories within an arm. The known benchmark selection removes treated children with nonwhite mothers (see dataset_research.md); observed imbalance alone does not prove any measured feature is a causal confounder. Known simulated outcomes are not evidence of real medical effectiveness. No independence across replications is assumed.','','## Figures and reproduction','','Run `python scripts/eda_ihdp.py` or notebook 02 from the installed environment. Six figures are written locally to ignored reports/figures/: treatment counts, numeric distributions, balance, factual outcomes, marginal overlap, and presentation summary. No row-level data are uploaded.','','## Boundary and next step','','Phase 3 complete. Optional training-only preprocessing is implemented separately in Phase 4; this EDA uses raw archived values. Phase 5 QA remains separate. No outcome transformations, resplitting, imputation or feature deletion occurred.']
    (root/'reports/eda_summary.md').write_text('\n'.join(lines)+'\n')
    return balance,overlap
