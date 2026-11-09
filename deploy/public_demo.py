"""Six-tab educational UI isolated from the IHDP research dashboard."""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from causallens.dashboard import effect_groups, json_safe
from causallens.evaluation.robustness import sensitivity_unobserved_confounding
from public_demo_utils import (
    synthetic_balance, disagreement_matrix, synthetic_profiles,
    naive_association, comparison_interpretation,
)
from causallens.synthetic_demo import FEATURES, GENERATOR_VERSION, fit_example
from causallens.evaluation.metrics import pehe_score
from causallens.evaluation.policy import simulate_capacity_constrained_policy

st.set_page_config(page_title='CausalLens · Synthetic demo', page_icon='🔬', layout='wide')
def chart(fig, width='stretch'):
    """Consistent responsive charts using the existing slate/teal theme."""
    fig.update_layout(template='plotly_white', font=dict(color='#0F172A'),
                      margin=dict(l=20, r=20, t=55, b=30),
                      colorway=['#0F766E', '#2563EB', '#D97706', '#9333EA', '#64748B', '#DC2626'],
                      legend=dict(orientation='h', y=-0.25), height=400)
    st.plotly_chart(fig, width=width, config={'displaylogo': False})


st.title('CausalLens')
st.info('SYNTHETIC DEMONSTRATION — all features, assignments, and outcomes are independently generated. '
        'No IHDP records or benchmark findings are displayed. These examples are not clinical evidence.')
st.write('Explore how association, estimated effects, and known simulated truth differ. '
         'Follow the six tabs from data exploration to a reproducible evidence summary.')
with st.expander('Research vs. public demonstration — data provenance'):
    st.markdown(
        '- This public application uses **independently generated synthetic data only**.\n'
        '- A separate research pipeline evaluates causal estimators on the **IHDP-100** semi-synthetic benchmark.\n'
        '- IHDP records and benchmark results are **not shown publicly** while hosting/display permissions remain unresolved.\n'
        '- Simulated examples here are educational and **do not constitute clinical evidence**.'
    )
with st.container(border=True):
    st.caption('How to use this demo — four steps')
    for col, label, hint in zip(st.columns(4), [
            '① Explore Association', '② Compare Estimates', '③ Simulate Decisions', '④ Assess Evidence'], [
            'Data & Overview — the raw association and confounding indicators.',
            'Model Evaluation — estimator accuracy against the known effect.',
            'Counterfactual Explorer & Policy Simulator — individual units and capacity-constrained decisions.',
            'Robustness & Sensitivity, then Evidence Summary — limitations, then export findings.']):
        col.markdown(f'**{label}**')
        col.caption(hint)

@st.cache_data(max_entries=4, show_spinner=False)
def example(seed):
    return fit_example(seed)

with st.sidebar:
    st.header('Synthetic scenario')
    seed = st.selectbox('Example seed', list(range(10)))
    st.caption('Ten reproducible simulations. Features have arbitrary units and no real-person meaning.')
with st.spinner('Fitting six estimators on synthetic training data…'):
    train, test, truth, predictions, records = example(seed)
with st.sidebar:
    name = st.selectbox('Estimator', list(predictions), index=2)
    st.caption('Data provenance is explained above; the research dashboard and IHDP findings remain separate.')
pred = predictions[name]
frame = pd.DataFrame(test['X'], columns=FEATURES)
frame['Treatment'] = test['T'].astype(int).astype(str)
frame['Observed synthetic outcome'] = test['Y']
frame['Oracle synthetic effect'] = truth['tau']
frame['Estimated effect'] = pred
metrics = pd.DataFrame(records)
naive = naive_association(test['Y'], test['T'])
estimated = float(pred.mean())
oracle = float(truth['tau'].mean())
k1, k2, k3 = st.columns(3)
k1.metric('Naive observed association', f'{naive:.3f}',
          help='Difference in mean observed outcome between treated and control units in this held-out population.')
k2.metric(f'Estimated effect · {name}', f'{estimated:.3f}',
          help='Mean predicted CATE from the selected estimator on the same held-out population.')
k3.metric('True simulated effect (oracle)', f'{oracle:.3f}',
          help='Mean of the known, generator-defined treatment effect on the same held-out population.')
st.caption(comparison_interpretation(naive, estimated, oracle, name))
st.caption('600 training units • 150 held-out units • five independent synthetic features • arbitrary outcome units')
tabs = st.tabs(['Data & Overview', 'Model Evaluation', 'Counterfactual Explorer',
                'Policy Simulator', 'Robustness & Sensitivity', 'Evidence Summary'])
with tabs[0]:
    st.subheader('Association is a starting point')
    st.write('Treatment assignment depends on features A and B, which also influence outcomes. '
             'The unadjusted outcome difference can therefore mix treatment effects with confounding.')
    chart(px.box(frame, x='Treatment', y='Observed synthetic outcome', color='Treatment'), width='stretch')
    st.subheader('Training covariate balance and treatment overlap')
    left, right = st.columns(2)
    with left:
        balance = synthetic_balance(train)
        fig = px.bar(balance, x='Signed SMD', y='Feature', orientation='h',
                     title='Synthetic covariate imbalance')
        for threshold in (-0.1, 0.1):
            fig.add_vline(x=threshold, line_dash='dot', line_color='#64748B')
        chart(fig)
    with right:
        x = train['X']
        propensity = 0.15 + 0.7 / (1 + np.exp(-(0.8*x[:, 0] - 0.5*x[:, 1])))
        overlap = pd.DataFrame({'Oracle assignment probability': propensity,
                                'Arm': np.where(train['T'] == 1, 'Treated', 'Control')})
        fig = px.histogram(overlap, x='Oracle assignment probability', color='Arm',
                           histnorm='probability density', barmode='overlay', opacity=0.6,
                           title='Synthetic treatment overlap', nbins=20)
        fig.update_xaxes(range=[0, 1])
        chart(fig)
    st.caption('Training data only. Signed SMD is treated minus control mean divided by pooled within-arm SD; '
               '±0.1 is a descriptive guide, not a validity test. Overlap uses known generator probabilities, '
               'not fitted propensity scores; neither plot rules out hidden confounding in real studies.')
    with st.expander('View raw synthetic sample table'):
        st.dataframe(frame[FEATURES + ['Treatment', 'Observed synthetic outcome']], hide_index=True)
    with st.expander('How the synthetic data are generated'):
        st.markdown('Features A–E are independent standard normal draws. Treatment probability is '
                    '`0.15 + 0.7 × sigmoid(0.8 A − 0.5 B)`. Control mean is '
                    '`2 + 0.8 A + 0.5 B² − 0.3 C`; effect is `0.7 + 0.9 A − 0.6 C`. '
                    'Observed outcomes add independent standard normal noise. These formulas were chosen '
                    'for education and were not calibrated to IHDP.')
with tabs[1]:
    st.subheader('Held-out synthetic evaluation')
    st.caption('Actual fits on this generated example. Rankings are illustrative and are not IHDP benchmark results.')
    sort_metric = st.selectbox('Compare models by', ['Root-PEHE', 'ATE error', 'Policy regret'],
                               help='All three errors are lower-is-better on the same held-out synthetic cohort.')
    labels = {'pehe': 'Root-PEHE', 'ate_error': 'ATE error', 'policy_risk': 'Policy regret'}
    comparison = metrics.rename(columns=labels).sort_values(sort_metric)
    with st.expander('Full metrics table (all estimators)'):
        st.dataframe(comparison, hide_index=True)
    for metric in [sort_metric] + [item for item in labels.values() if item != sort_metric]:
        panel = st.container() if metric == sort_metric else st.expander(metric + ' comparison')
        with panel:
            fig = px.bar(comparison, x=metric, y='estimator', orientation='h',
                         title=metric + ' · lower is better', text_auto='.3f')
            fig.update_yaxes(title=None, categoryorder='array', categoryarray=comparison['estimator'][::-1])
            chart(fig)
    st.caption('Policy regret here compares the positive-effect treatment rule with an unconstrained oracle. '
               'Capacity-constrained results are in Policy Simulator. Single-seed rankings have no confidence intervals; '
               'change the example seed to explore variation. Constant-effect models cannot express heterogeneity.')
    chart(px.scatter(frame, x='Oracle synthetic effect', y='Estimated effect',
                              color='Treatment'), width='stretch')
    effects = pd.DataFrame({'Oracle synthetic effect': truth['tau'], 'Estimated effect': pred})
    chart(px.histogram(effects.melt(var_name='Source', value_name='Synthetic effect'),
                       x='Synthetic effect', color='Source', barmode='overlay', opacity=0.55,
                       nbins=25, title='CATE distribution · selected estimator vs oracle'))
    feature = st.selectbox('Heterogeneity feature', FEATURES,
                           help='Association between a synthetic covariate and effects; not the effect of changing that covariate.')
    chart(px.scatter(frame, x=feature, y=['Oracle synthetic effect', 'Estimated effect'],
                     title='Treatment heterogeneity · held-out synthetic units'))
    groups = pd.DataFrame({'Predicted-effect group': effect_groups(pred),
                           'Oracle synthetic effect': truth['tau'], 'Estimated effect': pred})
    grouped = groups.groupby('Predicted-effect group', observed=True)[['Oracle synthetic effect', 'Estimated effect']].mean().reset_index()
    chart(px.bar(grouped, x='Predicted-effect group', y=['Oracle synthetic effect', 'Estimated effect'],
                 barmode='group', title='Mean effects within predicted-effect groups'))
    st.caption('Groups use the selected model’s predicted effects, keeping ties together. '
               'Distribution spread describes heterogeneity, not estimation uncertainty.')
    st.write('Root-PEHE measures effect prediction error; ATE error measures average-effect error. '
             'Lower is better. The oracle is known only because this is a simulation.')
    st.download_button('Download synthetic evaluation (CSV)', metrics.to_csv(index=False),
                       'causallens_synthetic_evaluation.csv', 'text/csv')
with tabs[2]:
    st.subheader('Inspect a fictional unit')
    profiles = synthetic_profiles(truth, predictions)
    profile = st.selectbox('Synthetic profile', list(profiles),
                           help='Presets select fictional held-out units using known synthetic truth or model disagreement.')
    unit = st.slider('Synthetic unit index', 0, len(frame)-1, 0, disabled=profiles[profile] is not None)
    if profiles[profile] is not None:
        unit = profiles[profile]
    st.caption(f'Selected fictional unit {unit} · {profile}. Presets are illustrative, not representative people.')
    st.dataframe(frame.iloc[[unit]][FEATURES], hide_index=True)
    chart(px.bar(pd.DataFrame({'Quantity': ['Oracle control mean', 'Oracle treated mean'],
                                       'Synthetic outcome': [truth['mu0'][unit], truth['mu1'][unit]]}),
                          x='Quantity', y='Synthetic outcome'), width='stretch')
    profile_effects = pd.DataFrame({'Estimator': list(predictions),
                                    'Estimated synthetic effect': [values[unit] for values in predictions.values()]})
    fig = px.bar(profile_effects, x='Estimated synthetic effect', y='Estimator', orientation='h',
                 title='Same synthetic profile · six fitted estimators')
    fig.add_vline(x=float(truth['tau'][unit]), line_dash='dash', line_color='#D97706',
                  annotation_text='Oracle effect')
    chart(fig)
    st.metric('Estimated effect for this synthetic unit' , f'{pred[unit]:.3f}')
    st.caption('Bars are known synthetic mean potential outcomes, not fitted predictions or observations '
               'of two real outcomes. No individual confidence interval is provided.')
with tabs[3]:
    st.subheader('Allocate a limited treatment budget')
    capacity = st.slider('Treatment capacity (%)', 0, 100, 40, help='Maximum share treated; only positive predicted effects are prioritized.')
    policy = simulate_capacity_constrained_policy(truth['mu0'], truth['mu1'], pred, capacity/100)
    strategy = pd.DataFrame(policy['strategies']).T.rename_axis('Strategy').reset_index()
    chart(px.bar(strategy, x='Strategy', y='policy_value'), width='stretch')
    with st.expander('Strategy details table'):
        st.dataframe(strategy, hide_index=True)
    st.caption('Model-guided allocation prioritizes positive predicted effects within capacity. '
               'The oracle uses known synthetic effects. Treat All is an unconstrained comparator; '
               'its constrained regret is unavailable below 100% capacity. Random value is an expectation.')
with tabs[4]:
    st.subheader('Disagreement and prediction stress')
    matrix = disagreement_matrix(predictions)
    fig = px.imshow(matrix, text_auto='.2f', color_continuous_scale='Teal',
                    labels={'color': 'Mean absolute effect difference'},
                    title='Estimator disagreement · synthetic outcome units', aspect='auto')
    chart(fig)
    st.caption('Disagreement is descriptive, not a calibrated uncertainty interval; it does not establish causal validity.')
    with st.expander('Prediction stress test (advanced)'):
        disagreement = np.std(np.stack(list(predictions.values())), axis=0)
        chart(px.histogram(x=disagreement, labels={'x': 'Across-estimator effect standard deviation'}), width='stretch')
        noise = st.slider('Prediction noise standard deviation', 0.0, 3.0, 0.0, 0.1, help='Adds a fixed seeded noise vector scaled by this value; models are not retrained.')
        stressed = pred + np.random.default_rng(904 + seed).normal(size=pred.size)*noise
        st.metric('Root-PEHE after synthetic prediction perturbation', f'{pehe_score(truth["tau"], stressed):.3f}')
        perturbation = np.random.default_rng(904 + seed).normal(size=pred.size)
        stress_curve = pd.DataFrame({'Noise standard deviation': np.linspace(0, 3, 16),
                                    'Synthetic Root-PEHE': [pehe_score(truth['tau'], pred + level*perturbation)
                                                           for level in np.linspace(0, 3, 16)]})
        fig = px.line(stress_curve, x='Noise standard deviation', y='Synthetic Root-PEHE', markers=True,
                      title='Prediction stress curve · one seeded perturbation')
        fig.add_vline(x=noise, line_dash='dot')
        chart(fig)
        st.caption('This perturbs predictions only; it is not a hidden-confounding analysis or model refit.')
    with st.expander('Linear omitted-variable sensitivity (advanced)'):
        st.subheader('Linear omitted-variable sensitivity · synthetic training data')
        sensitivity = sensitivity_unobserved_confounding(train['X'], train['T'], train['Y'])
        sensitivity_display = pd.DataFrame({
            'Target': ['Attenuate OLS estimate to zero', 'Lose significance at 5%'],
            'Required partial R² (%)': [sensitivity['robustness_value_point_estimate_percent'],
                                        sensitivity['robustness_value_significance_percent']]})
        chart(px.bar(sensitivity_display, x='Required partial R² (%)', y='Target', orientation='h',
                     text_auto='.2f', title='Cinelli–Hazlett robustness values · linear specification'))
        st.caption('Reuses the research sensitivity calculation for OLS Y ~ T + X on synthetic training data. '
                   'The values describe equal-strength partial R² with treatment and outcome needed for each target. '
                   'They do not apply to the selected nonlinear estimator or individual effects. The generator has '
                   'observed confounders and a nonlinear outcome term; this linear specification is deliberately limited.')
with tabs[5]:
    st.subheader('Evidence and limits')
    st.write('These are reproducible synthetic teaching examples, not scientific findings about IHDP or any '
             'population — see “Research vs. public demonstration” near the top for data provenance.')
    st.write('Identification assumptions: consistency, no interference, adequate overlap, and adjustment for all '
             'treatment–outcome confounders. The toy generator makes its confounders observed; real studies need '
             'substantive justification. This dashboard is not a treatment recommendation.')
    evidence = json_safe({'data_kind': 'fully synthetic; no IHDP records', 'generator': GENERATOR_VERSION,
                         'seed': seed, 'estimator': name, 'train_units': 600, 'test_units': 150,
                         'evaluation': records, 'policy': policy,
                         'synthetic_profile': {'selection': profile, 'unit_index': unit},
                         'prediction_stress': {'noise_sd': noise, 'root_pehe': pehe_score(truth['tau'], stressed)},
                         'linear_sensitivity': sensitivity,
                         'limits': 'Educational simulation only; no empirical benchmark or clinical claim.'})
    st.download_button('Download synthetic evidence (JSON)', json.dumps(evidence, indent=2, allow_nan=False),
                       'causallens_synthetic_evidence.json', 'application/json')
    st.caption('No dataset uploads, external data retrieval, credentials, or private files are used by this public demo.')
