"""Six-tab educational UI isolated from the IHDP research dashboard."""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from causallens.dashboard import json_safe
from causallens.synthetic_demo import FEATURES, GENERATOR_VERSION, fit_example
from causallens.evaluation.metrics import pehe_score
from causallens.evaluation.policy import simulate_capacity_constrained_policy

st.set_page_config(page_title='CausalLens · Synthetic demo', page_icon='🔬', layout='wide')
st.title('CausalLens')
st.info('SYNTHETIC DEMONSTRATION — all features, assignments, and outcomes are independently generated. '
        'No IHDP records or benchmark findings are displayed. These examples are not clinical evidence.')
st.write('Explore how association, estimated effects, and known simulated truth differ. '
         'Follow the six tabs from data exploration to a reproducible evidence summary.')

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
    st.caption('The research dashboard and IHDP findings remain separate. Public data permissions are unresolved.')
pred = predictions[name]
frame = pd.DataFrame(test['X'], columns=FEATURES)
frame['Treatment'] = test['T'].astype(int).astype(str)
frame['Observed synthetic outcome'] = test['Y']
frame['Oracle synthetic effect'] = truth['tau']
frame['Estimated effect'] = pred
metrics = pd.DataFrame(records)
c1, c2, c3 = st.columns(3)
c1.metric('Synthetic test units', len(test['X']))
c2.metric('Estimated test mean effect', f'{pred.mean():.3f}')
c3.metric('Oracle synthetic test mean effect', f'{truth["tau"].mean():.3f}')
st.caption('600 training units • 150 held-out units • five independent synthetic features • arbitrary outcome units')
tabs = st.tabs(['Data & Overview', 'Model Evaluation', 'Counterfactual Explorer',
                'Policy Simulator', 'Robustness & Sensitivity', 'Evidence Summary'])
with tabs[0]:
    st.subheader('Association is a starting point')
    st.write('Treatment assignment depends on features A and B, which also influence outcomes. '
             'The unadjusted outcome difference can therefore mix treatment effects with confounding.')
    st.plotly_chart(px.box(frame, x='Treatment', y='Observed synthetic outcome', color='Treatment'), width='stretch')
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
    st.dataframe(metrics, hide_index=True)
    st.plotly_chart(px.scatter(frame, x='Oracle synthetic effect', y='Estimated effect',
                              color='Treatment'), width='stretch')
    st.write('Root-PEHE measures effect prediction error; ATE error measures average-effect error. '
             'Lower is better. The oracle is known only because this is a simulation.')
    st.download_button('Download synthetic evaluation (CSV)', metrics.to_csv(index=False),
                       'causallens_synthetic_evaluation.csv', 'text/csv')
with tabs[2]:
    st.subheader('Inspect a fictional unit')
    unit = st.slider('Synthetic unit index', 0, len(frame)-1, 0)
    st.dataframe(frame.iloc[[unit]][FEATURES], hide_index=True)
    st.plotly_chart(px.bar(pd.DataFrame({'Quantity': ['Oracle control mean', 'Oracle treated mean'],
                                       'Synthetic outcome': [truth['mu0'][unit], truth['mu1'][unit]]}),
                          x='Quantity', y='Synthetic outcome'), width='stretch')
    st.metric('Estimated effect for this synthetic unit', f'{pred[unit]:.3f}')
    st.caption('Bars are known synthetic mean potential outcomes, not fitted predictions or observations '
               'of two real outcomes. No individual confidence interval is provided.')
with tabs[3]:
    st.subheader('Allocate a limited treatment budget')
    capacity = st.slider('Treatment capacity (%)', 0, 100, 40)
    policy = simulate_capacity_constrained_policy(truth['mu0'], truth['mu1'], pred, capacity/100)
    strategy = pd.DataFrame(policy['strategies']).T.rename_axis('Strategy').reset_index()
    st.dataframe(strategy, hide_index=True)
    st.plotly_chart(px.bar(strategy, x='Strategy', y='policy_value'), width='stretch')
    st.caption('Model-guided allocation prioritizes positive predicted effects within capacity. '
               'The oracle uses known synthetic effects. Treat All is an unconstrained comparator; '
               'its constrained regret is unavailable below 100% capacity. Random value is an expectation.')
with tabs[4]:
    st.subheader('Disagreement and prediction stress')
    disagreement = np.std(np.stack(list(predictions.values())), axis=0)
    st.plotly_chart(px.histogram(x=disagreement, labels={'x': 'Across-estimator effect standard deviation'}), width='stretch')
    noise = st.slider('Prediction noise standard deviation', 0.0, 3.0, 0.0, 0.1)
    stressed = pred + np.random.default_rng(904 + seed).normal(size=pred.size)*noise
    st.metric('Root-PEHE after synthetic prediction perturbation', f'{pehe_score(truth["tau"], stressed):.3f}')
    st.caption('This perturbs predictions only; it is not a hidden-confounding analysis or model refit. '
               'Disagreement is not a calibrated uncertainty interval. Neither diagnostic establishes causal validity.')
with tabs[5]:
    st.subheader('Evidence and limits')
    st.write('These are reproducible synthetic teaching examples, not scientific findings about IHDP or any population. '
             'The existing research results are preserved in the research workflow. Their public-display clearance '
             'has not been established, so they are withheld here.')
    st.write('Identification assumptions: consistency, no interference, adequate overlap, and adjustment for all '
             'treatment–outcome confounders. The toy generator makes its confounders observed; real studies need '
             'substantive justification. This dashboard is not a treatment recommendation.')
    evidence = json_safe({'data_kind': 'fully synthetic; no IHDP records', 'generator': GENERATOR_VERSION,
                         'seed': seed, 'estimator': name, 'train_units': 600, 'test_units': 150,
                         'evaluation': records, 'policy': policy,
                         'limits': 'Educational simulation only; no empirical benchmark or clinical claim.'})
    st.download_button('Download synthetic evidence (JSON)', json.dumps(evidence, indent=2, allow_nan=False),
                       'causallens_synthetic_evidence.json', 'application/json')
    st.caption('No dataset uploads, external data retrieval, credentials, or private files are used by this public demo.')
