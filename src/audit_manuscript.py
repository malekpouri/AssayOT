"""Check every numeric claim in the manuscript against the committed result files.

    python -m src.audit_manuscript

The manuscript quotes roughly fifty numbers computed elsewhere, and its figures are
regenerated from data while the surrounding prose is typed by hand. That asymmetry is
how a paper drifts from its own results: this script caught the text saying 0.602
where the data, and the auto-generated figure beside it, said 0.601.

Exits non-zero on any mismatch, so it can gate a release.
"""
import sys, re, pathlib
sys.path.insert(0, '.')
import pandas as pd
from src import config as C

R = pathlib.Path('results')
fails, checks = [], 0

def chk(label, claimed, actual, tol=5e-4):
    global checks
    checks += 1
    ok = (actual is not None) and abs(claimed - actual) <= tol
    if not ok:
        fails.append(f"  MISMATCH  {label}: paper says {claimed}, data says {actual}")
    return ok

def ap(f, model):
    p = R / f
    if not p.exists(): return None
    d = pd.read_csv(p)
    d = d[(d.metric == 'ap') & (d.model == model)]
    return float(d.value.iloc[0]) if len(d) else None

# --- headline AP values quoted in abstract / results -----------------------
chk('guide L1 AP',      0.711, ap('metrics_guide.csv', 'L1_supervised'))
chk('guide L3 AP',      0.652, ap('metrics_guide.csv', 'L3_assay'))
chk('guide L2 AP',      0.597, ap('metrics_guide.csv', 'L2_nnpu'))
chk('guide CFD AP',     0.465, ap('metrics_guide.csv', 'baseline_cfd'))
chk('guide posw AP',    0.428, ap('metrics_guide.csv', 'baseline_position_weighted'))
chk('guide mmcount AP', 0.403, ap('metrics_guide.csv', 'baseline_mismatch_count'))
chk('valid L3 AP',      0.714, ap('metrics_validated.csv', 'L3_assay'))
chk('valid L1 AP',      0.602, ap('metrics_validated.csv', 'L1_supervised'))
chk('valid L2 AP',      0.612, ap('metrics_validated.csv', 'L2_nnpu'))
# the anchored control: the paper's principal negative result
chk('valid C1 AP',      0.750, ap('metrics_validated.csv', 'C1_supervised_anchor'))
chk('valid no-anchor AP', 0.605, ap('metrics_validated.csv', 'A_no_validation'))
chk('valid CFD AP',     0.537, ap('metrics_validated.csv', 'baseline_cfd'))
# reimplemented published architectures
chk('guide R1 AP',      0.660, ap('metrics_guide_reference.csv', 'R1_cnn_bilstm'))
chk('guide R2 AP',      0.659, ap('metrics_guide_reference.csv', 'R2_gbt_elevation'))
chk('valid R1 AP',      0.653, ap('metrics_validated_reference.csv', 'R1_cnn_bilstm'))
chk('valid R2 AP',      0.580, ap('metrics_validated_reference.csv', 'R2_gbt_elevation'))
# the abstract's CFD-margin range, which was briefly quoted wrongly as 0.18-0.21
_g = pd.read_csv(R / 'metrics_guide.csv'); _g = _g[_g.metric.eq('ap')]
_cfd_g = float(_g[_g.model.eq('baseline_cfd')].value.iloc[0])
_m = [float(_g[_g.model.eq(m)].value.iloc[0]) - _cfd_g
      for m in ('L1_supervised', 'L2_nnpu', 'L3_assay')]
chk('abstract CFD margin, reports lo', 0.13, min(_m), 0.005)
chk('abstract CFD margin, reports hi', 0.25, max(_m), 0.005)
chk('valid reads AP',   0.660, ap('metrics_validated.csv', 'baseline_changeseq_reads'))
chk('valid noanchor',   0.605, ap('metrics_validated.csv', 'A_no_validation'))
chk('study L1 AP',      0.851, ap('metrics_study.csv', 'L1_supervised'))
chk('study L3 AP',      0.826, ap('metrics_study.csv', 'L3_assay'))
chk('study pilot AP',   0.802, ap('metrics_study.csv', 'L3_assay_pilot_only'))
chk('study L2 AP',      0.780, ap('metrics_study.csv', 'L2_nnpu'))
chk('study CFD AP',     0.748, ap('metrics_study.csv', 'baseline_cfd'))
chk('tech L1 AP',       0.838, ap('metrics_tech.csv', 'L1_supervised'))
chk('tech L3 AP',       0.822, ap('metrics_tech.csv', 'L3_assay'))
chk('tech pilot AP',    0.803, ap('metrics_tech.csv', 'L3_assay_pilot_only'))
chk('tech CFD AP',      0.731, ap('metrics_tech.csv', 'baseline_cfd'))

# --- ablations (fold-matched) ---------------------------------------------
a = R / 'metrics_guide_ablation.csv'
if a.exists():
    d = pd.read_csv(a); d = d[(d.metric=='ap') & ~d.model.str.endswith('[report]')]
    g = dict(zip(d.model, d.value))
    chk('abl L3',            0.562, g.get('L3_assay'))
    chk('abl no-anchor',     0.605, g.get('A_no_validation'))
    chk('abl no-obs',        0.579, g.get('A_no_obs_branch'))
    chk('abl no-fp',         0.570, g.get('A_no_fp'))
    chk('abl free-prop',     0.564, g.get('A_free_prop'))
    chk('abl L2 (lowest)',   0.477, g.get('L2_nnpu'))

# --- floor sensitivity -----------------------------------------------------
f = C.PROC / 'floor_sensitivity.csv'
if f.exists():
    d = pd.read_csv(f)
    r0 = d[d.floor == 0].iloc[0]; r1 = d[d.floor == 0.001].iloc[0]
    chk('floor 0 gs_neg rate',   0.225, float(r0.gs_neg_rate), 1e-3)
    chk('floor 0.1% gs_neg',     0.009, float(r1.gs_neg_rate), 1e-3)
    chk('floor 0 both rate',     0.983, float(r0.both_rate),  1e-3)
    chk('floor 0.1% both rate',  0.539, float(r1.both_rate),  1e-3)
    chk('gs_neg n',              559,   int(r0.gs_neg_n), 0)
    chk('both n',                232,   int(r0.both_n), 0)

# --- semisynth -------------------------------------------------------------
s = R / 'metrics_semisynth.csv'
if s.exists():
    d = pd.read_csv(s); d = d[d.metric=='ap']
    pv = d.pivot_table(index='model', columns='mechanism', values='value')
    for m, rnd, sig in [('L1_supervised',0.336,0.280),('L2_nnpu',0.334,0.224),
                        ('L3_assay',0.303,0.261)]:
        chk(f'semisynth {m} random', rnd, float(pv.loc[m,'random']), 1e-3)
        # the absolute values the paper now quotes alongside the relative ones, so the
        # "degrades least" figure can never again stand without them

        chk(f'semisynth {m} signal', sig, float(pv.loc[m,'signal_dependent']), 1e-3)

# --- external measured-activity benchmark (TrueOT) -------------------------
tp = R / 'metrics_trueot.csv'
if tp.exists():
    chk('trueot C2 AP', 0.793, ap('metrics_trueot.csv', 'C2_anchor_unweighted'))
    chk('trueot C1 AP', 0.777, ap('metrics_trueot.csv', 'C1_supervised_anchor'))
    chk('trueot L1 AP', 0.745, ap('metrics_trueot.csv', 'L1_supervised'))
    chk('trueot L3 AP', 0.725, ap('metrics_trueot.csv', 'L3_assay'))
    chk('trueot R1 AP', 0.683, ap('metrics_trueot.csv', 'R1_cnn_bilstm'))
    chk('trueot CFD AP', 0.590, ap('metrics_trueot.csv', 'baseline_cfd'))
    s = pd.read_csv(R / 'trueot_summary.csv').iloc[0]
    chk('trueot guides', 25, int(s.n_guides), 0)
    chk('trueot rows', 1034, int(s.n_rows), 0)
    chk('trueot positives', 192, int(s.n_pos), 0)
    chk('trueot studies', 9, int(s.n_studies), 0)
    chk('trueot base rate', 0.189, float(s.macro_base_rate), 0.001)
    # the deviation the paper declares
    chk('trueot train batch', 1024, int(s.train_batch), 0)
cp = R / 'metrics_trueot_contrasts.csv'
if cp.exists():
    c = pd.read_csv(cp).set_index(['a', 'b'])
    chk('trueot anchor-vs-plain delta', 0.032,
        float(c.loc[('C1_supervised_anchor', 'L1_supervised'), 'delta']), 0.002)
    chk('trueot anchor-vs-plain p', 0.27,
        float(c.loc[('C1_supervised_anchor', 'L1_supervised'), 'p']), 0.03)
    chk('trueot weighting delta', 0.016,
        float(c.loc[('C2_anchor_unweighted', 'C1_supervised_anchor'), 'delta']), 0.002)

# --- does the two-estimand reversal replicate? -----------------------------
# The paper's most-promoted claim. Pinned in all three protocols so the direction
# cannot be misstated: reversed on the pilot's confirmatory assay, NOT reversed on the
# external benchmark.
def _l1_minus_l3(f):
    d = pd.read_csv(R / f); d = d[d.metric.eq('ap')]
    return (float(d[d.model.eq('L1_supervised')].value.iloc[0])
            - float(d[d.model.eq('L3_assay')].value.iloc[0]))
chk('reversal, reports',          0.058, _l1_minus_l3('metrics_guide.csv'), 0.002)
chk('reversal, pilot activity',  -0.113, _l1_minus_l3('metrics_validated.csv'), 0.002)
if (R / 'metrics_trueot.csv').exists():
    chk('reversal, external activity', 0.021, _l1_minus_l3('metrics_trueot.csv'), 0.002)

# --- identifiability relative spreads --------------------------------------
i = R / 'identifiability.csv'
if i.exists():
    d = pd.read_csv(i); d['rel'] = d.activity_sd_across_seeds / d.activity_mean
    m = dict(zip(d.config, d.rel))
    chk('ident constant rel',    0.36, m.get('constant propensity'), 0.01)
    chk('ident constrained rel', 0.44, m.get('constrained + anchor'), 0.01)
    chk('ident free rel',        0.59, m.get('free propensity + anchor'), 0.01)
    chk('ident no-anchor rel',   0.24, m.get('constrained, no anchor'), 0.01)
    # The paper's correction: these are per-row means over a 200,000-row evaluation
    # set, so the fit spread is ~150 nats in total, not "under 20%". Pinned so the
    # retracted framing cannot creep back in.
    n_eval = int(d.n_eval_rows.iloc[0]) if 'n_eval_rows' in d.columns else 200_000
    nats = d.report_loglik * n_eval
    chk('ident fit spread, nats', 150, float(nats.max() - nats.min()), 2)
    tri = dict(zip(d.config, nats))
    chk('ident capacity fit gain, nats', 57,
        float(tri['free propensity + anchor'] - tri['constant propensity']), 2)
    sd = dict(zip(d.config, d.activity_sd_across_seeds))
    chk('ident constant raw sd',    3.1e-4, sd.get('constant propensity'), 5e-6)
    chk('ident constrained raw sd', 3.7e-4, sd.get('constrained + anchor'), 5e-6)
    chk('ident free raw sd',        5.4e-4, sd.get('free propensity + anchor'), 5e-6)

# --- benchmark composition -------------------------------------------------
pairs = pd.read_parquet(C.PAIRS); obs = pd.read_parquet(C.OBS)
chk('pairs',        219640, len(pairs), 0)
chk('observations', 434994, len(obs), 0)
chk('seq pairs',    150697, pairs.seq_key.nunique(), 0)
chk('max repeat',    14833, int(pairs.seq_key.value_counts().iloc[0]), 0)
chk('bulge pairs',   10515, int(pairs.has_bulge.sum()), 0)
ext = obs[obs.study_id != 'Lazzarotto2020']
chk('ext studies',       45, ext.study_id.nunique(), 0)
chk('ext technologies',  22, ext.assay_type.nunique(), 0)
cand = pd.read_parquet(C.CANDIDATES, columns=['pair_id','source_set'])
chk('candidates',   6765936, len(cand), 0)
chk('pilot cands',  3385057, int((cand.source_set.astype(str)=='pilot').sum()), 0)

print(f"{checks} numeric claims checked")
if fails:
    print(f"\n{len(fails)} MISMATCHES:")
    print("\n".join(fails))
else:
    print("all match the committed data")

sys.exit(1 if fails else 0)
