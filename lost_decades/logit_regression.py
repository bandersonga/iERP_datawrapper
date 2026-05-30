from arch.bootstrap import StationaryBootstrap
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

x_current      = -0.65  # current X value
start_year     = 1935   # filter to observations from this year onward (None = use all)
wf_start_year  = 1990   # walk-forward out-of-sample testing begins here

df = pd.read_csv('lost_decades.csv')

# ---- Clean data ----
if start_year is not None:
    df = df[df['Year'] >= start_year]
y_arr    = np.asarray(df['LostDecade'], dtype=float)
sprd_arr = np.asarray(df['X'], dtype=float)
mask     = ~(np.isnan(y_arr) | np.isnan(sprd_arr))
y_arr    = y_arr[mask]
sprd_arr = sprd_arr[mask]
data     = np.column_stack([y_arr, sprd_arr])
print(f"Sample: {len(y_arr)} observations"
      + (f" (from {start_year})" if start_year else ""))


def logit_coef(data):
    y_b = data[:, 0]
    X_b = sm.add_constant(data[:, 1:2], has_constant='add')
    try:
        return sm.Logit(y_b, X_b).fit(disp=0).params
    except Exception:
        return np.array([np.nan, np.nan])


# ---- Full-sample model and fit metrics ----
X_full   = sm.add_constant(sprd_arr, has_constant='add')
model_full = sm.Logit(y_arr, X_full).fit(disp=0)
test_params = model_full.params
pred_full   = model_full.predict()
brier_full  = np.mean((pred_full - y_arr) ** 2)
brier_base  = np.mean((y_arr.mean() - y_arr) ** 2)   # naive always-predict-base-rate

print(f"Full-sample params: {test_params}")
print(f"\n--- In-sample fit metrics ---")
print(f"McFadden pseudo-R²: {model_full.prsquared:.3f}")
print(f"Brier score:        {brier_full:.3f}  (baseline = {brier_base:.3f})")


# ===========================================================
# METHOD 1: Stationary Bootstrap (mean block length = 144)
# ===========================================================
print("\n--- Stationary Bootstrap (mean block = 144 months) ---")
bs      = StationaryBootstrap(144, data, seed=42)
results = bs.apply(logit_coef, 1000)

valid               = ~np.isnan(results[:, 1])
intercept_sb        = results[valid, 0]
sprd_sb             = results[valid, 1]
prob_sb             = 1 / (1 + np.exp(-(intercept_sb + sprd_sb * x_current)))
prob_grid_sb        = 1 / (1 + np.exp(-(intercept_sb[:, None] + sprd_sb[:, None] *
                           np.linspace(sprd_arr.min(), sprd_arr.max(), 300))))

print(f"Valid fits: {valid.sum()} / 1000")
print(f"sprd coefficient: mean={sprd_sb.mean():.3f}")
print(f"90% CI: [{np.percentile(sprd_sb, 5):.3f}, {np.percentile(sprd_sb, 95):.3f}]")
print(f"Fraction negative: {(sprd_sb < 0).mean():.3f}")
print(f"At X = {x_current}:")
print(f"  P(lost decade): {prob_sb.mean():.1%}")
print(f"  50% CI: [{np.percentile(prob_sb, 25):.1%}, {np.percentile(prob_sb, 75):.1%}]")
print(f"  90% CI: [{np.percentile(prob_sb, 5):.1%}, {np.percentile(prob_sb, 95):.1%}]")


# ===========================================================
# METHOD 2: Non-overlapping subsampling (every nth obs)
# ===========================================================
step        = 84
print(f"\n--- Non-overlapping subsample (every {step}th month) ---")
data_no     = data[::step]
y_no        = data_no[:, 0]
sprd_no     = data_no[:, 1]
print(f"Non-overlapping sample: {len(y_no)} observations  "
      f"({y_no.mean():.1%} lost decades)")

params_no = logit_coef(data_no)
print(f"Params: {params_no}")

if not np.isnan(params_no[1]):
    prob_no = 1 / (1 + np.exp(-(params_no[0] + params_no[1] * x_current)))
    print(f"At X = {x_current}:  P(lost decade) = {prob_no:.1%}")
else:
    print("Logit did not converge on non-overlapping subsample "
          "(too few observations — use as directional check only)")
    prob_no = None


# ===========================================================
# METHOD 3: Walk-forward backtest (expanding window, point-in-time)
# ===========================================================
min_train       = 240   # n months minimum training history
wf_step         = 12
label_lag       = 144   # months until outcome window fully resolves (~12 years)
wf_start_idx    = (wf_start_year - start_year) * 12
print(f"\n--- Walk-forward backtest (expanding window, step = {wf_step} month(s), from {wf_start_year}) ---")
wf_preds        = []
wf_actuals      = []
wf_test_indices = []

# At time t, only train on observations whose outcome has fully resolved,
# i.e. starting months at least label_lag months before t.
for t in range(wf_start_idx, len(y_arr), wf_step):
    train_end = t - label_lag
    if train_end < min_train:
        continue
    params_wf = logit_coef(data[:train_end])
    if not np.isnan(params_wf[1]):
        for i in range(t, min(t + wf_step, len(y_arr))):
            p = 1 / (1 + np.exp(-(params_wf[0] + params_wf[1] * sprd_arr[i])))
            wf_preds.append(p)
            wf_actuals.append(y_arr[i])
            wf_test_indices.append(i)

wf_preds        = np.array(wf_preds)
wf_actuals      = np.array(wf_actuals)
wf_test_indices = np.array(wf_test_indices)
brier_wf   = np.mean((wf_preds - wf_actuals) ** 2)
brier_base_wf = np.mean((wf_actuals.mean() - wf_actuals) ** 2)
print(f"Out-of-sample observations: {len(wf_preds)}")
print(f"Brier score (walk-forward): {brier_wf:.3f}  (baseline = {brier_base_wf:.3f})")


def roc_auc(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    pos, neg = y_true.sum(), (1 - y_true).sum()
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for label in y_true:
        if label:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos)
        fpr.append(fp / neg)
    auc = float(np.trapezoid(tpr, fpr))
    return np.array(fpr), np.array(tpr), auc


# Restricted in-sample: full model predictions on the walk-forward window only.
# This makes the AUC comparison apples-to-apples — same population, different model fit.
pred_is_restricted = pred_full[wf_test_indices]
y_is_restricted    = y_arr[wf_test_indices]

fpr_is,   tpr_is,   auc_is   = roc_auc(y_arr, pred_full)
fpr_is_r, tpr_is_r, auc_is_r = roc_auc(y_is_restricted, pred_is_restricted)
fpr_wf,   tpr_wf,   auc_wf   = roc_auc(wf_actuals, wf_preds)
print(f"AUC in-sample (full, {start_year}–present):        {auc_is:.3f}")
print(f"AUC in-sample (restricted, same window): {auc_is_r:.3f}")
print(f"AUC walk-forward (point-in-time):        {auc_wf:.3f}")


# ===========================================================
# PLOTS
# ===========================================================
x_grid = np.linspace(sprd_arr.min(), sprd_arr.max(), 300)

# ---- Coefficient histogram ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(sprd_sb, bins=50, color='steelblue', alpha=0.75, label='Stationary bootstrap')
ax.axvline(test_params[1], color='#1f4e79', linewidth=2, label=f'Full-sample ({test_params[1]:.3f})')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('sprd coefficient', fontsize=12)
ax.set_title('Bootstrap distribution of sprd coefficient\n(Stationary Bootstrap, mean block = 144)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sb_coef_hist.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Logistic curve ----
prob_mean_sb  = prob_grid_sb.mean(axis=0)
prob_lo_90    = np.percentile(prob_grid_sb, 5,  axis=0)
prob_hi_90    = np.percentile(prob_grid_sb, 95, axis=0)
prob_lo_50    = np.percentile(prob_grid_sb, 25, axis=0)
prob_hi_50    = np.percentile(prob_grid_sb, 75, axis=0)

intercept_full, slope_full = test_params
prob_full = 1 / (1 + np.exp(-(intercept_full + slope_full * x_grid)))

pd.DataFrame({
    'x':           x_grid,
    'fit':         prob_full,
    'bootstrap':   prob_mean_sb,
    'ci90_lo':     prob_lo_90,
    'ci90_hi':     prob_hi_90,
    'ci50_lo':     prob_lo_50,
    'ci50_hi':     prob_hi_50,
}).to_csv('logit_curves.csv', index=False)
print("Wrote logit_curves.csv")

fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(x_grid, prob_lo_90, prob_hi_90, color='steelblue', alpha=0.12,
                label='90% bootstrap band')
ax.fill_between(x_grid, prob_lo_50, prob_hi_50, color='steelblue', alpha=0.25,
                label='50% bootstrap band')
ax.plot(x_grid, prob_mean_sb, color='steelblue', linewidth=1.5, linestyle='--',
        label='Bootstrap mean')
ax.plot(x_grid, prob_full, color='#1f4e79', linewidth=2.2, label='Full-sample fit')

""" # Non-overlapping curve (if it converged)
if not np.isnan(params_no[1]):
    prob_no_grid = 1 / (1 + np.exp(-(params_no[0] + params_no[1] * x_grid)))
    ax.plot(x_grid, prob_no_grid, color='darkorange', linewidth=1.8, linestyle='-.',
            label='Non-overlapping fit') """

# Data points with jitter
jitter = np.random.uniform(-0.02, 0.02, size=len(y_arr))
ax.scatter(sprd_arr, y_arr + jitter, alpha=0.25, color='#444444', s=15, zorder=3,
           label='Observed (0/1)')

# # Current value
# current_prob = prob_sb.mean()
# ax.axvline(x_current, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
#            label=f'Current X = {x_current}')
# ax.scatter([x_current], [current_prob], color='red', s=120, zorder=5,
#            edgecolor='white', linewidth=2)
# ax.annotate(f'  P = {current_prob:.1%}', xy=(x_current, current_prob),
#             fontsize=11, color='red', fontweight='bold')

ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('P(Lost Decade)', fontsize=12)
ax.set_title('Probability of a Lost Decade vs. X\n(Stationary Bootstrap)',
             fontsize=13)
#ax.set_title('Probability of a Lost Decade vs. X\n(Stationary Bootstrap vs. Non-overlapping)',
#             fontsize=13)
ax.set_ylim(-0.08, 1.08)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.7), fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sb_logit_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- ROC curve ----
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_is, tpr_is, color='#aaaaaa', linewidth=1.5,
        label=f'In-sample, full window (AUC = {auc_is:.2f})')
ax.plot(fpr_is_r, tpr_is_r, color='#1f4e79', linewidth=2,
        label=f'In-sample, restricted window (AUC = {auc_is_r:.2f})')
ax.plot(fpr_wf, tpr_wf, color='darkorange', linewidth=2, linestyle='--',
        label=f'Walk-forward, point-in-time (AUC = {auc_wf:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Lost Decade Prediction', fontsize=13)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Walk-forward predicted probability over time ----
wf_year_axis = start_year + wf_test_indices / 12
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(wf_year_axis, wf_preds, color='#1f4e79', linewidth=1.8,
        label='P(Lost Decade) — walk-forward')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)

# Shade actual lost-decade periods
in_ld = False
ld_start = None
for idx, (yr, act) in enumerate(zip(wf_year_axis, wf_actuals)):
    if act == 1 and not in_ld:
        ld_start = yr
        in_ld = True
    elif act == 0 and in_ld:
        ax.axvspan(ld_start, yr, color='#C00000', alpha=0.12)
        in_ld = False
if in_ld:
    ax.axvspan(ld_start, wf_year_axis[-1], color='#C00000', alpha=0.12)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('P(Lost Decade)', fontsize=12)
ax.set_title(f'Walk-forward Predicted Probability Over Time\n(out-of-sample from {wf_start_year}, trained through t–12yr)',
             fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wf_probability_series.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Calibration plot ----
n_bins = 6
bins = np.linspace(0, 1, n_bins + 1)
bin_centers, actual_freq, pred_mean_bin = [], [], []
for i in range(n_bins):
    mask_b = (wf_preds >= bins[i]) & (wf_preds < bins[i + 1])
    if mask_b.sum() >= 5:
        bin_centers.append(wf_preds[mask_b].mean())
        actual_freq.append(wf_actuals[mask_b].mean())
        pred_mean_bin.append(wf_preds[mask_b].mean())

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1, label='Perfect calibration')
ax.scatter(bin_centers, actual_freq, color='#1f4e79', s=80, zorder=4, label='Walk-forward')
ax.plot(bin_centers, actual_freq, color='#1f4e79', linewidth=1.5)
ax.set_xlabel('Predicted probability', fontsize=12)
ax.set_ylabel('Actual frequency', fontsize=12)
ax.set_title('Calibration Plot — Walk-forward predictions', fontsize=13)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=150, bbox_inches='tight')
plt.show()
