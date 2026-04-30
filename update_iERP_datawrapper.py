import json
import os
import re
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date
from dotenv import load_dotenv
load_dotenv()  # reads .env if it exists; does nothing in CI

# ---- Configuration ----
csv_path    = 'data.csv'
x_col       = 'X'
y_col       = 'Y'
DW_CHART_ID = 'XrtGp'
DW_API_KEY  = os.environ['DW_API_KEY']


def get_cape() -> float:
    """Fetch the current Shiller CAPE ratio from multpl.com."""
    url = "https://www.multpl.com/shiller-pe"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8")
    match = re.search(r'id="current"[^>]*>.*?([0-9]+\.?[0-9]*)', html, re.DOTALL)
    if not match:
        raise ValueError("Could not parse CAPE value from multpl.com")
    return float(match.group(1))


def get_10y_yield() -> float:
    """Fetch the current 10-year Treasury yield from Yahoo Finance (as a decimal, e.g. 0.043)."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=5d&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
    return price / 100


def update_datawrapper_annotation(chart_id, api_key, cape, bond_yield, ierp_lo, ierp_hi,
                                  x_pos, y_pos, annotation_index=0):
    """Update the text and position of one text-annotation on a Datawrapper chart and republish it."""
    today = date.today().strftime("%m/%d/%Y")
    text = (f"{today}\nCAPE: {cape:.1f}, "
            f"10Y: {bond_yield*100:.2f}%\n"
            f"iERP: {ierp_lo*100:.1f}% | {ierp_hi*100:.1f}%")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    get_req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{chart_id}",
        headers=headers,
    )
    with urllib.request.urlopen(get_req, timeout=10) as resp:
        chart = json.loads(resp.read().decode("utf-8"))

    annotations = chart.get("metadata", {}).get("visualize", {}).get("text-annotations", [])
    annotations[annotation_index]["text"] = text
    annotations[annotation_index]["position"]["x"] = str(round(x_pos, 4))
    annotations[annotation_index]["position"]["y"] = str(round(y_pos, 4))

    payload = json.dumps({"metadata": {"visualize": {"text-annotations": annotations}}}).encode("utf-8")

    patch_req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{chart_id}",
        data=payload,
        headers=headers,
        method="PATCH",
    )
    with urllib.request.urlopen(patch_req, timeout=10) as resp:
        resp.read()

    publish_req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{chart_id}/publish",
        data=b"",
        headers={"Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(publish_req, timeout=30) as resp:
            resp.read()
        print(f"Datawrapper chart {chart_id} annotation updated and published: {text}")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"Datawrapper chart {chart_id} annotation updated (publish blocked — "
                  f"republish manually or add chart:publish scope to your API token)")
        else:
            raise


# ---- Fit model ----
df = pd.read_csv(csv_path)[[x_col, y_col]].dropna()
X  = sm.add_constant(df[x_col])
model = sm.OLS(df[y_col], X).fit(cov_type='HAC', cov_kwds={'maxlags': 84})

# ---- Live market data ----
cape_live  = get_cape()
yield_live = get_10y_yield()
x_live     = np.log(1 / cape_live) - np.log(yield_live)
X_live     = sm.add_constant(np.array([x_live]), has_constant='add')
pred_live  = model.get_prediction(X_live).summary_frame(alpha=0.50)
y_live_mean = pred_live['mean'].iloc[0]
y_live_lo   = pred_live['obs_ci_lower'].iloc[0]
y_live_hi   = pred_live['obs_ci_upper'].iloc[0]
iERP_lo     = np.exp(y_live_lo - np.log(1 + yield_live / 100)) - 1 - yield_live / 100
iERP_hi     = np.exp(y_live_hi - np.log(1 + yield_live / 100)) - 1 - yield_live / 100
print(f"Live: CAPE={cape_live:.2f}, 10Y={yield_live*100:.2f}%  →  X={x_live:.4f}")
print(f"  iERP 50% interval: [{iERP_lo*100:.2f}%, {iERP_hi*100:.2f}%]")

# ---- Update Datawrapper ----
update_datawrapper_annotation(DW_CHART_ID, DW_API_KEY, cape_live, yield_live, iERP_lo, iERP_hi,
                              x_pos=x_live, y_pos=y_live_mean)
