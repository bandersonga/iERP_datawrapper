import math
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datawrapper import update_annotation
from market_data import get_cape, get_10y_yield

CHART_ID = "XrtGp"
API_KEY  = os.environ["DW_API_KEY"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")


def fit_model():
    df = pd.read_csv(CSV_PATH)[["X", "Y"]].dropna()
    X  = sm.add_constant(df["X"])
    return sm.OLS(df["Y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 84})


def main():
    model     = fit_model()
    cape      = get_cape()
    yield_10y = get_10y_yield()
    x_live    = math.log(1 / cape) - math.log(yield_10y)

    X_live  = sm.add_constant(np.array([x_live]), has_constant="add")
    pred    = model.get_prediction(X_live).summary_frame(alpha=0.50)
    y_mean  = pred["mean"].iloc[0]
    y_lo    = pred["obs_ci_lower"].iloc[0]
    y_hi    = pred["obs_ci_upper"].iloc[0]

    ierp_lo = np.exp(y_lo - math.log(1 + yield_10y / 100)) - 1 - yield_10y / 100
    ierp_hi = np.exp(y_hi - math.log(1 + yield_10y / 100)) - 1 - yield_10y / 100

    today    = date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"
    text     = (f"{date_str}\nCAPE: {cape:.1f}, "
                f"10Y: {yield_10y*100:.2f}%\n"
                f"iERP: {ierp_lo*100:.1f}% | {ierp_hi*100:.1f}%")

    update_annotation(CHART_ID, API_KEY, text, x_pos=x_live, y_pos=y_mean)

    print(f"CAPE={cape:.1f}  10Y={yield_10y*100:.2f}%  X={x_live:.4f}")
    print(f"iERP 50% interval: [{ierp_lo*100:.2f}%, {ierp_hi*100:.2f}%]")


if __name__ == "__main__":
    main()
