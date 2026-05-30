import json
import math
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_data import get_cape, get_10y_yield

CHART_ID  = "3NrLG"
API_KEY   = os.environ["DW_API_KEY"]

# Full-sample logit regression coefficients (1935–present, refit only on data revisions)
INTERCEPT = -1.015
SLOPE     = -3.485


def update_chart(x: float, prob: float, text: str):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{CHART_ID}",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        chart = json.loads(resp.read().decode("utf-8"))

    vis = chart["metadata"]["visualize"]

    ann = vis["text-annotations"][0]
    ann["text"]          = text
    ann["position"]["x"] = str(round(x, 4))
    ann["position"]["y"] = str(round(prob, 4))

    vis["range-annotations"][0]["position"]["x0"] = str(round(x, 4))

    payload = json.dumps({"metadata": {"visualize": vis}}).encode("utf-8")
    patch_req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{CHART_ID}",
        data=payload, headers=headers, method="PATCH",
    )
    with urllib.request.urlopen(patch_req, timeout=10) as resp:
        resp.read()

    publish_req = urllib.request.Request(
        f"https://api.datawrapper.de/v3/charts/{CHART_ID}/publish",
        data=b"", headers={"Authorization": f"Bearer {API_KEY}"}, method="POST",
    )
    try:
        with urllib.request.urlopen(publish_req, timeout=30) as resp:
            resp.read()
        print(f"Chart {CHART_ID} updated and published.")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"Chart {CHART_ID} updated (publish blocked — republish manually).")
        else:
            raise


def main():
    cape      = get_cape()
    yield_10y = get_10y_yield()
    x         = math.log(1 / cape) - math.log(yield_10y)
    prob      = 1 / (1 + math.exp(-(INTERCEPT + SLOPE * x)))
    prob_pct  = round(prob * 100)

    today    = datetime.now()
    date_str = f"{today.month}/{today.day}/{today.year}"
    text     = f"{date_str}\nProbability = {prob_pct}%"

    update_chart(x, prob, text)
    print(f"CAPE={cape:.1f}  10Y={yield_10y*100:.2f}%  X={x:.4f}  P={prob_pct}%")


if __name__ == "__main__":
    main()
