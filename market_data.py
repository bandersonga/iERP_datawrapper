import json
import re
import urllib.request


def get_cape() -> float:
    url = "https://www.multpl.com/shiller-pe"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8")
    match = re.search(r'id="current"[^>]*>.*?([0-9]+\.?[0-9]*)', html, re.DOTALL)
    if not match:
        raise ValueError("Could not parse CAPE value from multpl.com")
    return float(match.group(1))


def get_10y_yield() -> float:
    """Returns the 10-year Treasury yield as a decimal (e.g. 0.043 for 4.3%)."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=5d&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
    return price / 100