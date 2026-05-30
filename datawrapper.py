import json
import urllib.request
import urllib.error


def update_annotation(chart_id: str, api_key: str, text: str, x_pos: float, y_pos: float,
                      annotation_index: int = 0):
    """Update the text and position of one text-annotation on a Datawrapper chart and republish it."""
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
        print(f"Chart {chart_id} annotation updated and published.")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"Chart {chart_id} annotation updated (publish blocked — "
                  f"republish manually or add chart:publish scope to your API token).")
        else:
            raise