"""
Pulls any new Strava activities (since the most recent one already in the
master CSV), appends them, and pulls their time-series streams — the
on-demand, single-click counterpart to update.yml's daily rebuild.

Credentials come from training-log/.env (STRAVA_CLIENT_ID,
STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN) or the environment — never
hardcoded, and .env is gitignored. Strava can rotate the refresh token on
use; when it does, this rewrites .env so the next sync still works.

Importable:
    from sync_strava import sync_new_activities
    result = sync_new_activities()
    # {"added": 2, "activities": [{"id":..., "title":..., "date":...}], "capped": False}

CLI:
    venv/bin/python sync_strava.py
"""

import csv
import os
import statistics
import time
from datetime import datetime

import requests

from build_site import STREAMS_DIR, find_master_csv
from pull_strava_streams import downsample_and_write, fetch_streams

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(REPO_ROOT, ".env")

MAX_NEW_PER_SYNC = 25  # safety cap so one click can't turn into a huge, slow backfill
API_BASE = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"


def _load_env_file():
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _rewrite_env_value(key, value):
    lines, seen = [], False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    seen = True
                else:
                    lines.append(line if line.endswith("\n") else line + "\n")
    if not seen:
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


def get_access_token():
    _load_env_file()
    client_id = os.environ.get("STRAVA_CLIENT_ID")
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET")
    refresh_token = os.environ.get("STRAVA_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "Missing Strava credentials — set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, "
            "STRAVA_REFRESH_TOKEN in training-log/.env"
        )
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    data = resp.json()
    if data.get("refresh_token") and data["refresh_token"] != refresh_token:
        os.environ["STRAVA_REFRESH_TOKEN"] = data["refresh_token"]
        _rewrite_env_value("STRAVA_REFRESH_TOKEN", data["refresh_token"])
    return data["access_token"]


def _normalize_start_date(raw):
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def get_new_activity_ids(access_token, after_epoch, known_ids):
    ids = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/athlete/activities",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"after": after_epoch, "per_page": 100, "page": page},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for a in batch:
            if str(a["id"]) not in known_ids:
                ids.append(str(a["id"]))
        if len(batch) < 100:
            break
        page += 1
    return ids


def build_row(access_token, activity_id, fieldnames):
    resp = requests.get(f"{API_BASE}/activities/{activity_id}", headers={"Authorization": f"Bearer {access_token}"})
    resp.raise_for_status()
    d = resp.json()

    streams = fetch_streams(activity_id, access_token)
    hr_std = ""
    if streams and streams != "rate_limited" and "heartrate" in streams:
        hr_values = [v for v in streams["heartrate"]["data"] if v is not None]
        if len(hr_values) >= 2:
            hr_std = round(statistics.pstdev(hr_values), 1)

    row = {
        "id": str(d["id"]),
        "title": d.get("name") or "",
        "description": d.get("description") or "",
        "activity_type": d.get("sport_type") or d.get("type") or "",
        "start_date": _normalize_start_date(d["start_date"]),
        "duration_moving_min": round(d["moving_time"] / 60.0, 1) if d.get("moving_time") is not None else "",
        "duration_elapsed_min": round(d["elapsed_time"] / 60.0, 1) if d.get("elapsed_time") is not None else "",
        "distance_miles": round(d["distance"] / 1609.34, 2) if d.get("distance") is not None else "",
        "total_elevation_gain_ft": round(d["total_elevation_gain"] * 3.28084, 1) if d.get("total_elevation_gain") is not None else "",
        "avg_hr": d.get("average_heartrate", ""),
        "hr_std": hr_std,
        "max_hr": d.get("max_heartrate", ""),
        "average_watts": d.get("average_watts", ""),
        "weighted_average_watts": d.get("weighted_average_watts", ""),
        "max_watts": d.get("max_watts", ""),
        # Strava doesn't surface rowing ergometer power via the API — left blank here,
        # same as export_full_history.py; filled in via the erg-capture upload flow instead.
        "rowing_avg_watts_manual": "",
        "rowing_splits": "",
        "average_speed_mph": round(d["average_speed"] * 2.23694, 2) if d.get("average_speed") is not None else "",
        "max_speed_mph": round(d["max_speed"] * 2.23694, 2) if d.get("max_speed") is not None else "",
        "average_cadence": d.get("average_cadence", ""),
        "calories": d.get("calories", ""),
        "strava_relative_effort": d.get("suffer_score", ""),
        "hr_device_name": d.get("device_name") or "Not Available",
        "gear_name": (d.get("gear") or {}).get("name") or "Not Available",
    }
    return {k: row.get(k, "") for k in fieldnames}, streams


def sync_new_activities():
    access_token = get_access_token()

    master_path = find_master_csv()
    with open(master_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)
    known_ids = {r["id"] for r in existing_rows}

    after_epoch = 0
    if existing_rows:
        latest = max(datetime.fromisoformat(r["start_date"]) for r in existing_rows)
        after_epoch = int(latest.timestamp())

    all_new_ids = get_new_activity_ids(access_token, after_epoch, known_ids)
    capped = len(all_new_ids) > MAX_NEW_PER_SYNC
    new_ids = all_new_ids[:MAX_NEW_PER_SYNC]

    added = []
    os.makedirs(STREAMS_DIR, exist_ok=True)
    for aid in new_ids:
        row, streams = build_row(access_token, aid, fieldnames)
        existing_rows.append(row)
        added.append({"id": aid, "title": row["title"], "date": row["start_date"][:10]})

        if streams and streams != "rate_limited":
            downsample_and_write(aid, streams, os.path.join(STREAMS_DIR, f"stream_{aid}.csv"))

        time.sleep(1)  # be polite between activities

    if added:
        existing_rows.sort(key=lambda r: r["start_date"])
        with open(master_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

    return {"added": len(added), "activities": added, "capped": capped}


def main():
    result = sync_new_activities()
    if result["added"]:
        print(f"Added {result['added']} new activity(ies):")
        for a in result["activities"]:
            print(f"  {a['date']}  {a['title']}  ({a['id']})")
    else:
        print("No new activities — already up to date.")
    if result["capped"]:
        print(f"Hit the {MAX_NEW_PER_SYNC}-per-sync cap — run again to pull more.")


if __name__ == "__main__":
    main()
