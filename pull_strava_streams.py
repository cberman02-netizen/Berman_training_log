"""
Pull time-series streams (heart rate, watts, cadence, altitude, time)
for every activity in your training CSV, downsample them, and save
one small CSV per activity into a `streams/` folder.

=== ONE-TIME SETUP ===

1. Create a Strava API application:
   - Go to https://www.strava.com/settings/api
   - Fill in any name/website (can be dummy values, e.g. "My Training Analysis" / "http://localhost")
   - Note your CLIENT_ID and CLIENT_SECRET

2. Authorize your app to read your activity data:
   - Paste this URL into your browser (replace YOUR_CLIENT_ID):

     https://www.strava.com/oauth/authorize?client_id=YOUR_CLIENT_ID&response_type=code&redirect_uri=http://localhost&approval_prompt=force&scope=activity:read_all

   - Click "Authorize"
   - You'll be redirected to a broken localhost page — that's fine.
     Copy the "code" value out of the browser's URL bar, e.g.:
     http://localhost/?state=&code=THIS_PART_IS_YOUR_CODE&scope=...

3. Exchange that code for tokens by running this in a terminal
   (replace the placeholders):

     curl -X POST https://www.strava.com/oauth/token \
       -d client_id=YOUR_CLIENT_ID \
       -d client_secret=YOUR_CLIENT_SECRET \
       -d code=YOUR_CODE \
       -d grant_type=authorization_code

   This returns JSON with an "access_token" and "refresh_token".
   Fill in CLIENT_ID, CLIENT_SECRET, and REFRESH_TOKEN below —
   the script will use the refresh token to mint fresh access
   tokens automatically (access tokens expire after 6 hours,
   refresh tokens don't).

=== USAGE ===

    pip install requests --break-system-packages
    python pull_strava_streams.py

Reads activity IDs from your training CSV, pulls each activity's
streams, downsamples to one row per RESAMPLE_SECONDS, and writes
streams/stream_<activity_id>.csv. Safe to re-run — it skips
activities that already have a stream file.
"""

import csv
import time
import os
import requests

# ---- FILL THESE IN ----
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REFRESH_TOKEN = "YOUR_REFRESH_TOKEN"
# ------------------------

INPUT_CSV = "Training_Spreadsheet_-_strava_activities_2025-06-01_to_now.csv"
OUTPUT_DIR = "streams"
RESAMPLE_SECONDS = 10  # downsample streams to one row per N seconds
REQUESTS_PER_WINDOW = 90  # stay under Strava's 100/15min limit with buffer
WINDOW_SECONDS = 15 * 60 + 30  # 15 min + a small buffer

STREAM_KEYS = "time,heartrate,watts,cadence,altitude,velocity_smooth,distance"


def get_access_token():
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_streams(activity_id, access_token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": STREAM_KEYS, "key_by_type": "true"},
    )
    if resp.status_code == 429:
        return "rate_limited"
    if resp.status_code == 404:
        return None  # no streams available for this activity (e.g. manual entry)
    resp.raise_for_status()
    return resp.json()


def downsample_and_write(activity_id, streams, out_path):
    if not streams or "time" not in streams:
        return False

    time_data = streams["time"]["data"]
    n = len(time_data)

    fields = ["time", "heartrate", "watts", "cadence", "altitude", "velocity_smooth", "distance"]
    series = {f: streams[f]["data"] if f in streams else [None] * n for f in fields}

    rows = []
    last_written_t = None
    for i in range(n):
        t = time_data[i]
        if last_written_t is not None and t - last_written_t < RESAMPLE_SECONDS:
            continue
        rows.append([series[f][i] if i < len(series[f]) else None for f in fields])
        last_written_t = t

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_CSV) as f:
        activities = list(csv.DictReader(f))

    to_fetch = []
    for a in activities:
        out_path = os.path.join(OUTPUT_DIR, f"stream_{a['id']}.csv")
        if os.path.exists(out_path):
            continue
        to_fetch.append(a["id"])

    print(f"{len(activities)} activities total, {len(to_fetch)} need streams pulled.")

    access_token = get_access_token()
    window_start = time.time()
    requests_this_window = 0
    pulled, skipped, failed = 0, 0, 0

    for i, activity_id in enumerate(to_fetch):
        if requests_this_window >= REQUESTS_PER_WINDOW:
            elapsed = time.time() - window_start
            sleep_for = max(0, WINDOW_SECONDS - elapsed)
            print(f"  Rate limit pause: sleeping {sleep_for/60:.1f} min...")
            time.sleep(sleep_for)
            window_start = time.time()
            requests_this_window = 0
            access_token = get_access_token()  # refresh in case it expired

        streams = fetch_streams(activity_id, access_token)
        requests_this_window += 1

        if streams == "rate_limited":
            print("  Hit rate limit unexpectedly, sleeping 15 min...")
            time.sleep(WINDOW_SECONDS)
            window_start = time.time()
            requests_this_window = 0
            access_token = get_access_token()
            streams = fetch_streams(activity_id, access_token)
            requests_this_window += 1

        out_path = os.path.join(OUTPUT_DIR, f"stream_{activity_id}.csv")
        if streams is None:
            print(f"  [{i+1}/{len(to_fetch)}] {activity_id}: no streams available, skipping")
            skipped += 1
        elif downsample_and_write(activity_id, streams, out_path):
            pulled += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(to_fetch)}] pulled {pulled}, skipped {skipped}, failed {failed}")
        else:
            failed += 1

    print(f"\nDone. Pulled: {pulled}  Skipped (no data): {skipped}  Failed: {failed}")
    print(f"Stream files are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
