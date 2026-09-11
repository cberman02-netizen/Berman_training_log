"""
Regenerates index.html from the latest Strava export + streams.

Expects, relative to repo root:
  data/strava_activities_2025-06-01_to_now.csv   (output of export_full_history.py)
  data/streams/stream_<id>.csv                   (output of export_streams.py)
  template.html                                  (site shell, with a /*__DATA__*/ placeholder)

Writes:
  index.html   (self-contained, ready for GitHub Pages)

Run this after both export scripts, e.g.:
  python export_full_history.py
  python export_streams.py
  python build_site.py
"""

import csv
import glob
import json
import os
import re
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
STREAMS_DIR = os.path.join(DATA_DIR, "streams")
TEMPLATE_PATH = os.path.join(REPO_ROOT, "template.html")
OUTPUT_PATH = os.path.join(REPO_ROOT, "index.html")

MAX_HR = 200
ZONE_BOUNDS = [0.60, 0.70, 0.80, 0.90, 1.01]
AEROBIC_KW = re.compile(r"UT2|UT1|\bZ2\b|\bZ1\b|steady", re.I)
M_TO_FT = 3.28084
MPS_TO_MPH = 2.23694

# ---- Edit this as your training phases change over time ----
PHASES = [
    {"start": "2025-08-01", "end": "2025-11-25", "label": "Base Building", "note": "Steady state + weekly hard piece, HR 150-160"},
    {"start": "2025-11-25", "end": "2025-12-25", "label": "HOP Block", "note": "6-week build to Hour of Power test \u2014 PR\u2019d"},
    {"start": "2025-12-26", "end": "2026-02-01", "label": "Transition", "note": "Coming off HOP block"},
    {"start": "2026-02-01", "end": "2026-05-15", "label": "Powerlifting Peak", "note": "3-4x/week lifting \u2014 385/275/475 total"},
    {"start": "2026-05-15", "end": "2026-12-25", "label": "HOP Buildup", "note": "Foundational aerobic base \u2192 threshold phase 8wks out"},
]
TARGET_EVENT = {"name": "Hour of Power", "date": "2026-12-25"}
# --------------------------------------------------------------


def find_master_csv():
    candidates = glob.glob(os.path.join(DATA_DIR, "strava_activities_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No strava_activities_*.csv found in {DATA_DIR}")
    return max(candidates, key=os.path.getmtime)


def load_stream(aid):
    path = os.path.join(STREAMS_DIR, f"stream_{aid}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def zone_minutes(rows):
    zones = [0.0] * 5
    prev_t = None
    for r in rows:
        if not r.get("heartrate") or not r.get("time"):
            continue
        t = float(r["time"])
        hr = float(r["heartrate"])
        if prev_t is not None:
            dt = t - prev_t
            if 0 < dt < 120:
                pct = hr / MAX_HR
                for zi, bound in enumerate(ZONE_BOUNDS):
                    if pct < bound:
                        zones[zi] += dt / 60.0
                        break
        prev_t = t
    return [round(z, 2) for z in zones]


def hr_drift(rows):
    hrs = [float(r["heartrate"]) for r in rows if r.get("heartrate")]
    if len(hrs) < 20:
        return None
    mid = len(hrs) // 2
    a1 = sum(hrs[:mid]) / mid
    a2 = sum(hrs[mid:]) / len(hrs[mid:])
    return round((a2 - a1) / a1 * 100, 1)


def power_hr_decoupling(rows):
    pts = [(float(r["heartrate"]), float(r["watts"])) for r in rows if r.get("heartrate") and r.get("watts") and float(r["watts"]) > 0]
    if len(pts) < 30:
        return None
    mid = len(pts) // 2
    first, second = pts[:mid], pts[mid:]
    r1 = sum(w for h, w in first) / sum(h for h, w in first)
    r2 = sum(w for h, w in second) / sum(h for h, w in second)
    return round((r2 - r1) / r1 * 100, 1)


# Matches "2x20:00/4:00r", "6x15:00/1:30r", "3x20'/2'r", "5x12'/80\"r" — the
# free-text notation this app's erg workout_type field already uses. Meter-
# based intervals ("15x500m/1:00r") deliberately don't match: without a fixed
# per-piece duration, there's no reliable way to say which stream seconds are
# work vs rest, so watts_per_beat() falls back to the whole-activity average
# HR for those instead of guessing.
INTERVAL_STRUCTURE_RE = re.compile(
    r"^(\d{1,2})\s*x\s*"
    r"(?:(\d{1,3}):(\d{2})|(\d{1,3})['’])"
    r"\s*/\s*"
    r"(?:(\d{1,3}):(\d{2})|(\d{1,3})['’]|(\d{1,3})[\"”])"
    r"\s*r",
    re.I,
)


def parse_interval_structure(workout_type):
    """('2x20:00/4:00r', ...) -> (reps=2, piece_sec=1200, rest_sec=240), or
    None if workout_type is blank or doesn't match the "NxD/Rr" notation."""
    if not workout_type:
        return None
    m = INTERVAL_STRUCTURE_RE.match(workout_type.strip())
    if not m:
        return None
    reps = int(m.group(1))
    piece_sec = int(m.group(2)) * 60 + int(m.group(3)) if m.group(2) is not None else int(m.group(4)) * 60
    if m.group(5) is not None:
        rest_sec = int(m.group(5)) * 60 + int(m.group(6))
    elif m.group(7) is not None:
        rest_sec = int(m.group(7)) * 60
    else:
        rest_sec = int(m.group(8))
    if reps < 1 or piece_sec <= 0:
        return None
    return reps, piece_sec, rest_sec


def erg_avg_watts(ec):
    """Average watts across an erg capture's intervals, matching the same
    piece-grouping as the JS ergIntervalAverages() in template.html: rows are
    grouped by piece (the part of the label before "."), averaged within
    each piece first, then averaged across pieces — so a piece entered as
    several sub-splits ("2.1".."2.5") counts once overall, the same as a
    piece entered as a single whole-piece row ("1")."""
    if ec.get("avg_watts") is not None:
        return ec["avg_watts"]
    intervals = ec.get("intervals") or []
    if not intervals:
        return None

    pieces = {}
    for iv in intervals:
        key = str(iv.get("label", "")).split(".")[0]
        pieces.setdefault(key, []).append(iv)

    def piece_watts(rows):
        vals = []
        for iv in rows:
            if iv.get("watts") is not None:
                vals.append(iv["watts"])
            elif ec.get("device") != "BikeErg" and iv.get("split_sec"):
                vals.append(2.80 / (iv["split_sec"] / 500) ** 3)
        return sum(vals) / len(vals) if vals else None

    per_piece = [v for v in (piece_watts(rows) for rows in pieces.values()) if v is not None]
    return sum(per_piece) / len(per_piece) if per_piece else None


def _windowed_avg_hr(rows, window_start, window_end):
    """Time-weighted average HR within [window_start, window_end) of elapsed
    stream seconds — same dt-weighting convention as zone_minutes (a sample's
    HR is attributed to the seconds since the previous sample)."""
    total_hr_dt, total_dt = 0.0, 0.0
    prev_t = None
    for r in rows:
        if not r.get("heartrate") or not r.get("time"):
            continue
        t = float(r["time"])
        hr = float(r["heartrate"])
        if prev_t is not None:
            dt = t - prev_t
            if 0 < dt < 120:
                overlap_start, overlap_end = max(prev_t, window_start), min(t, window_end)
                if overlap_end > overlap_start:
                    total_hr_dt += hr * (overlap_end - overlap_start)
                    total_dt += overlap_end - overlap_start
        prev_t = t
    return (total_hr_dt / total_dt) if total_dt > 0 else None


def watts_per_beat(ec, rows, activity_avg_hr):
    """Avg watts / avg HR for an indoor-rowing erg capture. When workout_type
    parses as an interval structure (e.g. "2x20:00/4:00r") AND the actual
    recorded duration is close to what that structure implies (no baked-in
    warmup/cooldown padding thrown off the assumed back-to-back timing), the
    HR average excludes rest — computed per work piece, then averaged across
    pieces equally, same weighting as erg_avg_watts. Otherwise falls back to
    the activity's own overall average HR from Strava."""
    avg_watts = erg_avg_watts(ec)
    if avg_watts is None:
        return None

    avg_hr = None
    structure = parse_interval_structure(ec.get("workout_type"))
    if structure and rows:
        reps, piece_sec, rest_sec = structure
        expected_total = reps * piece_sec + (reps - 1) * rest_sec
        last_t = rows[-1].get("time")
        actual_total = float(last_t) if last_t else None
        if actual_total is not None and abs(actual_total - expected_total) <= max(120, expected_total * 0.08):
            piece_avgs = []
            for i in range(reps):
                start = i * (piece_sec + rest_sec)
                end = start + piece_sec
                pa = _windowed_avg_hr(rows, start, end)
                if pa is not None:
                    piece_avgs.append(pa)
            if piece_avgs:
                avg_hr = sum(piece_avgs) / len(piece_avgs)

    if avg_hr is None:
        avg_hr = activity_avg_hr
    if not avg_hr:
        return None
    return round(avg_watts / avg_hr, 2)


def downsample_stream(rows, n=48):
    if not rows:
        return None
    L = len(rows)
    idx = sorted(set(int(i * (L - 1) / (n - 1)) for i in range(n))) if L > n else list(range(L))

    def safe(v):
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return None

    def safe_conv(v, factor):
        try:
            return round(float(v) * factor, 1)
        except (TypeError, ValueError):
            return None

    return {
        "t": [safe(rows[i]["time"]) for i in idx],
        "hr": [safe(rows[i]["heartrate"]) for i in idx],
        "alt": [safe_conv(rows[i].get("altitude"), M_TO_FT) for i in idx],  # Strava altitude is meters -> ft
        "w": [safe(rows[i]["watts"]) for i in idx],
        "v": [safe_conv(rows[i].get("velocity_smooth"), MPS_TO_MPH) for i in idx],  # m/s -> mph
    }


def f(x):
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return None


ERG_CAPTURES_PATH = os.path.join(DATA_DIR, "erg_captures.json")
TITLE_OVERRIDES_PATH = os.path.join(DATA_DIR, "title_overrides.json")
COMMUTE_OVERRIDES_PATH = os.path.join(DATA_DIR, "commute_overrides.json")


def build_data():
    """Computes the same `data` dict that gets embedded in index.html — reused
    by serve.py to render the live site with fresh data (including erg
    captures) on every request, without duplicating this logic."""
    with open(find_master_csv()) as fh:
        master = list(csv.DictReader(fh))

    erg_captures = {}
    if os.path.exists(ERG_CAPTURES_PATH):
        with open(ERG_CAPTURES_PATH) as fh:
            erg_captures = json.load(fh)

    title_overrides = {}
    if os.path.exists(TITLE_OVERRIDES_PATH):
        with open(TITLE_OVERRIDES_PATH) as fh:
            title_overrides = json.load(fh)

    commute_overrides = {}
    if os.path.exists(COMMUTE_OVERRIDES_PATH):
        with open(COMMUTE_OVERRIDES_PATH) as fh:
            commute_overrides = json.load(fh)

    activities = []
    for m in master:
        aid = m["id"]
        # start_date is UTC; a late-night activity can roll into the next UTC day,
        # so prefer start_date_local (the athlete's own wall-clock time) for the
        # calendar date everything in the app is bucketed by. Older rows synced
        # before this field existed fall back to start_date (UTC).
        dt = datetime.fromisoformat(m.get("start_date_local") or m["start_date"])
        dur = f(m["duration_moving_min"]) or 0
        spd = f(m["average_speed_mph"])
        entry = {
            "id": aid, "title": title_overrides.get(aid) or m["title"], "type": m["activity_type"],
            "desc": m.get("description") or "",
            "isCommute": bool(commute_overrides.get(aid)),
            "date": dt.strftime("%Y-%m-%d"), "ts": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "dur": dur, "dist": f(m["distance_miles"]), "elev": f(m["total_elevation_gain_ft"]),
            "hr": f(m["avg_hr"]), "maxhr": f(m["max_hr"]), "hrsd": f(m["hr_std"]),
            "watts": f(m["average_watts"]), "rowW": f(m.get("rowing_avg_watts_manual")),
            "cadence": f(m["average_cadence"]), "speed": spd if spd else None,  # 0 mph on indoor rowing is meaningless, treat as absent
            "gear": m["gear_name"] if m.get("gear_name") not in (None, "Not Available") else None,
            "effort": f(m.get("strava_relative_effort")),
        }

        rows = load_stream(aid)
        stream_has_watts = False
        if rows:
            entry["stream"] = downsample_stream(rows)
            entry["zones"] = zone_minutes(rows)
            stream_has_watts = any(r.get("watts") for r in rows)
            title = m["title"] or ""
            is_aerobic_row = m["activity_type"] == "Rowing" and AEROBIC_KW.search(title) and dur >= 20
            is_steady_ride = m["activity_type"] in ("Ride", "GravelRide") and stream_has_watts and dur >= 40
            if is_aerobic_row:
                d = hr_drift(rows)
                if d is not None:
                    entry["decoupling"] = {"value": d, "metric": "hr_drift"}
            elif is_steady_ride:
                d = power_hr_decoupling(rows)
                if d is not None:
                    entry["decoupling"] = {"value": d, "metric": "power_hr"}

        # Watts source: manual erg entry > real power-meter stream > Strava's own speed/grade estimate
        if entry["rowW"]:
            entry["wattsSource"] = "manual"
        elif stream_has_watts:
            entry["wattsSource"] = "meter"
        elif entry["watts"]:
            entry["wattsSource"] = "estimate"

        if aid in erg_captures:
            ec = erg_captures[aid]
            entry["ergCapture"] = ec
            if m["activity_type"] == "Rowing" and ec.get("device") != "BikeErg":
                wpb = watts_per_beat(ec, rows, entry["hr"])
                if wpb is not None:
                    entry["wattsPerBeat"] = wpb

        activities.append(entry)

    activities.sort(key=lambda a: a["ts"])

    return {
        "activities": activities,
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "phases": PHASES,
        "target_event": TARGET_EVENT,
        "max_hr": MAX_HR,
    }


def build():
    data = build_data()

    with open(TEMPLATE_PATH) as fh:
        template = fh.read()

    final = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))

    with open(OUTPUT_PATH, "w") as fh:
        fh.write(final)

    activities = data["activities"]
    print(f"Built {OUTPUT_PATH}: {len(activities)} activities, "
          f"{sum(1 for a in activities if 'stream' in a)} with streams, "
          f"{os.path.getsize(OUTPUT_PATH)/1024:.0f} KB")


if __name__ == "__main__":
    build()
