# Training Log — Handoff Brief for Claude Code

## What this is

A personal training log (rowing/erg/bike/run) tracking toward an "Hour of
Power" test on Dec 25, 2026. It's currently a self-contained static HTML
site built from a Strava CSV export, per-activity stream data, and
manually-transcribed Concept2 PM5/BikeErg screenshot data. The site itself
is done and working. This handoff is about building a **new companion
tool**: a local app that lets the user upload a photo of their erg
monitor screen directly onto a specific workout, extract the numbers via
Claude's vision API, and write them into the site's data — replacing a
process that, up to now, a chat-based Claude did by hand, one screenshot
at a time.

## Current files (all in this handoff)

Verified working layout — `python3 build_site.py` runs successfully
against exactly this structure and produces a correct `index.html`
(confirmed: all 94 `erg_captures.json` entries merge in, matched 1:1 by
activity ID against the CSV):

```
training-log-handoff/
├── HANDOFF.md              (this file)
├── template.html           (site shell — edit this for UI changes)
├── training-log.html       (pre-built reference copy of the current site)
├── build_site.py           (reads data/, writes index.html)
├── pull_strava_streams.py  (fetches streams/ from Strava's API)
├── update.yml              (GitHub Actions workflow, daily rebuild)
└── data/
    ├── strava_activities_2025-06-01_to_now.csv   (master activity export)
    ├── erg_captures.json                          (PM5/BikeErg screen data)
    └── streams/            (305 per-activity time-series CSVs)
```

`build_site.py` expects this exact layout — it looks for
`data/strava_activities_*.csv` (glob match), `data/erg_captures.json`,
and `data/streams/stream_<id>.csv`, and writes `index.html` next to
itself using `template.html` as the shell. Don't flatten this structure.

- **`training-log.html`** — a pre-built reference copy of the site as of
  this handoff, so you can open it directly without running the build.
  Once you run `build_site.py` for real, its actual output filename is
  `index.html` (GitHub Pages convention) — rename/adjust as needed.
- **`template.html`** — the site shell. Contains a `/*__DATA__*/`
  placeholder that `build_site.py` replaces with the contents it
  generates. **This is the file to edit** if the UI needs to change —
  the built HTML is always a generated artifact, never hand-edit it.
- **`build_site.py`** — reads everything under `data/`, computes
  decoupling/HR-zone-minutes/etc., and writes the final HTML in one
  step (the template-merge is already inside this script — confirmed
  by the test run above).
- **`data/erg_captures.json`** — flat JSON, keyed by Strava activity ID
  (string). Holds the manually-extracted PM5/BikeErg screen data. Schema
  below. **This is the file the new upload tool needs to write into.**
- **`update.yml`** — GitHub Actions workflow, currently set up for a daily
  cron rebuild of the site from the CSV. Uses the same `data/` layout
  above — check it references `python3 build_site.py` correctly relative
  to wherever this ends up in the actual repo structure.
- **`pull_strava_streams.py`** — one-time/periodic script to pull
  time-series streams (HR, watts, cadence, altitude) per activity from
  Strava's API into `data/streams/`.
- **`data/strava_activities_2025-06-01_to_now.csv`** — the master
  activity CSV (cleaned; 11 duplicate Garmin/COROS entries already
  removed from the raw export).

## erg_captures.json schema

```json
{
  "<strava_activity_id>": {
    "source_image": "IMG_1234.jpeg",
    "total_time_sec": 3960.0,
    "total_meters": 14216.0,
    "workout_type": "4x15:00/1:30r",
    "device": "RowErg",              // or "BikeErg"
    "avg_split_sec": 126.6,          // RowErg only
    "avg_spm": 18.0,                 // RowErg only, not always present
    "avg_watts": 172.5,              // both device types
    "avg_rpm": 83.0,                 // BikeErg only
    "intervals": [                   // optional — only ~81 of 94 entries have this
      {"label": "1", "split_sec": 125.7, "hr": null, "spm": 20},
      {"label": "2", "split_sec": 126.3, "hr": null}
    ]
  }
}
```

Notes on the schema:
- `intervals[].label` is either an interval number ("1", "2"...) or an
  elapsed-time label for steady-state pieces with periodic splits
  ("5:00", "10:00"...).
- `intervals[].spm` (stroke rate) is only populated for 11 of the 94
  entries — it was added late and only backfilled for a subset. Not a
  blocker, just don't assume it's always there.
- For BikeErg, watts were read directly off the screen (no conversion
  needed). For RowErg, watts are derived from split via the Concept2
  formula (below) both at data-build time and live in the site's JS.
- `total_time_sec` on the PM5 includes rest periods between intervals;
  Strava's `duration_moving_min` in the CSV usually does not. Don't be
  surprised if they don't match exactly.

## Concept2 pace ⇄ watts formula

```python
watts = 2.80 / (split_seconds / 500) ** 3
split_seconds = 500 * (2.80 / watts) ** (1/3)
```

## Hard-won lessons from doing this extraction by hand (put these in the vision prompt)

1. **The PM5's on-screen clock is unreliable** — it drifts and
   periodically resets to a stale stored date. In the old workflow (bulk
   screenshots, no user-specified activity) this caused real mismatches;
   matching by computed watts against the CSV's `rowing_avg_watts_manual`
   field was far more reliable than trusting the date on screen. **This
   specific problem goes away** in the new design, since the user attaches
   a photo directly to a workout they've already opened — no
   date-matching needed. Just don't have the extractor try to infer or
   validate the date from the screen.

2. **Multi-screenshot sessions have overlapping rows.** The PM5 shows a
   paginated/rolling view of intervals — if a workout has more intervals
   than fit on one screen, two (or three) photos of it will have
   several rows in common at the seam. If the new tool supports
   uploading more than one photo per workout, it needs to detect and
   dedupe the overlap (match on identical split+rate+HR combinations)
   rather than concatenating and double-counting. This came up
   constantly in the manual extraction — session lengths of 16-22
   intervals routinely needed 2-3 photos.

3. **BikeErg screens use a different column layout** — `watt` and `rpm`
   columns instead of `/500m` split and `s/m` stroke rate. The header
   row on the PM5 tells you which mode you're in (look for "watt rpm" vs
   "/500m s/m"). Extraction logic needs to branch on this.

4. **HR is not always shown.** Some screens/rows have a heart column,
   some don't (strap not connected, or just not displayed on that
   screen mode). Don't require it.

5. **The screen has two zones**: a summary header row (bold/larger,
   shows totals for the whole piece — title, date, total time, total
   meters, avg split or avg watts, avg rate) and then a table of
   individual interval/split rows below it. Both matter — the header
   gives you `total_time_sec`, `total_meters`, `avg_split_sec`/`avg_watts`,
   `avg_spm`/`avg_rpm`; the rows give you `intervals[]`.

6. **Some interval workouts are logged as continuous steady pieces with
   periodic split markers** (e.g. a single 40:00 row broken into splits
   at 5:00, 10:00, 15:00...) rather than discrete reps. The header title
   distinguishes these ("40:00" vs "4x15:00/1:30r") — the extractor
   should preserve whichever structure is actually on screen rather than
   forcing everything into a fixed interval count.

7. Real photos have glare, extreme angles, and low light. Expect the
   model to occasionally misread a digit — see "review step" below.

## The three-piece architecture (agreed with the user)

1. **`extract_erg.py`** — a standalone module (importable, not just a
   CLI) that takes one or more image paths (for a single workout) plus
   optionally the known device type if the user knows it, calls the
   Anthropic API with a vision-capable model, and returns data in the
   `erg_captures.json` interval schema above. Should also be runnable
   standalone from the CLI for testing against a folder of images
   without going through the web UI.
   - API key read from an environment variable (e.g. `ANTHROPIC_API_KEY`),
     **never** hardcoded or committed.
   - If multiple photos are passed for one workout, apply the dedup
     logic from lesson #2 above.
   - Return both the parsed structured data AND enough raw info
     (e.g. what device type it detected, confidence caveats if the
     model expresses any) to support a review step.

2. **Local Flask app** (or similar — Flask/FastAPI, whatever Claude Code
   thinks fits) — runs locally only (`python app.py`, opened at
   `localhost:####`). Lets the user:
   - Browse/search their workouts (can reuse `data.json`/the CSV as the
     source — no need to duplicate the whole site's UI, just enough to
     find and open the right activity)
   - Upload one or more photos to a specific activity
   - Calls `extract_erg.py` directly as a function import (not a
     subprocess)
   - **Shows the extracted values back to the user for confirmation
     before writing** — recommended based on this project's history: a
     handful of real extraction errors were only caught by a human
     double-checking values against context (duration, expected pace)
     during manual work. Don't auto-commit blind.
   - On confirm, writes/updates the entry in `erg_captures.json` for that
     exact activity ID (no ID-matching needed — the user already chose
     the activity).

3. **The public site** — unchanged. Still `training-log.html` on GitHub
   Pages, still built by running `build_site.py` after `erg_captures.json`
   is updated, then committing and pushing. The local app never touches
   GitHub or the public site directly; it only edits the local
   `erg_captures.json`, which then flows through the existing pipeline.

## Security note

The Anthropic API key must only ever exist in the local environment
(env var) used by the Flask app / `extract_erg.py`. It must never appear
in `template.html`, `training-log.html`, or anything that ends up on
GitHub Pages — that's fully public. This was the whole reason for the
three-piece split instead of putting upload+extraction in the public
site directly.

## Open items / not yet decided

- GitHub repo + Pages hosting itself hasn't been set up yet as far as
  this conversation covered — confirm with the user whether a repo
  already exists or needs creating.
- Exact review/confirm UI for the Flask app (a form pre-filled with
  extracted values the user can edit before saving is probably simplest).
- Whether `update.yml`'s cron rebuild should also pick up
  `erg_captures.json` changes automatically, or whether the user will
  run `build_site.py` manually after using the local upload tool.
- Local app's port/framework choice, and whether it needs any
  persistence beyond directly editing `erg_captures.json`.
