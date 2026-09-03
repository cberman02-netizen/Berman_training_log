"""
Local dev server for the training log.

Serves the SAME template.html the public site uses, with live data (including
erg captures) computed fresh on every request via build_site.build_data() —
so localhost:5055 looks and behaves identically to training-log.html, with
several added capabilities, all wired up client-side in template.html and
only shown when /api/health succeeds (the static GitHub Pages build has no
backend and so never shows them): uploading/entering erg data, a "Sync
Strava" button, editable activity titles, and a commute checkbox per
activity.

Run:
    venv/bin/python serve.py
Then open http://127.0.0.1:5055

Never touches GitHub or the public site directly — it only reads data/ and
writes data/erg_captures.json. Run build_site.py yourself afterward (as
before) to regenerate index.html and push.
"""

import json
import os
from datetime import datetime

from flask import Flask, Response, jsonify, request

from build_site import TEMPLATE_PATH, build_data
from extract_erg_local import extract_erg_data
from sync_strava import sync_new_activities

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CAPTURES_DIR = os.path.join(DATA_DIR, "captures")
ERG_CAPTURES_PATH = os.path.join(DATA_DIR, "erg_captures.json")
TITLE_OVERRIDES_PATH = os.path.join(DATA_DIR, "title_overrides.json")
COMMUTE_OVERRIDES_PATH = os.path.join(DATA_DIR, "commute_overrides.json")

app = Flask(__name__)


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp_path, path)


def load_erg_captures():
    return _load_json(ERG_CAPTURES_PATH)


def save_erg_captures(captures):
    _save_json(ERG_CAPTURES_PATH, captures)


def load_title_overrides():
    return _load_json(TITLE_OVERRIDES_PATH)


def save_title_overrides(overrides):
    _save_json(TITLE_OVERRIDES_PATH, overrides)


def load_commute_overrides():
    return _load_json(COMMUTE_OVERRIDES_PATH)


def save_commute_overrides(overrides):
    _save_json(COMMUTE_OVERRIDES_PATH, overrides)


@app.route("/")
def index():
    data = build_data()
    with open(TEMPLATE_PATH) as fh:
        template = fh.read()
    html = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    return Response(html, mimetype="text/html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/extract", methods=["POST"])
def api_extract():
    activity_id = request.form.get("activity_id", "").strip()
    if not activity_id:
        return jsonify({"error": "missing activity_id"}), 400

    files = [f for f in request.files.getlist("photos") if f and f.filename]
    if not files:
        return jsonify({"error": "no photos uploaded"}), 400

    device_hint = request.form.get("device") or None
    multi_piece = request.form.get("multi_piece") in ("1", "true", "on")

    act_dir = os.path.join(CAPTURES_DIR, activity_id)
    os.makedirs(act_dir, exist_ok=True)
    saved_paths = []
    for f in files:
        dest = os.path.join(act_dir, f.filename)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(f.filename)
            dest = os.path.join(act_dir, f"{stem}_{datetime.now().strftime('%H%M%S')}{ext}")
        f.save(dest)
        saved_paths.append(dest)

    try:
        result = extract_erg_data(saved_paths, device_hint=device_hint, multi_piece=multi_piece)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    data = result["data"]
    data["source_image"] = ", ".join(os.path.basename(p) for p in saved_paths)
    return jsonify({"data": data, "warnings": result["warnings"]})


@app.route("/api/sync-strava", methods=["POST"])
def api_sync_strava():
    try:
        result = sync_new_activities()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


@app.route("/api/save-title", methods=["POST"])
def api_save_title():
    payload = request.get_json(force=True, silent=True) or {}
    activity_id = str(payload.get("activity_id", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not activity_id or not title:
        return jsonify({"error": "missing activity_id or title"}), 400

    overrides = load_title_overrides()
    overrides[activity_id] = title
    save_title_overrides(overrides)

    return jsonify({"ok": True, "title": title})


@app.route("/api/save-commute", methods=["POST"])
def api_save_commute():
    payload = request.get_json(force=True, silent=True) or {}
    activity_id = str(payload.get("activity_id", "")).strip()
    is_commute = bool(payload.get("is_commute"))
    if not activity_id:
        return jsonify({"error": "missing activity_id"}), 400

    overrides = load_commute_overrides()
    if is_commute:
        overrides[activity_id] = True
    else:
        overrides.pop(activity_id, None)
    save_commute_overrides(overrides)

    return jsonify({"ok": True, "is_commute": is_commute})


@app.route("/api/save", methods=["POST"])
def api_save():
    payload = request.get_json(force=True, silent=True) or {}
    activity_id = str(payload.get("activity_id", "")).strip()
    entry = payload.get("entry")
    if not activity_id or not isinstance(entry, dict):
        return jsonify({"error": "missing activity_id or entry"}), 400

    entry = {k: v for k, v in entry.items() if v is not None and v != ""}

    captures = load_erg_captures()
    captures[activity_id] = entry
    save_erg_captures(captures)

    return jsonify({"ok": True, "entry": entry})


if __name__ == "__main__":
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    app.run(debug=True, port=5055)
