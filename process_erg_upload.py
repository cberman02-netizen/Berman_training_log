"""
Runs inside the "Extract erg screenshot" GitHub Actions workflow — the
remote-upload counterpart to serve.py's /api/extract, for use when there's
no local server (e.g. uploading from a phone browser via the published site).

Expects photo(s) already committed to uploads/<activity_id>/ by the client
(the GitHub Contents API PUT that happens before this workflow is dispatched).
Runs local OCR on them and writes the result to data/pending/<activity_id>.json
for the client to poll for and pick up in its review UI — never writes
erg_captures.json directly, so nothing is saved without the user reviewing it
first, same as the local flow.

CLI:
    python process_erg_upload.py --activity-id 12345 [--device RowErg] [--multi-piece]
"""

import argparse
import glob
import json
import os
import sys

from extract_erg_local import extract_erg_data

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(REPO_ROOT, "uploads")
PENDING_DIR = os.path.join(REPO_ROOT, "data", "pending")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity-id", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--multi-piece", default="")
    args = ap.parse_args()

    activity_id = args.activity_id.strip()
    act_dir = os.path.join(UPLOADS_DIR, activity_id)
    paths = sorted(
        p for p in glob.glob(os.path.join(act_dir, "*")) if p.lower().endswith(IMAGE_EXTS)
    )
    if not paths:
        print(f"No photos found in {act_dir}", file=sys.stderr)
        sys.exit(1)

    device_hint = args.device.strip() or None
    multi_piece = args.multi_piece.strip().lower() in ("1", "true", "on", "yes")
    result = extract_erg_data(paths, device_hint=device_hint, multi_piece=multi_piece)

    data = result["data"]
    data["source_image"] = ", ".join(os.path.basename(p) for p in paths)

    os.makedirs(PENDING_DIR, exist_ok=True)
    out_path = os.path.join(PENDING_DIR, f"{activity_id}.json")
    with open(out_path, "w") as f:
        json.dump({"data": data, "warnings": result["warnings"]}, f, indent=1)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
