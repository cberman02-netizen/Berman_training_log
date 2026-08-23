"""
Extracts structured erg data (PM5 RowErg / BikeErg monitor screens) from one
or more photos, using the Anthropic API's vision capability.

Importable:
    from extract_erg import extract_erg_data
    result = extract_erg_data(["IMG_1998.jpeg", "IMG_1999.jpeg"], device_hint="RowErg")

CLI (for testing against a folder of images without the web UI):
    python extract_erg.py IMG_1998.jpeg IMG_1999.jpeg
    python extract_erg.py --device RowErg photos/*.jpeg

Requires ANTHROPIC_API_KEY in the environment. Never hardcode it here.
"""

import argparse
import base64
import glob
import json
import os
import sys

import anthropic
from PIL import Image
import io

MODEL = "claude-sonnet-5"
MAX_EDGE_PX = 1568  # Anthropic's recommended max long-edge for vision input

SYSTEM_PROMPT = """You extract structured data from photos of Concept2 PM5 performance
monitor screens (RowErg or BikeErg). You will be given one or more photos that all belong
to the SAME workout. Read the screen(s) carefully and return ONLY a single JSON object
(no prose, no markdown fences) matching this exact schema:

{
  "device": "RowErg" | "BikeErg",
  "workout_type": string,          // the workout title/description as shown on screen, e.g. "4x15:00/1:30r" or "40:00"
  "total_time_sec": number,        // total elapsed time for the whole piece, including rest, in seconds
  "total_meters": number,
  "avg_split_sec": number | null,  // RowErg only: average /500m split in seconds. null for BikeErg.
  "avg_spm": number | null,        // RowErg only: average stroke rate. Omit/null if not shown.
  "avg_watts": number | null,      // both device types: average watts for the whole piece
  "avg_rpm": number | null,        // BikeErg only: average cadence. null for RowErg.
  "intervals": [
    {
      "label": string,             // interval number ("1","2"...) OR elapsed-time label for steady pieces with periodic splits ("5:00","10:00"...)
      "split_sec": number | null,  // RowErg /500m split for this row, in seconds. For BikeErg use null here and put pace info in watts/rpm instead.
      "watts": number | null,      // BikeErg row watts (RowErg rows generally don't show per-row watts on screen; leave null)
      "spm": number | null,        // RowErg stroke rate for this row, if shown
      "rpm": number | null,        // BikeErg cadence for this row, if shown
      "hr": number | null          // heart rate for this row, if shown
    }
  ]
}

Hard-won lessons — apply these:

1. Do NOT read or infer any date/timestamp from the screen. The PM5's on-screen clock
   drifts and is unreliable. The workout this belongs to is already known by the caller;
   your job is only the numbers.

2. If multiple photos are provided, they may be a paginated/rolling view of the SAME
   interval table — later photos often re-show several of the same rows as earlier
   photos (the overlap happens at the seam where the screen scrolled). Treat all photos
   together as one source, identify rows that are duplicates across photos (same label,
   same split/watts, same rate, same HR), and return each real interval only ONCE in the
   final "intervals" list, in the correct workout order. Do not double-count overlapping
   rows. Use the summary header (whichever photo has it, usually the first) for the
   total_time_sec/total_meters/avg_* fields.

3. Screen layout differs by device: RowErg headers show "/500m" and "s/m" columns
   (split and stroke rate). BikeErg headers show "watt" and "rpm" columns. Use the header
   row on screen to determine device — do not guess from context. Populate only the
   fields relevant to that device; leave the other device's fields null.

4. HR is not always shown (strap not connected, or that screen mode omits it). Leave it
   null rather than guessing.

5. Each screen has a summary header zone (bold/larger — title, total time, total meters,
   avg split or avg watts, avg rate) and a table of individual rows below it. Use the
   header for the top-level total_/avg_ fields and the table rows for "intervals".

6. Some workouts are logged as one continuous steady piece with periodic split markers
   (e.g. a single 40:00 effort with rows at 5:00, 10:00, 15:00...) rather than discrete
   reps with rest. The on-screen title distinguishes these ("40:00" vs "4x15:00/1:30r").
   Preserve whichever structure is actually on screen — use elapsed-time labels for
   steady pieces, rep numbers for discrete interval sets. Do not force a fixed count.

7. Photos may have glare, extreme angles, or low light and a digit may be genuinely
   illegible. If you are not confident about a specific number, still provide your best
   reading, but add a short note about it in a top-level "warnings" array of strings
   (e.g. "avg_watts on interval 3 was partly obscured by glare, low confidence"). If
   everything was clearly legible, return an empty "warnings" array.

Return ONLY the JSON object, with "warnings" as an additional top-level key alongside
"device"/"workout_type"/etc.
"""


def _load_and_resize(path):
    with open(path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / long_edge
        new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_erg_data(image_paths, device_hint=None):
    """
    image_paths: list of file paths, all photos of ONE workout (1+ photos).
    device_hint: optional "RowErg" or "BikeErg" if the caller already knows it.

    Returns:
        {
          "data": {...parsed erg_captures.json-shaped entry, minus source_image...},
          "warnings": [...],
          "raw_text": "...",   # raw model output, for debugging/review
        }
    Raises RuntimeError if the model's response isn't valid/parseable JSON.
    """
    if not image_paths:
        raise ValueError("extract_erg_data requires at least one image path")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")

    client = anthropic.Anthropic(api_key=api_key)

    content = []
    for p in image_paths:
        b64 = _load_and_resize(p)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })

    prompt_text = "Extract the erg data from the attached photo(s) of one workout."
    if device_hint:
        prompt_text += f" The user has indicated this is a {device_hint} workout."
    if len(image_paths) > 1:
        prompt_text += f" These {len(image_paths)} photos are of the SAME workout — dedupe overlapping rows as instructed."
    content.append({"type": "text", "text": prompt_text})

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = "".join(block.text for block in resp.content if block.type == "text").strip()

    parsed = _parse_json_response(raw_text)

    warnings = parsed.pop("warnings", [])
    return {"data": parsed, "warnings": warnings, "raw_text": raw_text}


def _parse_json_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\n\nRaw response:\n{raw_text}") from e


def main():
    ap = argparse.ArgumentParser(description="Extract erg data from photo(s) of one workout")
    ap.add_argument("images", nargs="+", help="image file(s) or glob pattern(s)")
    ap.add_argument("--device", choices=["RowErg", "BikeErg"], default=None)
    args = ap.parse_args()

    paths = []
    for pattern in args.images:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"File(s) not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    result = extract_erg_data(paths, device_hint=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
