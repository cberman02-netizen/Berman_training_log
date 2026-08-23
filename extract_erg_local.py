"""
Extracts structured erg data (PM5 RowErg / BikeErg monitor screens) from one
or more photos, using local OCR only (Tesseract via pytesseract + OpenCV
preprocessing). No network calls, no API key, no per-image cost.

Tuned primarily for RowErg screens (split/rate), since that's the primary
use case. BikeErg (watt/rpm) parsing is included but less battle-tested.

Same interface as extract_erg.py, so serve.py can import either one:
    from extract_erg_local import extract_erg_data
    result = extract_erg_data(["IMG_1998.jpeg", "IMG_1999.jpeg"], device_hint="RowErg")

CLI:
    python extract_erg_local.py IMG_1998.jpeg IMG_1999.jpeg
    python extract_erg_local.py --device RowErg --debug photos/*.jpeg

Accuracy is meaningfully lower than the Anthropic vision version, especially
on angled/glare-heavy photos — treat every extraction as a draft to check in
the review UI, not a final answer. For best results, fill as much of the
frame as possible with the PM5 screen and avoid glare across the numbers.

Requires the `tesseract` binary on PATH (e.g. `brew install tesseract`).
"""

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

MIN_TOKEN_CONF = 40  # tesseract confidence (0-100) below which a token is ignored entirely
LOW_CONF_WARN = 65  # confidence below which we keep the value but flag a warning

TIME_HMS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[.,]?(\d)?")
TIME_MS_RE = re.compile(r"^(\d{1,3}):(\d{2})[.,]?(\d)?")
INT_RE = re.compile(r"^(\d{1,5})")

RATE_RANGE = (10, 42)          # plausible stroke rate (spm)
RPM_RANGE = (30, 130)          # plausible bike cadence
HR_RANGE = (70, 230)           # plausible heart rate
WATTS_RANGE = (20, 600)        # plausible bike watts
SPLIT_SEC_RANGE = (55, 420)    # plausible /500m split, as a time-token disambiguator
METERS_RANGE = (50, 200000)    # plausible total meters, as a bare-int disambiguator


# The PM5's small dot-matrix digits routinely get misread as visually similar
# letters, especially after the dilation pass that improves overall recognition
# (see _preprocess_variants) at the cost of thin punctuation like ":" — applied
# only as a fallback when the strict parse fails, and only to otherwise-numeric-
# looking tokens, so real words ("Sep", "View") are never touched.
_DIGIT_CONFUSION = str.maketrans({"i": "1", "I": "1", "l": "1", "|": "1", "o": "0", "O": "0"})
_MOSTLY_DIGITS_RE = re.compile(r"^[\dilIoO|:.,]+$")
NO_COLON_SPLIT_RE = re.compile(r"^(\d)(\d{2})[.,]?(\d)?$")


def _normalize_digits(text):
    return text.translate(_DIGIT_CONFUSION) if _MOSTLY_DIGITS_RE.match(text) else text


def _parse_time_to_sec_strict(token):
    m = TIME_HMS_RE.match(token)
    if m:
        h, mi, s, tenth = m.groups()
        return int(h) * 3600 + int(mi) * 60 + int(s) + (int(tenth) / 10 if tenth else 0)
    m = TIME_MS_RE.match(token)
    if m:
        mi, s, tenth = m.groups()
        return int(mi) * 60 + int(s) + (int(tenth) / 10 if tenth else 0)
    return None


def _parse_time_to_sec(token):
    val = _parse_time_to_sec_strict(token)
    if val is not None:
        return val
    normalized = _normalize_digits(token)
    return _parse_time_to_sec_strict(normalized) if normalized != token else None


def _parse_split_no_colon(token, allow_ambiguous=False):
    # A split like "1:57.5" with its colon dropped by OCR reads as "157.5" — still
    # unambiguous as M:SS.s given a plausible single-digit-minute rowing split. But a
    # bare 4-digit run with no decimal at all ("2041") is just as likely to be a meters
    # value, so by default only accept those without a decimal separator when they're
    # exactly 3 digits (a flat M:SS with no tenths, e.g. "157" for 1:57) — callers can
    # pass allow_ambiguous=True as a last resort when nothing safer is available.
    normalized = _normalize_digits(token)
    m = NO_COLON_SPLIT_RE.match(normalized)
    if not m:
        return None
    has_sep = "." in normalized or "," in normalized
    if not has_sep and len(normalized) != 3 and not allow_ambiguous:
        return None
    mi, s, tenth = m.groups()
    return int(mi) * 60 + int(s) + (int(tenth) / 10 if tenth else 0)


def _watts_from_split(split_sec):
    return round(2.80 / (split_sec / 500) ** 3, 1)


def _leading_int(text):
    m = INT_RE.match(text)
    if m:
        return int(m.group(1))
    normalized = _normalize_digits(text)
    m = INT_RE.match(normalized) if normalized != text else None
    return int(m.group(1)) if m else None


MIN_SCREEN_AREA_FRAC = 0.15  # a detected bright region smaller than this isn't trusted as "the screen"


def _crop_to_screen(gray):
    # The PM5's backlit LCD is much brighter than the gym/bezel around it in a typical
    # photo — find that bright region and crop to it so OCR isn't drowned in background
    # noise (equipment, floor, reflections). Falls back to the full frame if nothing
    # confidently screen-sized is found (e.g. a very tight photo of just the screen already).
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_SCREEN_AREA_FRAC * h * w:
        return gray

    x, y, bw, bh = cv2.boundingRect(biggest)
    pad_x, pad_y = round(bw * 0.02), round(bh * 0.02)
    x0, y0 = max(0, x + pad_x), max(0, y + pad_y)
    x1, y1 = min(w, x + bw - pad_x), min(h, y + bh - pad_y)
    return gray[y0:y1, x0:x1]


def _load_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _crop_to_screen(gray)
    h, w = gray.shape
    max_dim = 2200
    if max(h, w) < max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    return gray


def _preprocess_variants(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.medianBlur(enhanced, 3)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # The PM5's dot-matrix digits are made of small disconnected segments that read
    # fine to a human but confuse Tesseract's connected-component analysis. A light
    # dilate fuses each digit's dots into a single solid glyph, which measurably
    # improves recognition on real (non-synthetic) photos of the screen.
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(otsu_inv, kernel, iterations=1)
    otsu_inv_dilated = cv2.bitwise_not(dilated)

    return {"gray": enhanced, "otsu": otsu, "otsu_inv": otsu_inv, "otsu_inv_dilated": otsu_inv_dilated}


NUMERIC_TOKEN_RE = re.compile(r"^\d{1,3}[:.,]?\d{0,3}[:.,]?\d{0,3}$")


def _ocr_variant(img, psm=6):
    # No character whitelist: restricting to digits/punctuation collapses the natural
    # word-gaps around letters (e.g. "20s/m"), which merges whole rows into one token.
    # Letters come through fine here since parsing below only keeps numeric/time tokens.
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    tokens = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        tokens.append({
            "text": text, "conf": conf,
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
            "line_key": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        })
    return tokens


def _score(tokens):
    return sum(1 for t in tokens if t["conf"] >= MIN_TOKEN_CONF and NUMERIC_TOKEN_RE.match(t["text"]))


def _best_ocr(gray):
    variants = _preprocess_variants(gray)
    best_tokens, best_score, best_name = [], -1, None
    for name, img in variants.items():
        tokens = _ocr_variant(img)
        score = _score(tokens)
        if score > best_score:
            best_tokens, best_score, best_name = tokens, score, name
    return best_tokens, best_name


def _group_lines(tokens):
    by_line = {}
    for t in tokens:
        by_line.setdefault(t["line_key"], []).append(t)
    lines = []
    for key, toks in by_line.items():
        toks.sort(key=lambda t: t["left"])
        avg_height = sum(t["height"] for t in toks) / len(toks)
        top = min(t["top"] for t in toks)
        text = " ".join(t["text"] for t in toks)
        lines.append({"tokens": toks, "avg_height": avg_height, "top": top, "text": text})
    lines.sort(key=lambda l: l["top"])
    return lines


def _detect_device(all_text, device_hint):
    if device_hint:
        return device_hint
    lowered = all_text.lower()
    if "500m" in lowered or "s/m" in lowered:
        return "RowErg"
    if "watt" in lowered or "rpm" in lowered:
        return "BikeErg"
    return None


def _line_has_time_and_meters(line):
    # The PM5's summary/totals row is the one line with BOTH a time value and a
    # meters value together — the title line above it ("40:00") has time but no
    # meters, and the "Sep 05 2025" / column-label lines have neither.
    has_time = False
    has_meters = False
    for t in line["tokens"]:
        if _parse_time_to_sec(t["text"]) is not None or _parse_split_no_colon(t["text"]) is not None:
            has_time = True
        v = _leading_int(t["text"])
        if v is not None and METERS_RANGE[0] <= v <= METERS_RANGE[1] and v > 60:
            has_meters = True
    return has_time and has_meters


def _find_header_line(lines):
    # Real PM5 screens don't reliably make the summary row visually bigger than the
    # interval rows below it (only the workout title above it is) — so try content
    # first (a line with both a time and a meters value is almost certainly the
    # totals row), and only fall back to the "clearly taller line" heuristic
    # (lesson #5) when content matching finds nothing.
    digit_lines = [l for l in lines if any(c.isdigit() for c in l["text"])]
    pool = digit_lines or lines
    if not pool:
        return None

    for line in sorted(pool, key=lambda l: l["top"]):
        if _line_has_time_and_meters(line):
            return line

    if len(pool) < 2:
        return None
    candidate = max(pool, key=lambda l: l["avg_height"])
    others = [l["avg_height"] for l in pool if l is not candidate]
    median_other = sorted(others)[len(others) // 2]
    topmost_two = sorted(pool, key=lambda l: l["top"])[:2]
    if candidate["avg_height"] >= median_other * 1.15 and candidate in topmost_two:
        return candidate
    return None


def _low_conf_note(field, tok):
    return f"{field} read as '{tok['text']}' with low OCR confidence ({tok['conf']:.0f}%) — please verify"


def _parse_header(line, device, warnings):
    out = {}
    time_toks = [(t, _parse_time_to_sec(t["text"])) for t in line["tokens"] if _parse_time_to_sec(t["text"]) is not None]
    if not time_toks:
        # colon dropped by OCR entirely — fall back to the no-colon split reading
        time_toks = [(t, _parse_split_no_colon(t["text"])) for t in line["tokens"] if _parse_split_no_colon(t["text"]) is not None]
    time_toks.sort(key=lambda p: p[1])

    if device == "RowErg" and time_toks:
        split_tok, split_val = min(
            time_toks, key=lambda p: abs(p[1] - sum(SPLIT_SEC_RANGE) / 2)
        ) if len(time_toks) == 1 else time_toks[0]
        total_tok, total_val = time_toks[-1]
        if SPLIT_SEC_RANGE[0] <= split_val <= SPLIT_SEC_RANGE[1]:
            out["avg_split_sec"] = round(split_val, 1)
            if split_tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("avg_split_sec", split_tok))
        if total_val != split_val:
            out["total_time_sec"] = round(total_val, 1)
            if total_tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("total_time_sec", total_tok))
    elif time_toks:
        total_tok, total_val = time_toks[-1]
        out["total_time_sec"] = round(total_val, 1)
        if total_tok["conf"] < LOW_CONF_WARN:
            warnings.append(_low_conf_note("total_time_sec", total_tok))

    used_tops = {id(t) for t, _ in time_toks}
    bare_ints = [(t, _leading_int(t["text"])) for t in line["tokens"] if id(t) not in used_tops]
    bare_ints = [(t, v) for t, v in bare_ints if v is not None]

    meters_candidates = [(t, v) for t, v in bare_ints if METERS_RANGE[0] <= v <= METERS_RANGE[1] and v > 60]
    if meters_candidates:
        tok, val = max(meters_candidates, key=lambda p: p[1])
        out["total_meters"] = float(val)
        if tok["conf"] < LOW_CONF_WARN:
            warnings.append(_low_conf_note("total_meters", tok))

    if device == "RowErg":
        rate_candidates = [(t, v) for t, v in bare_ints if RATE_RANGE[0] <= v <= RATE_RANGE[1]]
        if rate_candidates:
            tok, val = rate_candidates[-1]
            out["avg_spm"] = float(val)
            if tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("avg_spm", tok))
        if "avg_split_sec" in out:
            out["avg_watts"] = _watts_from_split(out["avg_split_sec"])
    else:
        watts_candidates = [(t, v) for t, v in bare_ints if WATTS_RANGE[0] <= v <= WATTS_RANGE[1]]
        rpm_candidates = [(t, v) for t, v in bare_ints if RPM_RANGE[0] <= v <= RPM_RANGE[1]]
        if watts_candidates:
            tok, val = watts_candidates[0]
            out["avg_watts"] = float(val)
            if tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("avg_watts", tok))
        if rpm_candidates:
            tok, val = rpm_candidates[-1]
            out["avg_rpm"] = float(val)
            if tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("avg_rpm", tok))

    return out


def _parse_table_rows(lines, device, warnings):
    # Deliberately does NOT skip header_line: header detection is a best-effort guess
    # on real photos (see _find_header_line), and if it guessed wrong the line is a
    # real interval row — dropping it would silently lose data. A correctly-identified
    # header line just becomes one harmless extra row the user can delete on review.
    intervals = []
    for line in lines:
        # Drop pure-punctuation OCR noise (stray "|", "_", border artifacts) — every
        # real label/value token has a digit, and keeping noise tokens around shifts
        # which token gets treated as toks[0] (the label).
        toks = [t for t in line["tokens"] if any(c.isdigit() for c in t["text"])]
        if not toks:
            continue

        time_vals = [(t, _parse_time_to_sec(t["text"])) for t in toks if _parse_time_to_sec(t["text"]) is not None]
        bare_ints = [(t, _leading_int(t["text"])) for t in toks]
        bare_ints = [(t, v) for t, v in bare_ints if v is not None]
        if not time_vals and not bare_ints:
            continue

        first = toks[0]
        first_time = _parse_time_to_sec(first["text"])
        first_int = _leading_int(first["text"])
        label_is_guess = False
        if first_time is not None:
            label = first["text"]
            remaining_time_vals = [(t, v) for t, v in time_vals if t is not first]
        elif first_int is not None and first_int <= 99 and len(first["text"]) <= 2:
            label = first["text"]
            remaining_time_vals = time_vals
        else:
            # The label column (elapsed-time markers especially) is the least reliable
            # OCR target on real photos. Don't drop an otherwise-readable row over it —
            # keep the row with the raw (possibly garbled) text as a placeholder label
            # for the user to fix, and still try to harvest its split/rate/watts/hr.
            label = first["text"]
            remaining_time_vals = time_vals
            label_is_guess = True

        row = {"label": label}
        used_ids = {id(first)}

        if device == "RowErg":
            split_candidates = [(t, v) for t, v in remaining_time_vals if SPLIT_SEC_RANGE[0] <= v <= SPLIT_SEC_RANGE[1]]
            if not split_candidates:
                # Colon dropped by OCR entirely (common on the dilated variant) — "157.5"
                # is still unambiguous as 1:57.5 within a plausible rowing split range.
                # Try unambiguous readings (has a decimal, or exactly 3 digits) first;
                # only fall back to a bare 4-digit run (ambiguous with meters) if nothing
                # safer turned up anywhere else in the row.
                for allow_ambiguous in (False, True):
                    for t in toks:
                        if id(t) in used_ids:
                            continue
                        v = _parse_split_no_colon(t["text"], allow_ambiguous=allow_ambiguous)
                        if v is not None and SPLIT_SEC_RANGE[0] <= v <= SPLIT_SEC_RANGE[1]:
                            split_candidates = [(t, v)]
                            break
                    if split_candidates:
                        break
            if split_candidates:
                tok, val = split_candidates[0]
                row["split_sec"] = round(val, 1)
                used_ids.add(id(tok))
                if tok["conf"] < LOW_CONF_WARN:
                    warnings.append(_low_conf_note(f"interval {label} split_sec", tok))

            rate_candidates = [(t, v) for t, v in bare_ints if id(t) not in used_ids and RATE_RANGE[0] <= v <= RATE_RANGE[1]]
            if rate_candidates:
                tok, val = rate_candidates[0]
                row["spm"] = float(val)
                used_ids.add(id(tok))
                if tok["conf"] < LOW_CONF_WARN:
                    warnings.append(_low_conf_note(f"interval {label} spm", tok))
        else:
            watts_candidates = [(t, v) for t, v in bare_ints if id(t) not in used_ids and WATTS_RANGE[0] <= v <= WATTS_RANGE[1]]
            if watts_candidates:
                tok, val = watts_candidates[0]
                row["watts"] = float(val)
                used_ids.add(id(tok))
            rpm_candidates = [(t, v) for t, v in bare_ints if id(t) not in used_ids and RPM_RANGE[0] <= v <= RPM_RANGE[1]]
            if rpm_candidates:
                tok, val = rpm_candidates[0]
                row["rpm"] = float(val)
                used_ids.add(id(tok))

        hr_candidates = [(t, v) for t, v in bare_ints if id(t) not in used_ids and HR_RANGE[0] <= v <= HR_RANGE[1]]
        row["hr"] = float(hr_candidates[-1][1]) if hr_candidates else None

        if "split_sec" in row or "watts" in row:
            if label_is_guess:
                warnings.append(f"Could not confidently read the label for a row (split {row.get('split_sec')}) — shown as '{label}', please fix")
            intervals.append(row)

    return intervals


def _row_signature(row):
    return (row.get("label"), row.get("split_sec"), row.get("spm"), row.get("watts"), row.get("rpm"), row.get("hr"))


def _dedup_intervals(all_intervals):
    seen = set()
    merged = []
    for row in all_intervals:
        sig = _row_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(row)
    return merged


def extract_erg_data(image_paths, device_hint=None):
    """
    image_paths: list of file paths, all photos of ONE workout (1+ photos).
    device_hint: optional "RowErg" or "BikeErg" if the caller already knows it.

    Returns {"data": {...erg_captures.json-shaped entry...}, "warnings": [...], "raw_text": "..."}
    """
    if not image_paths:
        raise ValueError("extract_erg_data requires at least one image path")

    warnings = []
    header_fields = {}
    all_intervals = []
    raw_text_parts = []

    all_text_for_device = ""
    per_image = []
    for path in image_paths:
        gray = _load_gray(path)
        tokens, variant_used = _best_ocr(gray)
        lines = _group_lines(tokens)
        text = "\n".join(l["text"] for l in lines)
        all_text_for_device += "\n" + text
        per_image.append((path, lines, text, variant_used))

    device = _detect_device(all_text_for_device, device_hint) or "RowErg"
    if not device_hint and "RowErg" == device and "500m" not in all_text_for_device.lower() and "s/m" not in all_text_for_device.lower():
        warnings.append("Could not confirm device type from the photo (no '/500m' or 's/m' text found) — defaulted to RowErg, please verify")

    any_header_found = False
    for path, lines, text, variant_used in per_image:
        raw_text_parts.append(f"--- {os.path.basename(path)} (variant: {variant_used}) ---\n{text}")

        # A photo may legitimately have no header (a continuation page of a scrolled
        # interval table, lesson #2) — that's normal, not worth a warning on its own.
        header_line = _find_header_line(lines)
        if header_line is not None:
            any_header_found = True
            parsed_header = _parse_header(header_line, device, warnings)
            for k, v in parsed_header.items():
                header_fields.setdefault(k, v)

        intervals = _parse_table_rows(lines, device, warnings)
        all_intervals.extend(intervals)

    if not any_header_found:
        warnings.append("Could not identify a summary header row in any photo — total time/meters/avg fields will need to be filled in manually")

    merged_intervals = _dedup_intervals(all_intervals)
    if not merged_intervals:
        warnings.append("No interval rows were confidently parsed — you'll likely need to fill these in by hand")

    data = {
        "device": device,
        "workout_type": "",
        **header_fields,
    }
    if merged_intervals:
        data["intervals"] = merged_intervals

    if "total_time_sec" not in data:
        warnings.append("Could not read total_time_sec from any photo — please fill in manually")
    if "total_meters" not in data:
        warnings.append("Could not read total_meters from any photo — please fill in manually")

    return {"data": data, "warnings": warnings, "raw_text": "\n\n".join(raw_text_parts)}


def main():
    ap = argparse.ArgumentParser(description="Extract erg data from photo(s) of one workout using local OCR only")
    ap.add_argument("images", nargs="+", help="image file(s) or glob pattern(s)")
    ap.add_argument("--device", choices=["RowErg", "BikeErg"], default=None)
    ap.add_argument("--debug", action="store_true", help="print raw OCR text in addition to parsed result")
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

    if args.debug:
        print(result["raw_text"])
        print("\n=== parsed ===")

    import json
    print(json.dumps({"data": result["data"], "warnings": result["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
