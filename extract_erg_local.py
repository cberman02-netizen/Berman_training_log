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


NO_COLON_TOTAL_TIME_RE = re.compile(r"^(\d{2})(\d{2})(\d)$")


def _parse_total_time_no_colon(token):
    # A longer total_time like "35:23.2" losing BOTH its colon and decimal point to
    # OCR reads as "35232" — distinct from _parse_split_no_colon's pattern because a
    # /500m split's minutes digit is (almost) always single-digit, while a workout's
    # total time routinely runs into double-digit minutes. Only ever used for the
    # header's total_time candidate, never for splits, and rejects an implausible
    # seconds field (>59). Also rejects an exact whole-minute-to-the-tenth reading
    # (seconds AND tenths both "0", e.g. "10000") — a real stopwatch total essentially
    # never lands there, but a round meters value like 10000m does, and without this
    # guard a token like that gets wrongly claimed as a time, hiding it from meters.
    normalized = _normalize_digits(token)
    m = NO_COLON_TOTAL_TIME_RE.match(normalized)
    if not m:
        return None
    mi, s, tenth = m.groups()
    if int(s) > 59:
        return None
    if s == "00" and tenth == "0":
        return None
    return int(mi) * 60 + int(s) + int(tenth) / 10


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
    return _crop_to_screen(gray)


def _resize_to(gray, target_dim):
    h, w = gray.shape
    scale = target_dim / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=interp)


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


# Real photos vary hugely in native crop resolution (a tight close-up of just the
# screen vs. a wider shot). Counter-intuitively, feeding Tesseract an ultra-high-res
# crop of this dot-matrix font often makes recognition WORSE — the gaps between a
# digit's individual dot segments become individually resolvable and read as noise
# rather than blending into a glyph — so try a couple of target resolutions per photo
# (on top of the existing threshold variants) and keep whichever combination scores
# best, rather than assuming higher resolution is always better.
TARGET_DIMS = (1600, 1200)


def _best_ocr(gray, top_n=3):
    # The single highest-scoring (variant, resolution) combo overall sometimes still
    # misses one specific row entirely (observed: a header row cleanly legible to the
    # eye that one combo's line-segmentation dropped, while two other combos scoring
    # only slightly lower read it fine) — so keep the top few candidates, not just the
    # winner, and let the caller fall back through them for header detection specifically.
    scored = []
    for target_dim in TARGET_DIMS:
        resized = _resize_to(gray, target_dim)
        variants = _preprocess_variants(resized)
        for name, img in variants.items():
            tokens = _ocr_variant(img)
            scored.append((_score(tokens), tokens, f"{name}@{target_dim}"))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(tokens, name) for _, tokens, name in scored[:top_n]]


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


HEADER_DOMINANCE_RATIO = 1.4  # a real header's meters total must clearly exceed any single row's


def _line_time_and_meters(line):
    # The PM5's summary/totals row is the one line with BOTH a time value and a
    # meters value together — the title line above it ("40:00") has time but no
    # meters, and the "Sep 05 2025" / column-label lines have neither. Returns
    # (has_both, meters_value) — meters_value is used to rank candidates, since
    # a plain INTERVAL row (or even a date line, where the year can look like a
    # meters reading) has exactly the same time+meters shape as a real header.
    has_time = False
    meters_val = None
    for t in line["tokens"]:
        if (_parse_time_to_sec(t["text"]) is not None or _parse_split_no_colon(t["text"]) is not None
                or _parse_total_time_no_colon(t["text"]) is not None):
            has_time = True
        v = _leading_int(t["text"])
        if v is not None and METERS_RANGE[0] <= v <= METERS_RANGE[1] and v > 60:
            if meters_val is None or v > meters_val:
                meters_val = v
    return has_time and meters_val is not None, meters_val


def _find_header_line(lines):
    # Real PM5 screens don't reliably make the summary row visually bigger than the
    # interval rows below it (only the workout title above it is) — so try content
    # first (a line with both a time and a meters value is almost certainly the
    # totals row) — but a plain interval row (or a date line, where the year can
    # read as a meters value) has the exact same shape, so when several lines
    # qualify, only trust the one whose meters value clearly dominates the rest
    # (a real total should dwarf any single segment) rather than just taking
    # whichever came first on screen. Falls back to the "clearly taller line"
    # heuristic (lesson #5) only when content matching finds nothing at all.
    digit_lines = [l for l in lines if any(c.isdigit() for c in l["text"])]
    pool = digit_lines or lines
    if not pool:
        return None

    candidates = []
    for line in pool:
        has_both, meters_val = _line_time_and_meters(line)
        if has_both:
            candidates.append((line, meters_val))

    if len(candidates) == 1:
        return candidates[0][0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p[1], reverse=True)
        top_line, top_meters = candidates[0]
        _, runner_up_meters = candidates[1]
        if top_meters >= HEADER_DOMINANCE_RATIO * runner_up_meters:
            return top_line
        return None  # ambiguous — safer to leave totals blank than guess wrong

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
    # A header row commonly carries TWO time-shaped values — total_time (usually
    # colon-intact, e.g. "45:03.9") and avg_split (often losing its colon to OCR,
    # e.g. "2106.0" for "2:06.0"). Merging both sources unconditionally (rather than
    # only falling back to the no-colon reading when NO strict match exists at all)
    # matters a lot here: with only the strict total_time token present, the old
    # single-candidate path couldn't tell split from total and silently kept neither.
    strict_time_toks = [(t, _parse_time_to_sec(t["text"])) for t in line["tokens"] if _parse_time_to_sec(t["text"]) is not None]
    strict_ids = {id(t) for t, _ in strict_time_toks}
    loose_time_toks = [
        (t, _parse_split_no_colon(t["text"])) for t in line["tokens"]
        if id(t) not in strict_ids and _parse_split_no_colon(t["text"]) is not None
    ]
    loose_ids = strict_ids | {id(t) for t, _ in loose_time_toks}
    # The PM5 header's column order is always time, meter, split, rate (left to
    # right) — so a no-colon total_time reading (which shares its digit-length
    # range with plenty of plausible meters values, e.g. "16168" reads equally
    # well as 16:16.8) is only trustworthy for the LEFTMOST token on the line,
    # never for one further right that the column order says should be meters.
    leftmost_tok = min(line["tokens"], key=lambda t: t["left"]) if line["tokens"] else None
    total_time_toks = [
        (t, _parse_total_time_no_colon(t["text"])) for t in line["tokens"]
        if id(t) not in loose_ids and t is leftmost_tok and _parse_total_time_no_colon(t["text"]) is not None
    ]
    time_toks = strict_time_toks + loose_time_toks + total_time_toks
    time_toks.sort(key=lambda p: p[1])

    if device == "RowErg" and time_toks:
        # The split is whichever candidate actually falls in a plausible /500m split
        # range — not just "the smaller of however many candidates we found". A lone
        # candidate outside that range (e.g. a 35-minute total_time_sec) is unambiguously
        # the total, not a "maybe-split" that blocks total_time_sec from being set too.
        split_candidates = [(t, v) for t, v in time_toks if SPLIT_SEC_RANGE[0] <= v <= SPLIT_SEC_RANGE[1]]
        split_tok = None
        if split_candidates:
            split_tok, split_val = split_candidates[0]
            out["avg_split_sec"] = round(split_val, 1)
            if split_tok["conf"] < LOW_CONF_WARN:
                warnings.append(_low_conf_note("avg_split_sec", split_tok))
        remaining = [(t, v) for t, v in time_toks if t is not split_tok]
        if remaining:
            total_tok, total_val = remaining[-1]
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


def _ocr_all_images(image_paths):
    # Each image carries its top few (candidate_lines) OCR readings, not just the
    # single best-scoring one — header detection falls back through them (see
    # _find_header_across_candidates) since the top overall scorer can still miss
    # one row that a close second/third candidate reads fine. Interval-row parsing
    # keeps using just the top candidate, which is already reliable for that.
    all_text_for_device = ""
    per_image = []
    for path in image_paths:
        gray = _load_gray(path)
        candidates = _best_ocr(gray)
        candidate_lines = [(_group_lines(tokens), name) for tokens, name in candidates]
        primary_lines, primary_name = candidate_lines[0]
        text = "\n".join(l["text"] for l in primary_lines)
        all_text_for_device += "\n" + text
        per_image.append((path, candidate_lines, text, primary_name))
    return per_image, all_text_for_device


def _find_header_across_candidates(candidate_lines):
    for lines, _name in candidate_lines:
        header_line = _find_header_line(lines)
        if header_line is not None:
            return header_line
    return None


def _detect_device_with_warning(all_text_for_device, device_hint, warnings):
    device = _detect_device(all_text_for_device, device_hint) or "RowErg"
    if not device_hint and device == "RowErg" and "500m" not in all_text_for_device.lower() and "s/m" not in all_text_for_device.lower():
        warnings.append("Could not confirm device type from the photo (no '/500m' or 's/m' text found) — defaulted to RowErg, please verify")
    return device


def _extract_single_workout(image_paths, device_hint, warnings, raw_text_parts):
    # Default mode: all photos are the SAME workout, possibly spanning a scrolled/
    # paginated interval table (lesson #2) — merge headers (first photo wins) and
    # dedupe overlapping rows across photos.
    per_image, all_text_for_device = _ocr_all_images(image_paths)
    device = _detect_device_with_warning(all_text_for_device, device_hint, warnings)

    header_fields = {}
    all_intervals = []
    any_header_found = False
    for path, candidate_lines, text, variant_used in per_image:
        raw_text_parts.append(f"--- {os.path.basename(path)} (variant: {variant_used}) ---\n{text}")

        # A photo may legitimately have no header (a continuation page of a scrolled
        # interval table) — that's normal, not worth a warning on its own.
        header_line = _find_header_across_candidates(candidate_lines)
        if header_line is not None:
            any_header_found = True
            parsed_header = _parse_header(header_line, device, warnings)
            for k, v in parsed_header.items():
                header_fields.setdefault(k, v)

        primary_lines = candidate_lines[0][0]
        intervals = _parse_table_rows(primary_lines, device, warnings)
        all_intervals.extend(intervals)

    if not any_header_found:
        warnings.append("Could not identify a summary header row in any photo — total time/meters/avg fields will need to be filled in manually")

    merged_intervals = _dedup_intervals(all_intervals)
    if not merged_intervals:
        warnings.append("No interval rows were confidently parsed — you'll likely need to fill these in by hand")

    data = {"device": device, "workout_type": "", **header_fields}
    if merged_intervals:
        data["intervals"] = merged_intervals

    if "total_time_sec" not in data:
        warnings.append("Could not read total_time_sec from any photo — please fill in manually")
    if "total_meters" not in data:
        warnings.append("Could not read total_meters from any photo — please fill in manually")

    return data


def _extract_multi_piece_workout(image_paths, device_hint, warnings, raw_text_parts):
    # Multi-piece mode: each photo is its OWN complete, self-contained piece — e.g.
    # a "2x20:00/4:00r" done as two separately-started PM5 sessions instead of one
    # interval-mode workout. Each photo's header becomes one rep in the combined
    # intervals list (never merged/deduped against another photo's header, since
    # they're genuinely different data, not overlapping fragments of one table),
    # and the top-level totals are summed/recomputed across all reps.
    per_image, all_text_for_device = _ocr_all_images(image_paths)
    device = _detect_device_with_warning(all_text_for_device, device_hint, warnings)

    intervals = []
    total_time_sec = 0.0
    total_meters = 0.0
    spms, watts_list, rpms = [], [], []
    have_time = have_meters = False

    for i, (path, candidate_lines, text, variant_used) in enumerate(per_image):
        rep = i + 1
        raw_text_parts.append(f"--- {os.path.basename(path)} (variant: {variant_used}) ---\n{text}")

        header_line = _find_header_across_candidates(candidate_lines)
        if header_line is None:
            warnings.append(f"Photo {rep}: could not identify a summary row for this piece — its numbers will need to be filled in manually")
            continue

        piece = _parse_header(header_line, device, warnings)
        row = {"label": str(rep)}
        if device == "RowErg":
            if "avg_split_sec" in piece:
                row["split_sec"] = piece["avg_split_sec"]
            if "avg_spm" in piece:
                row["spm"] = piece["avg_spm"]
                spms.append(piece["avg_spm"])
        else:
            if "avg_watts" in piece:
                row["watts"] = piece["avg_watts"]
                watts_list.append(piece["avg_watts"])
            if "avg_rpm" in piece:
                row["rpm"] = piece["avg_rpm"]
                rpms.append(piece["avg_rpm"])
        row["hr"] = None
        if "split_sec" in row or "watts" in row:
            intervals.append(row)
        else:
            warnings.append(f"Photo {rep}: found a summary row but couldn't read its split/watts — please fill in row {rep} by hand")

        if "total_time_sec" in piece:
            total_time_sec += piece["total_time_sec"]
            have_time = True
        if "total_meters" in piece:
            total_meters += piece["total_meters"]
            have_meters = True

        # Sub-splits within this piece (e.g. periodic markers inside a single 20:00
        # effort) are real data too — keep them, scoped under this rep's label so
        # they don't collide with another piece's "5:00" marker.
        primary_lines = candidate_lines[0][0]
        sub_lines = [l for l in primary_lines if l is not header_line]
        for sub in _parse_table_rows(sub_lines, device, warnings):
            sub["label"] = f"{rep}.{sub['label']}"
            intervals.append(sub)

    data = {"device": device, "workout_type": ""}
    if have_time:
        data["total_time_sec"] = round(total_time_sec, 1)
    if have_meters:
        data["total_meters"] = round(total_meters, 1)
    if have_time and have_meters and total_meters > 0:
        data["avg_split_sec"] = round(total_time_sec / (total_meters / 500), 1)
        if device == "RowErg":
            data["avg_watts"] = _watts_from_split(data["avg_split_sec"])
    if device == "RowErg" and spms:
        data["avg_spm"] = round(sum(spms) / len(spms), 1)
    if device == "BikeErg":
        if watts_list:
            data["avg_watts"] = round(sum(watts_list) / len(watts_list), 1)
        if rpms:
            data["avg_rpm"] = round(sum(rpms) / len(rpms), 1)

    if intervals:
        data["intervals"] = intervals
    else:
        warnings.append("No piece summaries were confidently parsed — you'll likely need to fill these in by hand")

    warnings.append(
        f"Multi-piece mode: total_time_sec/total_meters are the sum of all {len(image_paths)} photos' own "
        "readings and do NOT include rest between pieces — add rest time by hand if you want total_time_sec "
        "to reflect wall-clock duration."
    )
    return data


def extract_erg_data(image_paths, device_hint=None, multi_piece=False):
    """
    image_paths: file paths for one workout (1+ photos).
    device_hint: optional "RowErg" or "BikeErg" if the caller already knows it.
    multi_piece: False (default) — photos are the SAME workout, possibly a scrolled
        interval table spanning multiple screens; overlapping rows get deduped.
        True — each photo is its OWN separately-recorded piece (e.g. a "2x20:00/4:00r"
        done as two standalone PM5 sessions instead of one interval-mode workout);
        each photo's header becomes one rep, and totals are summed across photos.

    Returns {"data": {...erg_captures.json-shaped entry...}, "warnings": [...], "raw_text": "..."}
    """
    if not image_paths:
        raise ValueError("extract_erg_data requires at least one image path")

    warnings = []
    raw_text_parts = []

    if multi_piece:
        data = _extract_multi_piece_workout(image_paths, device_hint, warnings, raw_text_parts)
    else:
        data = _extract_single_workout(image_paths, device_hint, warnings, raw_text_parts)

    return {"data": data, "warnings": warnings, "raw_text": "\n\n".join(raw_text_parts)}


def main():
    ap = argparse.ArgumentParser(description="Extract erg data from photo(s) of one workout using local OCR only")
    ap.add_argument("images", nargs="+", help="image file(s) or glob pattern(s)")
    ap.add_argument("--device", choices=["RowErg", "BikeErg"], default=None)
    ap.add_argument("--multi-piece", action="store_true",
                     help="each photo is its own separately-recorded piece (e.g. 2x20:00/4:00r done as two standalone sessions), not a scrolled table of one workout")
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

    result = extract_erg_data(paths, device_hint=args.device, multi_piece=args.multi_piece)

    if args.debug:
        print(result["raw_text"])
        print("\n=== parsed ===")

    import json
    print(json.dumps({"data": result["data"], "warnings": result["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
