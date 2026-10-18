#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Generates the CLASSIFY task's label and info-card tables for alif_object_detection.

Reads a labels file (one class per line, in model output order) and an optional knowledge
base JSON (a list of objects with ``class_name``, ``description``, ``specs`` and
``led_status``), and writes a C++ source defining, in the given namespaces::

    g_classLabels[], g_numClassLabels        - the labels, as before
    g_classCardTitles[]                      - display names (first letter capitalised)
    g_classCardSummaries[]                   - up to two lines, "" when the KB has no entry
    g_backgroundClassIdx                     - index of the background label, or -1

Cards are matched by label name (case-insensitive), so any labels file works. Each summary
line is trimmed (at a clause boundary when possible) to fit the given pixel width with the
LVGL Montserrat font whose glyph advances are read from the font's C file.
"""
import argparse
import json
import re
from pathlib import Path

ASCII_FIRST = 32
ASCII_LAST = 126
DEFAULT_ADVANCE_PX = 9.0
SAFETY = 0.92
REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-", "\u00b0": "", "\u00d7": "x", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"', "\u00b5": "u", "\u2026": "...",
}


def load_advances(font_path: Path | None) -> dict[str, float]:
    """
    Read printable-ASCII glyph advance widths from an LVGL font C file.

    :param font_path:   Path to e.g. lv_font_montserrat_16.c, or None.
    :returns:           Map of character to advance in pixels (empty if unavailable).
    """
    if font_path is None or not font_path.is_file():
        return {}
    text = font_path.read_text(encoding="utf-8", errors="replace")
    advs = [int(m) for m in re.findall(r"\.adv_w\s*=\s*(\d+)", text)]
    # Glyph id 0 is reserved; ids 1..95 are ASCII 32..126 (adv_w is in 1/16 px).
    if len(advs) < ASCII_LAST - ASCII_FIRST + 2:
        return {}
    return {chr(c): advs[c - ASCII_FIRST + 1] / 16.0 for c in range(ASCII_FIRST, ASCII_LAST + 1)}


def text_width(text: str, advances: dict[str, float]) -> float:
    """
    Estimate the rendered width of a single line of text.

    :param text:        Line of ASCII text.
    :param advances:    Glyph advances from load_advances.
    :returns:           Width in pixels.
    """
    return sum(advances.get(ch, DEFAULT_ADVANCE_PX) for ch in text)


def to_ascii(text: str) -> str:
    """
    Map text to the printable ASCII the built-in Montserrat fonts carry.

    :param text:    Input text.
    :returns:       ASCII text with whitespace collapsed.
    """
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = "".join(ch if ASCII_FIRST <= ord(ch) <= ASCII_LAST else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(text: str) -> str:
    """
    Return the first sentence of a paragraph, without its final period.

    :param text:    Paragraph.
    :returns:       First sentence.
    """
    parts = re.split(r"(?<=[a-z0-9)])\.\s+(?=[A-Z0-9])", text.strip(), maxsplit=1)
    return parts[0].rstrip(". ")


def trim_words(text: str, max_px: float, advances: dict[str, float]) -> str:
    """
    Trim text at a word boundary and append an ellipsis so that it fits.

    :param text:        Candidate line.
    :param max_px:      Usable width in pixels.
    :param advances:    Glyph advances from load_advances.
    :returns:           Trimmed text, or "" if not even one word fits.
    """
    kept = ""
    for word in text.split(" "):
        cand = (kept + " " + word).strip()
        if text_width(cand + "...", advances) > max_px:
            break
        kept = cand
    return (kept.rstrip(",;:") + "...") if kept else ""


def trim_clauses(text: str, max_px: float, advances: dict[str, float]) -> str:
    """
    Keep as many leading comma-separated clauses of text as fit.

    :param text:        Candidate line.
    :param max_px:      Usable width in pixels.
    :param advances:    Glyph advances from load_advances.
    :returns:           Trimmed text, or "" if the first clause does not fit.
    """
    kept = ""
    for clause in text.split(", "):
        cand = (kept + ", " + clause) if kept else clause
        if text_width(cand, advances) > max_px:
            break
        kept = cand
    return kept


def fit_line(text: str, max_px: float, advances: dict[str, float]) -> str:
    """
    Trim text to one line: drop parentheticals, then trailing clauses, then words.

    A clause cut is used only if it keeps at least half the line, so a short first clause
    does not replace a more informative word-trimmed line.

    :param text:        Candidate line.
    :param max_px:      Usable width in pixels.
    :param advances:    Glyph advances from load_advances.
    :returns:           Text that fits in max_px.
    """
    if text_width(text, advances) <= max_px:
        return text
    text = re.sub(r"\s*\([^)]*\)", "", text)
    if text_width(text, advances) <= max_px:
        return text
    clauses = trim_clauses(text, max_px, advances)
    if clauses and text_width(clauses, advances) >= 0.5 * max_px:
        return clauses
    return trim_words(text, max_px, advances)


def make_summary(entry: dict, max_px: float, advances: dict[str, float]) -> str:
    """
    Build the two-line card summary: what it is, then its key LED meaning (or a spec).

    :param entry:       Knowledge-base object.
    :param max_px:      Usable width in pixels.
    :param advances:    Glyph advances from load_advances.
    :returns:           One or two lines separated by a newline.
    """
    lines = []
    desc = to_ascii(entry.get("description", ""))
    if desc:
        lines.append(fit_line(first_sentence(desc), max_px, advances))
    led = to_ascii(entry.get("led_status", ""))
    if led and not re.search(r"\b(no|not have)\s+(an?\s+)?LEDs?\b", led, re.I):
        head, _, rest = led.partition(":")
        if rest and len(head) <= 60:
            led = rest.strip()
        detail = "LED: " + first_sentence(led)
    else:
        spec = re.sub(r"^NVL72( compute tray)?:\s*", "", to_ascii(entry.get("specs", "")))
        detail = first_sentence(spec) if spec else ""
    if detail:
        lines.append(fit_line(detail, max_px, advances))
    return "\n".join(line for line in lines if line)


PAGE_FIELDS = [("Specs", "specs"), ("Troubleshooting", "troubleshooting"),
               ("LED status", "led_status"), ("Maintenance", "maintenance")]


def make_pages(entry: dict) -> str:
    """
    Build the knowledge pages shown by voice ("right" / "left") for one class.

    Pages are separated by a form feed; each page is its title, a newline, then one sentence per
    line. The card label wraps long lines itself.

    :param entry:   Knowledge-base object.
    :returns:       Pages joined by "\\f", or "" when the entry has none.
    """
    pages = []
    for title, key in PAGE_FIELDS:
        text = to_ascii(entry.get(key, ""))
        if not text:
            continue
        sentences = re.split(r"(?<=[a-z0-9)%])\.\s+(?=[A-Z0-9])", text)
        body = "\n".join(s.strip().rstrip(".") + "." for s in sentences if s.strip())
        pages.append(title + "\n" + body)
    return "\f".join(pages)


def c_string(text: str) -> str:
    """
    Quote text as a C string literal.

    :param text:    Text to quote.
    :returns:       C string literal.
    """
    return ('"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            .replace("\f", "\\f") + '"')


def read_labels(path: Path) -> list[str]:
    """
    Read a labels file the way CMake's file(STRINGS) does: one per line, blank lines skipped.

    :param path:    Labels file.
    :returns:       Labels in model output order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.rstrip("\r") for line in lines if line.strip()]


def main():
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--kb", type=Path, default=None, help="Knowledge-base JSON (optional)")
    parser.add_argument("--font", type=Path, default=None, help="LVGL font .c for widths")
    parser.add_argument("--width", type=float, default=440.0, help="Summary width in px")
    parser.add_argument("--background-label", default="background")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespaces", nargs="*", default=["arm", "app", "object_detection"])
    args = parser.parse_args()

    labels = read_labels(args.labels)
    entries = {}
    if args.kb is not None and args.kb.is_file():
        for entry in json.loads(args.kb.read_text(encoding="utf-8")):
            entries[str(entry.get("class_name", "")).strip().lower()] = entry
    advances = load_advances(args.font)
    max_px = args.width * SAFETY

    titles, summaries, pages = [], [], []
    background = -1
    for idx, label in enumerate(labels):
        name = to_ascii(label)
        if background < 0 and name.lower() == args.background_label.strip().lower():
            background = idx
        titles.append(name[:1].upper() + name[1:])
        entry = entries.get(name.lower())
        summaries.append(make_summary(entry, max_px, advances) if entry else "")
        pages.append(make_pages(entry) if entry else "")

    out = ["/* Generated by gen_class_cards.py - do not edit. */", ""]
    out += [f"namespace {ns} {{" for ns in args.namespaces]
    out.append("extern const char* const g_classLabels[] = {")
    out += [f"    {c_string(label)}," for label in labels]
    out.append("};")
    out.append(f"extern const int g_numClassLabels = {len(labels)};")
    out.append("extern const char* const g_classCardTitles[] = {")
    out += [f"    {c_string(title)}," for title in titles]
    out.append("};")
    out.append("extern const char* const g_classCardSummaries[] = {")
    out += [f"    {c_string(summary)}," for summary in summaries]
    out.append("};")
    out.append("extern const char* const g_classCardPages[] = {")
    out += [f"    {c_string(page)}," for page in pages]
    out.append("};")
    out.append(f"extern const int g_backgroundClassIdx = {background};")
    out += ["} /* namespace " + ns + " */" for ns in reversed(args.namespaces)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"gen_class_cards: {len(labels)} labels, {sum(1 for s in summaries if s)} cards, "
          f"background index {background}, font widths {'yes' if advances else 'no'}")


if __name__ == "__main__":
    main()
