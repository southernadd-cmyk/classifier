#!/usr/bin/env python3
"""Prepare a small, balanced teaching sample from Blumenau's CommonsSpeech ZIP.

Source: https://reshare.ukdataservice.ac.uk/854292/1/speeches.zip
License: Open Parliament Licence v3.0; attribution in output and website.
Does not commit or publish the 1 GB source archive.
"""
import argparse
import csv
import html
import io
import json
import random
import re
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = "https://reshare.ukdataservice.ac.uk/854292/1/speeches.zip"
SEED = 20261119
# Four time periods, both parties. Reservoir pool is intentionally bigger than final sample.
PER_PERIOD = 1800
RESERVOIR = PER_PERIOD * 4
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
TAG_RE = re.compile(r"<[^>]*>")
SPACE_RE = re.compile(r"\s+")


def normalise_party(raw):
    text = SPACE_RE.sub(" ", str(raw or "").strip().lower())
    if re.match(r"^(lab|labour)(?:\b|[- ])", text):
        return "Labour"
    if re.match(r"^(con|conservative|tory)(?:\b|[- ])", text):
        return "Conservative"
    return None


def period_for_year(year):
    if year < 1979 or year > 2019:
        return None
    if year < 1990:
        return "1979-1989"
    if year < 2000:
        return "1990-1999"
    if year < 2010:
        return "2000-2009"
    return "2010-2019"


def clean_text(raw):
    text = html.unescape(TAG_RE.sub(" ", str(raw or "")))
    text = re.sub(r"\[[^\]]{0,120}\]", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def process_archive(archive, out, per_period=PER_PERIOD, max_per_mp=12):
    randomizer = random.Random(SEED)
    pools = defaultdict(list)
    seen = Counter()
    processed = Counter()
    valid_files = 0
    with zipfile.ZipFile(archive) as bundle:
        entries = [name for name in bundle.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
        if not entries:
            raise ValueError("Source ZIP has no CSV files.")
        for name in entries:
            with bundle.open(name) as binary:
                reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline=""))
                cols = {col.strip().lower(): col for col in (reader.fieldnames or []) if col}
                needed = {"body", "hdate"}
                if not needed.issubset(cols) or not ("party_short" in cols or "party" in cols):
                    print("Skip unrecognised CSV:", name, file=sys.stderr)
                    continue
                valid_files += 1
                for row in reader:
                    raw_party = row.get(cols.get("party_short", "")) or row.get(cols.get("party", ""))
                    party = normalise_party(raw_party)
                    if not party:
                        continue
                    date = (row.get(cols["hdate"]) or "").strip()
                    found = re.search(r"(?:19|20)\d\d", date)
                    if not found:
                        continue
                    year = int(found.group())
                    period = period_for_year(year)
                    if not period:
                        continue
                    speaker_id = (
                        row.get(cols.get("person_id", "")) or
                        row.get(cols.get("mnis_person_id", "")) or
                        row.get(cols.get("name", "")) or ""
                    ).strip()
                    if not speaker_id:
                        continue
                    body = clean_text(row.get(cols["body"], ""))
                    words = WORD_RE.findall(body)
                    if not (45 <= len(words) <= 500):
                        continue
                    if len(body) < 200:
                        continue
                    # Limit page weight and token count without publishing entire transcripts.
                    body = body[:1500].rsplit(" ", 1)[0]
                    key = (period, party)
                    processed[key] += 1
                    item = {"text": body, "party": party, "mp": speaker_id, "year": year}
                    count = processed[key]
                    pool = pools[key]
                    if len(pool) < per_period * 4:
                        pool.append(item)
                    else:
                        slot = randomizer.randrange(count)
                        if slot < len(pool):
                            pool[slot] = item
            print("Processed", name, "eligible rows so far:", sum(processed.values()), flush=True)
    if not valid_files:
        raise ValueError("No CommonsSpeech-compatible CSV found (expected body, hdate, party).")

    selected = []
    for period in ["1979-1989", "1990-1999", "2000-2009", "2010-2019"]:
        party_picks = {}
        for party in ("Labour", "Conservative"):
            pool = pools[(period, party)].copy()
            randomizer.shuffle(pool)
            by_speaker = Counter()
            picks = []
            for entry in pool:
                if by_speaker[entry["mp"]] >= max_per_mp:
                    continue
                by_speaker[entry["mp"]] += 1
                picks.append(entry)
                if len(picks) >= per_period:
                    break
            party_picks[party] = picks
        # Keep the two parties balanced within each decade, or omit an empty decade.
        n = min(len(party_picks["Labour"]), len(party_picks["Conservative"]))
        if n < 30:
            print("Not enough balanced speeches for period:", period, "count per party:", n, file=sys.stderr)
            continue
        for party in ("Labour", "Conservative"):
            selected.extend(party_picks[party][:n])
    if not selected:
        raise ValueError("No balanced Labour/Conservative sample could be prepared.")

    randomizer.shuffle(selected)
    by_party = Counter(r["party"] for r in selected)
    speakers = {party: len(set(r["mp"] for r in selected if r["party"] == party)) for party in by_party}
    if min(speakers.values()) < 8:
        raise ValueError("Not enough distinct MPs per party to split by speaker.")
    result = {
        "dataset": "House of Commons Parliamentary Debates 1979-2019",
        "credit": "Jack Blumenau (2021), UK Data Service, 10.5255/UKDA-SN-854292",
        "source": "https://reshare.ukdataservice.ac.uk/854292/",
        "license": "Open Parliament Licence v3.0",
        "sampling": f"Balanced reservoir by period and party (up to {per_period} per group, max {max_per_mp} per speaker per period); excludes speeches shorter than 45 or longer than 500 words.",
        "classes": ["Labour", "Conservative"],
        "counts": dict(by_party),
        "speakers": speakers,
        "examples": selected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("Saved", len(selected), "speeches across", sum(speakers.values()), "distinct party-MP identities to", out, flush=True)
    return result


def download_archive(destination):
    req = urllib.request.Request(SOURCE, headers={"User-Agent": "NeuralNetworkLab/1.0 (educational open-data project)"})
    print("Downloading original CommonsSpeech source (approx. 1 GB)...", flush=True)
    with urllib.request.urlopen(req, timeout=180) as resp, destination.open("wb") as out:
        if resp.status != 200:
            raise RuntimeError("Unexpected HTTP status " + str(resp.status))
        total = 0
        while True:
            chunk = resp.read(1024 * 1024 * 4)
            if not chunk:
                break
            total += len(chunk)
            out.write(chunk)
            if total // (1024 * 1024 * 100) != (total - len(chunk)) // (1024 * 1024 * 100):
                print("Downloaded", total // (1024 * 1024), "MiB", flush=True)
    if not zipfile.is_zipfile(destination):
        raise RuntimeError("Source was not a valid ZIP; refusing to publish dataset.")
    print("Download complete:", total, "bytes", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="Use an existing local source ZIP instead of downloading.")
    parser.add_argument("--out", type=Path, default=Path("data/commons-speech-sample.json"))
    parser.add_argument("--per-period", type=int, default=PER_PERIOD)
    args = parser.parse_args()
    csv.field_size_limit(50_000_000)
    if args.zip:
        process_archive(args.zip, args.out, args.per_period)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "speeches.zip"
            download_archive(archive)
            process_archive(archive, args.out, args.per_period)


if __name__ == "__main__":
    main()
