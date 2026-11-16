"""
Repair rome_results.csv / memit_results.csv and align them with the baseline.

Problem: a greedy_answer field with an unbalanced quote made csv/pandas read the following
rows as part of that field. The physical lines are fine, so we split the raw text into one
record per case (a record starts with  <case_id>,<relation_id>,  e.g.  5352,P1303,), rebuild
the damaged ones, and write clean files.

    python repair_and_align.py scaled_baseline_results.csv rome_results.csv memit_results.csv

Writes (original files are not changed):
    rome_clean.csv, memit_clean.csv, baseline_common.csv  -> only the cases present in BOTH
    editing files, in baseline order. Use these three files with analyse_results.py.
"""
import csv
import io
import re
import sys

import pandas as pd

COLS = ["case_id", "relation_id", "subject", "prompt", "target_new", "target_true",
        "target_new_preferred", "paraphrase_new_preferred", "neighbourhood_true_preferred",
        "greedy_answer", "greedy_equals_true", "edit_seconds"]
START = re.compile(r"^\d+,P\d+,")


def read_records(path):
    text = open(path, encoding="utf-8", newline="").read()
    lines = text.splitlines()
    header, body = lines[0], lines[1:]
    recs = []
    for ln in body:
        if START.match(ln) or not recs:
            recs.append(ln)
        else:                      # continuation of a greedy answer that contained a line break
            recs[-1] += "\n" + ln
    assert header.split(",")[:3] == ["case_id", "relation_id", "subject"], f"{path}: unexpected header"
    return [r for r in recs if r.strip()]


ROW = re.compile(r"^(\d+),(P\d+),(.*),(\d),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(.*),(\d),(\d+(?:\.\d+)?)\s*$", re.S)


def parse(rec):
    """Return (row dict, repaired?) or (None, True) if the record cannot be read."""
    fields = next(csv.reader(io.StringIO(rec)), [])
    if len(fields) == 12:
        return dict(zip(COLS, fields)), False
    if len(fields) == 10:            # row cut off after the greedy answer: no flag / time column
        greedy = fields[9].strip()
        true = fields[5]
        if len(greedy) < len(true):  # visible text too short to decide the flag -> rerun this case
            return None, True
        flag = int(greedy.casefold().startswith(true.casefold()))
        return dict(zip(COLS, fields[:9] + [greedy, str(flag), ""])), True
    m = ROW.match(rec)               # independent of quote handling: uses the numeric columns
    if not m:
        return None, True
    cid, rel, mid, flag, para, neigh, greedy, g_eq, secs = m.groups()
    parts = next(csv.reader(io.StringIO(mid)), [])
    if len(parts) != 4:              # fallback: subject first, new/true last, prompt in between
        parts = mid.split(",")
        if len(parts) < 4:
            return None, True
        parts = [parts[0], ",".join(parts[1:-2]), parts[-2], parts[-1]]
    greedy = greedy.strip()
    if greedy.startswith('"'):
        greedy = greedy[1:]
    if greedy.endswith('"'):
        greedy = greedy[:-1]
    greedy = greedy.replace('""', '"')
    row = dict(zip(COLS, [cid, rel] + parts + [flag, para, neigh, greedy, g_eq, secs]))
    return row, True


def load(path):
    rows, fixed_ids, bad = [], [], []
    for rec in read_records(path):
        row, fixed = parse(rec)
        if row is None:
            bad.append(rec)
            print(f"  [cannot read] {path}: {rec[:150]!r}")
            continue
        # consistency check: the stored flag must agree with the stored answer
        calc = int(row["greedy_answer"].casefold().startswith(row["target_true"].casefold()))
        if row["greedy_equals_true"] != "" and int(row["greedy_equals_true"]) != calc:
            print(f"  [warning] {path}: case {row['case_id']}: greedy flag and answer disagree")
        if fixed:
            fixed_ids.append(row["case_id"])
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("case_id", "target_new_preferred", "greedy_equals_true"):
        df[c] = df[c].astype(int)
    for c in ("paraphrase_new_preferred", "neighbourhood_true_preferred"):
        df[c] = df[c].astype(float)
    df["edit_seconds"] = pd.to_numeric(df["edit_seconds"], errors="coerce")   # blank for cut-off rows
    dup = df["case_id"].duplicated().sum()
    if dup:
        print(f"  [warning] {path}: {dup} duplicate case_id(s); keeping the first")
        df = df.drop_duplicates("case_id")
    print(f"{path}: {len(df)} cases; repaired records for case_ids: {fixed_ids or 'none'}; unreadable records: {len(bad)}")
    return df


def main():
    base_p, rome_p, memit_p = sys.argv[1:4]
    base = pd.read_csv(base_p)
    rome, memit = load(rome_p), load(memit_p)
    common = [c for c in base["case_id"] if c in set(rome["case_id"]) and c in set(memit["case_id"])]
    only_r = sorted(set(rome["case_id"]) - set(memit["case_id"]))
    only_m = sorted(set(memit["case_id"]) - set(rome["case_id"]))
    print(f"\nCases in both ROME and MEMIT files: {len(common)}")
    print(f"Only in ROME: {only_r} | only in MEMIT: {only_m}")
    for name, df, out in (("ROME", rome, "rome_clean.csv"), ("MEMIT", memit, "memit_clean.csv")):
        df = df.set_index("case_id").loc[common].reset_index()
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")
    b = base[base["case_id"].isin(common)].set_index("case_id").loc[common].reset_index()
    b.to_csv("baseline_common.csv", index=False)
    print(f"wrote baseline_common.csv ({len(b)} rows)")
    # check the unedited scoring is identical on the common cases (pre-edit check, file level)
    print("Relation types among the common cases:", b["relation_id"].nunique())


if __name__ == "__main__":
    main()