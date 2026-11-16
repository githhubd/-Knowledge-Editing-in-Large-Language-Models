"""
Small extra numbers used in the report that analyse_results.py does not print.
    python extra_stats.py scaled_baseline_results.csv rome_results.csv memit_results.csv
"""
import hashlib
import sys

import pandas as pd
from scipy import stats

base_path, rome_path, memit_path = sys.argv[1:4]
base = pd.read_csv(base_path)

# test-set identifier: SHA-256 of the sorted case_id list, written as comma-separated integers
ids = ",".join(str(i) for i in sorted(base["case_id"]))
print("SHA-256 of sorted case_id list:", hashlib.sha256(ids.encode()).hexdigest())

# exploratory: relation-level link between target and neighbourhood values (unedited model)
rel = base.groupby("relation_id")[["target_new_preferred", "neighbourhood_true_preferred"]].mean()
rho, p = stats.spearmanr(rel["target_new_preferred"], rel["neighbourhood_true_preferred"])
print(f"Spearman rho over {len(rel)} relations (target vs neighbourhood): {rho:.3f}, p = {p:.3g}")

# edit times
for name, path in (("ROME", rome_path), ("MEMIT", memit_path)):
    df = pd.read_csv(path)
    t = df["edit_seconds"]
    print(f"{name}: n = {len(df)}, mean edit time {t.mean():.1f} s, min {t.min():.1f}, max {t.max():.1f}")
    if len(df) > 1:
        rest = t.iloc[1:]
        print(f"   without the first edit: mean {rest.mean():.1f} s, min {rest.min():.1f}, max {rest.max():.1f}")

# same cases in the unedited baseline (pilot comparison)
for name, path in (("ROME", rome_path), ("MEMIT", memit_path)):
    df = pd.read_csv(path)
    b = base.set_index("case_id").loc[df["case_id"]]
    cols = ["target_new_preferred", "paraphrase_new_preferred",
            "neighbourhood_true_preferred", "greedy_equals_true"]
    print(f"\nPilot on the same {len(df)} cases: unedited vs {name}")
    print(pd.DataFrame({"unedited": b[cols].mean(), name: df[cols].mean()}).round(3))