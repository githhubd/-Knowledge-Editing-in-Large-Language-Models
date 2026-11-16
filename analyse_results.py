"""
Post-processing of the CSV files written by scaled_counterfact_pipeline.py.

Everything reported in the Results section can be regenerated with this script,
so that each number has a source file.

Expected columns (same as the pipeline output):
    case_id, relation_id, target_new_preferred, paraphrase_new_preferred,
    neighbourhood_true_preferred, greedy_equals_true, target_new, target_true,
    prompt, greedy_answer

Examples
    # one condition, one run
    python analyse_results.py --results scaled_baseline_results.csv --label "Unedited GPT-2"

    # three sampling seeds of the same condition -> mean +/- SD
    python analyse_results.py --results run42.csv run43.csv run44.csv --label "Unedited GPT-2"

    # paired comparison of two conditions on the same cases (e.g. before / after an edit)
    python analyse_results.py --results scaled_baseline_results.csv --label "Unedited GPT-2" \
        --compare rome_results.csv --compare-label "ROME"
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

METRICS = {
    "target_new_preferred": "Target (new preferred)",
    "paraphrase_new_preferred": "Paraphrase (new preferred)",
    "neighbourhood_true_preferred": "Neighbourhood (true preferred)",
    "greedy_equals_true": "Greedy answer = true object",
}
BOOT = 10000
RNG_SEED = 20240


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in METRICS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df["case_id"].duplicated().any():
        raise SystemExit(f"{path}: duplicate case_id values")
    return df


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summary(df, rng):
    rows = []
    for col, name in METRICS.items():
        lo, hi = boot_ci(df[col], rng)
        rows.append({"metric": name, "n": len(df), "mean": df[col].mean(),
                     "boot_ci_low": lo, "boot_ci_high": hi})
    return pd.DataFrame(rows)


def per_relation(df):
    g = df.groupby("relation_id")[list(METRICS)].mean()
    g.insert(0, "n", df.groupby("relation_id").size())
    return g.round(3).reset_index()


def mcnemar_exact(a, b):
    """a, b: 0/1 per case on the same cases. Exact two-sided McNemar test."""
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    return only_a, only_b, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_report(base, other, label_a, label_b, rng):
    merged = base.merge(other, on="case_id", suffixes=("_a", "_b"))
    if len(merged) != len(base) or len(merged) != len(other):
        print(f"WARNING: only {len(merged)} case_ids are shared by both files")
    lines, pvals, names = [], [], []
    for col, name in METRICS.items():
        a, b = merged[col + "_a"].to_numpy(), merged[col + "_b"].to_numpy()
        diff = b - a
        idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
        bd = diff[idx].mean(axis=1)
        lo, hi = np.percentile(bd, 2.5), np.percentile(bd, 97.5)
        if set(np.unique(np.concatenate([a, b]))) <= {0, 1}:
            n_a, n_b, p = mcnemar_exact(a, b)
            test = f"McNemar exact (only {label_a}: {n_a}, only {label_b}: {n_b})"
        else:
            nz = diff[diff != 0]
            p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
            test = "Wilcoxon signed-rank on per-case fractions"
        names.append((name, a.mean(), b.mean(), diff.mean(), lo, hi, test))
        pvals.append(p)
    adj = holm(np.array(pvals))
    lines.append(f"Paired comparison {label_a} -> {label_b}, n = {len(merged)} cases")
    for (name, ma, mb, d, lo, hi, test), p, pa in zip(names, pvals, adj):
        lines.append(f"{name}: {ma:.3f} -> {mb:.3f}, diff {d:+.3f} "
                     f"[95% paired bootstrap CI {lo:+.3f}, {hi:+.3f}]; {test}; "
                     f"p = {p:.4g}, Holm-adjusted p = {pa:.4g}")
    return "\n".join(lines)


def make_error_sample(df, label, outdir, size=40, seed=7):
    """Random sample of failures for manual coding (Appendix B).
    Failure = the target prompt where the edit did not win (target_new_preferred == 0)."""
    fails = df[df["target_new_preferred"] == 0]
    take = fails.sample(n=min(size, len(fails)), random_state=seed)
    cols = ["case_id", "relation_id", "prompt", "target_new", "target_true",
            "greedy_answer", "paraphrase_new_preferred", "neighbourhood_true_preferred"]
    out = take[cols].copy()
    out["error_type"] = ""      # to be filled in by hand
    out["comment"] = ""
    path = os.path.join(outdir, f"error_sample_{label.replace(' ', '_')}.csv")
    out.to_csv(path, index=False)
    print(f"{len(fails)} failures in total, {len(out)} written to {path} for manual coding")


def fig_metrics(tables, outdir):
    labels = list(METRICS.values())
    x = np.arange(len(labels))
    w = 0.8 / len(tables)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (name, t) in enumerate(tables.items()):
        m = t["mean"].to_numpy()
        err = np.vstack([m - t["boot_ci_low"], t["boot_ci_high"] - m])
        ax.bar(x + i * w - 0.4 + w / 2, m, w, yerr=err, capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax.set_ylabel("Proportion of cases")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_metrics.png"), dpi=200)
    plt.close(fig)


def fig_relations(rel, outdir):
    rel = rel.sort_values("neighbourhood_true_preferred")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(rel))
    ax.barh(y - 0.2, rel["target_new_preferred"], 0.4, label="Target (new preferred)")
    ax.barh(y + 0.2, rel["neighbourhood_true_preferred"], 0.4, label="Neighbourhood (true preferred)")
    ax.set_yticks(y)
    ax.set_yticklabels(rel["relation_id"], fontsize=7)
    ax.set_xlabel("Proportion of cases")
    ax.set_ylabel("Relation (Wikidata property)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_per_relation.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--label", default="condition")
    ap.add_argument("--compare")
    ap.add_argument("--compare-label", default="condition B")
    ap.add_argument("--outdir", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    runs = [load(p) for p in args.results]
    if len(runs) > 1:
        per_run = pd.DataFrame({p: {c: d[c].mean() for c in METRICS} for p, d in zip(args.results, runs)}).T
        per_run.to_csv(os.path.join(args.outdir, "per_run_means.csv"))
        print("\nMean +/- SD over runs (sample SD, n-1):")
        for c, name in METRICS.items():
            print(f"  {name}: {per_run[c].mean():.3f} +/- {per_run[c].std(ddof=1):.3f}")
        print(per_run.round(3))

    base = runs[0]
    tables = {args.label: summary(base, rng)}
    print(f"\n{args.label}: bootstrap 95% CI ({BOOT} resamples) for the first file")
    print(tables[args.label].round(3).to_string(index=False))
    k = int(base["target_new_preferred"].sum())
    lo, hi = wilson(k, len(base))
    print(f"Target: {k}/{len(base)} = {k/len(base):.3f}, Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    k = int(base["greedy_equals_true"].sum())
    lo, hi = wilson(k, len(base))
    print(f"Greedy: {k}/{len(base)} = {k/len(base):.3f}, Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    tables[args.label].to_csv(os.path.join(args.outdir, "summary_main.csv"), index=False)

    rel = per_relation(base)
    rel.to_csv(os.path.join(args.outdir, "per_relation.csv"), index=False)
    print(f"\nRelation types in sample: {len(rel)}; cases per relation min/median/max = "
          f"{rel['n'].min()}/{rel['n'].median()}/{rel['n'].max()}")

    if args.compare:
        other = load(args.compare)
        tables[args.compare_label] = summary(other, rng)
        text = paired_report(base, other, args.label, args.compare_label, rng)
        print("\n" + text)
        with open(os.path.join(args.outdir, "paired_tests.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        make_error_sample(other, args.compare_label, args.outdir)

    fig_metrics(tables, args.outdir)
    fig_relations(rel, args.outdir)
    print(f"\nTables and figures written to {args.outdir}/")


if __name__ == "__main__":
    main()