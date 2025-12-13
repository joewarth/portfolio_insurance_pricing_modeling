"""
utils.py: Shared helper functions for Insurance Pricing Project.

Summary:
Utilities that are dependency-heavy and very project-specific.
I make no claim that these are "pythontic" and well-written.

Usage:
from utils import *
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools

from pandas.api.types import is_numeric_dtype, CategoricalDtype
from sklearn.model_selection import StratifiedShuffleSplit

def fmt_cap(cap):
    if cap is None: return "uncapped"
    cap = int(cap)
    if cap >= 1_000_000: return f"{cap//1_000_000}MIL"
    if cap >= 1_000:     return f"{cap//1_000}K"
    return str(cap)



def _normalize_cap_label(cap):
    """Return label suffix used in capped columns, e.g. 100000 -> '100K', '1M' -> '1MIL'."""
    if cap is None:
        return None
    if isinstance(cap, (int, float)) and not np.isnan(cap):
        cap = int(cap)
        if cap >= 1_000_000:
            return f"{cap // 1_000_000}MIL"
        elif cap >= 1_000:
            return f"{cap // 1_000}K"
        else:
            return str(cap)
    s = str(cap).upper().replace(" ", "").replace("MN", "M").replace("MM", "M")
    s = s.replace("MILLION", "MIL").replace("MIO", "MIL")
    s = s.replace("MIL", "MIL").replace("M", "MIL").replace("K", "K")
    return s

def _resolve_claim_col(df, cap):
    """Pick ClaimAmount or a capped column based on 'cap'."""
    if cap is None:
        return "ClaimAmount", "uncapped"
    label = _normalize_cap_label(cap)
    target = f"ClaimAmount_capped_{label}"
    if target in df.columns:
        return target, label
    # fallback: try to find a best-effort match
    candidates = [c for c in df.columns if c.startswith("ClaimAmount_capped_")]
    # try suffix match (e.g., endswith '100K' or '1MIL')
    matches = [c for c in candidates if c.endswith(label)]
    if matches:
        return matches[0], label
    raise ValueError(
        f"Could not find capped column for cap='{cap}'. "
        f"Available: {', '.join(candidates[:8]) + (' ...' if len(candidates)>8 else '')}"
    )

def runmultiplot(
    data: pd.DataFrame,
    dimension: str,
    metric: str = "Frequency",
    cap=None,
    nstd_max: int = 1,
    figsize=(20, 13)
):
    """
    Plot Exposure bars by `dimension` with a line for the chosen metric and n-std bands.
    If `dimension` is numeric, it will be auto-binned (quantile bins).
    """
    metric = metric.strip().title()
    if metric not in {"Frequency", "Severity", "Pure Premium"}:
        raise ValueError("metric must be one of {'Frequency','Severity','Pure Premium'}")

    df = data.copy()

    # --- resolve claim column based on cap
    claim_col, cap_label = ("ClaimAmount", "uncapped")
    if metric in {"Severity", "Pure Premium"}:
        _resolver = globals().get("_resolve_claim_col", None)
        if cap is not None and callable(_resolver):
            claim_col, cap_label = _resolver(df, cap)
        elif cap is not None:
            cap_str = str(cap).upper().replace(",", "").replace("_", "").replace(" ", "")
            mapping = {"100K": "100000", "1M": "1000000", "1MIL": "1000000"}
            cap_norm = mapping.get(cap_str, cap_str)
            cand = f"ClaimAmount_capped_{cap_norm}"
            claim_col, cap_label = (cand, cap_norm) if cand in df.columns else ("ClaimAmount", "uncapped")

    # --- AUTO-BIN if numeric (preserve category order)
    dim_col = dimension
    bin_label_order = None  # will hold the ordered labels for plotting
    if is_numeric_dtype(df[dimension]):
        uniq = df[dimension].dropna().unique()
        if len(uniq) > 1:
            binned = pd.qcut(df[dimension], q=min(12, len(uniq)), duplicates="drop")
            if isinstance(binned.dtype, CategoricalDtype):
                cats = list(binned.cat.categories)          # ordered IntervalIndex
                labels = [str(iv) for iv in cats]           # unique, readable
                bin_label_order = labels[:]                  # preserve order for plotting
                binned = binned.cat.rename_categories(labels)
            df["_dim_binned"] = binned.astype(str).fillna("NA")
        else:
            df["_dim_binned"] = df[dimension].astype(str).fillna("NA")
            bin_label_order = pd.Index(df["_dim_binned"]).drop_duplicates().tolist()
        dim_col = "_dim_binned"

    # --- per-row metric (for SE calc)
    if metric == "Frequency":
        df["_metric_row"] = np.divide(
            df["ClaimNb"], df["Exposure"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["Exposure"].to_numpy(dtype=float) != 0)
        )
    elif metric == "Severity":
        df["_metric_row"] = np.divide(
            df[claim_col], df["ClaimNb"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["ClaimNb"].to_numpy(dtype=float) != 0)
        )
    else:  # Pure Premium
        df["_metric_row"] = np.divide(
            df[claim_col], df["Exposure"],
            out=np.full(len(df), np.nan, dtype=float),
            where=(df["Exposure"].to_numpy(dtype=float) != 0)
        )

    # --- aggregate
    agg = {
        "Exposure": "sum",
        "ClaimNb": "sum",
        claim_col: "sum",
        "_metric_row": ["mean", "std", "count"]
    }
    temp = (
        df.groupby(dim_col, dropna=False, observed=False)
          .agg(agg)
          .reset_index()
    )
    temp.columns = [c if isinstance(c, str) else "_".join([p for p in c if p]) for c in temp.columns]

    # --- group-level metric + SE
    if metric == "Frequency":
        temp["Metric"] = np.divide(
            temp["ClaimNb_sum"], temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            np.sqrt(temp["ClaimNb_sum"]), temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
    elif metric == "Severity":
        temp["Metric"] = np.divide(
            temp[f"{claim_col}_sum"], temp["ClaimNb_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["ClaimNb_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            temp["_metric_row_std"], np.sqrt(temp["_metric_row_count"].clip(lower=1)),
            out=np.zeros(len(temp), dtype=float),
            where=(temp["_metric_row_count"].to_numpy(dtype=float) > 0)
        )
    else:  # Pure Premium
        temp["Metric"] = np.divide(
            temp[f"{claim_col}_sum"], temp["Exposure_sum"],
            out=np.zeros(len(temp), dtype=float),
            where=(temp["Exposure_sum"].to_numpy(dtype=float) != 0)
        )
        temp["SE"] = np.divide(
            temp["_metric_row_std"], np.sqrt(temp["_metric_row_count"].clip(lower=1)),
            out=np.zeros(len(temp), dtype=float),
            where=(temp["_metric_row_count"].to_numpy(dtype=float) > 0)
        )

    # --- portfolio line
    exposure_sum = df["Exposure"].sum()
    claimnb_sum  = df["ClaimNb"].sum()
    claim_sum    = df[claim_col].sum()
    if metric == "Frequency":
        portfolio_metric = (claimnb_sum / exposure_sum) if exposure_sum else 0.0
    elif metric == "Severity":
        portfolio_metric = (claim_sum / claimnb_sum) if claimnb_sum else 0.0
    else:
        portfolio_metric = (claim_sum / exposure_sum) if exposure_sum else 0.0

    # --- x order & ranks
    if dim_col == "_dim_binned" and bin_label_order is not None:
        order = bin_label_order  # interval order left->right
    else:
        order = pd.Index(temp[dim_col].astype(str)).drop_duplicates().tolist()

    temp = temp.set_index(dim_col).loc[order].reset_index()
    temp["Rank"] = np.arange(len(temp))

    # --- plot
    fig, ax1 = plt.subplots(figsize=figsize)

    sns.barplot(x=dim_col, y="Exposure_sum", data=temp, estimator=sum, order=order, alpha=0.7, ax=ax1)
    if ax1.containers:
        ax1.bar_label(ax1.containers[0])
    ax1.set_ylabel("Exposure")
    ax1.set_xlabel(dimension)  # <- keep axis label as the original column name
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")

    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(metric)

    sns.lineplot(x="Rank", y="Metric", data=temp, marker="o", markersize=10, ax=ax2, label=metric)

    for n in range(1, nstd_max + 1):
        ax2.fill_between(
            temp["Rank"].to_numpy(),
            np.maximum((temp["Metric"] - n * temp["SE"]).to_numpy(), 0.0),
             np.maximum((temp["Metric"] + n * temp["SE"]).to_numpy(), 0.0),
            alpha=0.25,
            label=(f"±{n}·SE")
        )

    ax2.axhline(y=portfolio_metric, linestyle="--", linewidth=2, label="Portfolio")

    # align tick positions with labels (Rank-based line over bar centers)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks(temp["Rank"])
    ax1.set_xticklabels(order, rotation=45, ha="right")

    # legend (dedup)
    handles, labels = ax2.get_legend_handles_labels()
    seen = set(); h2=[]; l2=[]
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax2.legend(h2, l2, loc="upper left")

    title_cap = "" if (metric == "Frequency" or cap_label == "uncapped") else f" (cap {cap_label})"
    plt.title(f"{metric}{title_cap} by {dimension if dim_col==dimension else f'{dimension} (binned)'} vs Portfolio")
    plt.tight_layout()

    return fig

def stratified_split_match_portfolio_freq(
    df: pd.DataFrame,
    group_col: str = "IDpol",
    exposure_col: str = "Exposure",
    claim_col: str = "ClaimNb",
    test_size: float = 0.20,
    q: int = 10,                 # initial number of bins
    tol: float = 0.005,          # |train_pf - test_pf| tolerance
    max_tries: int = 200,
    random_state: int = 42
):
    req = {group_col, exposure_col, claim_col}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Collapse to one row per policy just in case
    g = (df.groupby(group_col, as_index=True)
           .agg(Exposure_sum=(exposure_col, "sum"),
                ClaimNb_sum=(claim_col, "sum")))
    g = g[g["Exposure_sum"] > 0].copy()
    if g.empty:
        raise ValueError("No policies with positive exposure after aggregation.")

    g["pol_freq"] = g["ClaimNb_sum"] / g["Exposure_sum"]

    # Find a valid number of bins (>=2) so each bin can allocate to train/test
    def make_bins(g, q_try):
        bins = pd.qcut(g["pol_freq"], q=min(q_try, g["pol_freq"].nunique()),
                       duplicates="drop")
        return bins

    def bins_ok(bins, test_size):
        # need at least one item in train and test for every class
        vc = bins.value_counts()
        return (len(vc) >= 2) and all((vc * test_size >= 1) & (vc * (1 - test_size) >= 1))

    q_try = min(q, g["pol_freq"].nunique())
    bins = make_bins(g, q_try)
    while not bins_ok(bins, test_size):
        q_try -= 1
        if q_try < 2:
            # fall back to a single strat variable (has_claim) if freq bins too sparse
            bins = (g["ClaimNb_sum"] > 0).astype(int)
            break
        bins = make_bins(g, q_try)

    labels = bins.astype(str)
    target_pf = g["ClaimNb_sum"].sum() / g["Exposure_sum"].sum()

    def portfolio_freq(sub):
        return sub["ClaimNb_sum"].sum() / sub["Exposure_sum"].sum()

    best = None
    best_diff = float("inf")
    seed = random_state

    for _ in range(max_tries):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        # X is unused by SSS; pass zeros of the right length
        train_idx, test_idx = next(sss.split(np.zeros(len(g)), labels))

        tr, te = g.iloc[train_idx], g.iloc[test_idx]
        tr_pf, te_pf = portfolio_freq(tr), portfolio_freq(te)
        diff = abs(tr_pf - te_pf)
        if diff < best_diff:
            best_diff = diff
            best = (tr.index.values, te.index.values, tr_pf, te_pf, seed, q_try)
            if diff <= tol:
                break
        seed += 1

    if best is None:
        raise RuntimeError("Failed to produce a split. Try increasing max_tries or relaxing tol.")

    train_pols, test_pols, tr_pf, te_pf, used_seed, used_q = best

    out = df.copy()
    out["set"] = np.where(out[group_col].isin(test_pols), "test", "train")

    print(f"Used seed: {used_seed} | bins: {used_q if used_q>=2 else 'has_claim'}")
    print(f"Overall PF: {target_pf:.6f}")
    print(f"Train PF  : {tr_pf:.6f}")
    print(f"Test  PF  : {te_pf:.6f}")
    print(f"|Train-Test|: {abs(tr_pf - te_pf):.6f} (tol={tol})")

    return out