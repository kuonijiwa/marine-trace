"""产量核验（第二核心检查）：声明的数量，是否超出这块海按卫星看到的养殖规模所能产出的上限？

要拦的造假：本来就在基地里有一块海、真在养，再把基地外的货"洗"进来，用同一块地块的名义多卖。
位置核验对这种造假无效（地块是真的、也确实在养），只能比"卖出的量"和"这块海能产的量"。

做法（通俗说法）：
  1. 卫星看规模：养殖季里这块地块活动指标的中位数 m。按混合模型 m ≈ k × 覆盖率，
     k = "满覆盖时的指标值"，用一半地块（标定组）的 m / 参考覆盖率 的中位数标定。
     于是卫星估计覆盖率 cover_sat = m / k，产能上限 ∝ cover_sat × 面积 × 单产。
  2. 声明量：真实声明 = 参考覆盖率 × 利用率 u（u = quantity_rel / 0.95，在 0.53–1 之间，代表实际产量
     不超过产能）；产量冒用 = 真实声明 × 放大倍数 r。单产和面积在比值里约掉，所以不需要知道具体吨数。
  3. 判定：产量比 = 声明量 / 卫星产能。超过容差 τ → 报警。τ 在标定组上取"真实声明的 95% 分位"，
     即预设约 5% 的误报率；然后在另一半地块（检验组）上报告误报率和各放大倍数的检出率。
  4. 顺带：位置冒用的假围栏如果挪到了空海面，卫星产能≈0，产量检查也会报警。

参考覆盖率来自已发表数据集（Fu et al. 2021，2016–2019 年），声明年份是 2020–2025，
期间地块变化会混入误差——论文里要说明。

输出（默认 outputs/<area>_quantity/）：
  quantity_metrics.json   k、τ、检验组误报率、各放大倍数检出率、AUC、卫星覆盖率与参考覆盖率的相关
  quantity_check.png      (a) 检出率 vs 放大倍数（含误报率线）(b) 卫星覆盖率 vs 参考覆盖率
  quantity_scored.csv     每份声明的产量比与判定

用法：
  python -m rs_verify.quantity --area ailun --claims data/claims/ailun_claims.csv \
      --claim-geoms data/claims/ailun_claim_geoms.geojson \
      --raft-reference data/reference/ailun_fu2021_reference.geojson \
      --activity data/activity/ailun_activity_*.csv --metric abs_excess
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402
from .verify import prepare_activity  # noqa: E402

Q_MAX = 0.95          # claims.py 里真实声明 quantity_rel 的上限
FALSE_ALARM_TARGET = 0.05
CLEAN_OVERLAP = 0.1


def reference_cover(claim_geoms: dict, raft_mask: geo.RasterMask) -> dict[str, float]:
    """每个参考地块（R 开头、不含 _S）在已发表数据集里的覆盖率。"""
    out = {}
    for f in claim_geoms["features"]:
        p = f["properties"]
        if p.get("kind") == "reference":
            out[str(p["pid"])] = raft_mask.fraction(f["geometry"])
    return out


def window_median(act: pd.DataFrame, claims: pd.DataFrame, min_scenes: int = C.MIN_SCENES_IN_WINDOW) -> pd.Series:
    a_by = dict(tuple(act.groupby("pid")))
    vals = {}
    for c in claims.itertuples(index=False):
        a = a_by.get(c.pid)
        if a is None:
            continue
        w = a[(a["date"] >= c.start) & (a["date"] <= c.end)]
        if len(w) >= min_scenes:
            vals[c.claim_id] = float(w["metric"].median())
    return pd.Series(vals, dtype=float)


def split_parcels(pids: list[str], seed: int = C.RANDOM_SEED) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed + 1)
    p = sorted(pids)
    rng.shuffle(p)
    h = len(p) // 2
    return set(p[:h]), set(p[h:])


def run(claims: pd.DataFrame, act: pd.DataFrame, cover_ref: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    cl = claims.copy()
    cl["pid"] = cl["pid"].astype(str)
    cl["ref_pid"] = cl["ref_pid"].astype(str)
    cl["start"] = pd.to_datetime(cl["start"])
    cl["end"] = pd.to_datetime(cl["end"])
    cl = cl[cl["label"].isin(["true", "fake_quantity", "fake_location"])].copy()

    cl["cover_ref"] = cl["ref_pid"].map(cover_ref)
    cl["m"] = cl["claim_id"].map(window_median(act, cl))
    # 声明量（单位：满覆盖产能）：原地块参考覆盖率 × 利用率 × 放大倍数
    cl["declared"] = cl["cover_ref"] * cl["quantity_rel"] / Q_MAX

    calib, test = split_parcels(list(cover_ref))
    cl["split"] = np.where(cl["ref_pid"].isin(calib), "calib", "test")

    t_cal = cl[(cl["label"] == "true") & (cl["split"] == "calib") & cl["m"].notna() & (cl["cover_ref"] > 0)]
    k = float(np.median(t_cal["m"] / t_cal["cover_ref"])) if len(t_cal) else float("nan")
    cl["cover_sat"] = (cl["m"] / k).clip(lower=0, upper=1.0)
    eps = 1e-6
    cl["qty_ratio"] = cl["declared"] / cl["cover_sat"].clip(lower=eps)
    # 数据不足（窗口内影像太少）→ 无法核验，按"报警"处理（与位置核验一致：不予通过）
    cl.loc[cl["m"].isna(), "qty_ratio"] = np.inf

    t_cal_ratio = cl[(cl["label"] == "true") & (cl["split"] == "calib")]["qty_ratio"].replace(np.inf, np.nan).dropna()
    tau = float(np.quantile(t_cal_ratio, 1 - FALSE_ALARM_TARGET)) if len(t_cal_ratio) else float("nan")
    cl["flag"] = cl["qty_ratio"] > tau

    te = cl[cl["split"] == "test"]
    true_te = te[te["label"] == "true"]
    qty_te = te[te["label"] == "fake_quantity"]
    loc_te = te[(te["label"] == "fake_location") & (te["overlap_ref"] < CLEAN_OVERLAP)]

    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        s = np.r_[pos["qty_ratio"].replace(np.inf, 1e9), neg["qty_ratio"].replace(np.inf, 1e9)]
        y = np.r_[np.zeros(len(pos)), np.ones(len(neg))]
        return float(roc_auc_score(y, s))

    ref_rows = te[te["label"] == "true"].dropna(subset=["m"])
    per_parcel = ref_rows.groupby("ref_pid").agg(cover_ref=("cover_ref", "first"), cover_sat=("cover_sat", "median"))
    ok = len(per_parcel) > 2 and per_parcel["cover_ref"].std() > 0 and per_parcel["cover_sat"].std() > 0
    r = float(np.corrcoef(per_parcel["cover_ref"], per_parcel["cover_sat"])[0, 1]) if ok else float("nan")

    m = {
        "metric": C.ACTIVITY_METRIC,
        "k_full_cover_metric": k,
        "tolerance_tau": tau,
        "n_parcels": {"calib": len(calib), "test": len(test)},
        "test_false_alarm_true": float(true_te["flag"].mean()) if len(true_te) else float("nan"),
        "test_detection_by_inflation": {float(i): float(g["flag"].mean()) for i, g in qty_te.groupby("inflation")},
        "test_auc_quantity": auc(true_te, qty_te),
        "test_detection_clean_location_fakes": float(loc_te["flag"].mean()) if len(loc_te) else float("nan"),
        "test_cover_sat_vs_ref_pearson_r": r,
        "test_cover_sat_median": float(per_parcel["cover_sat"].median()) if len(per_parcel) else float("nan"),
        "test_cover_ref_median": float(per_parcel["cover_ref"].median()) if len(per_parcel) else float("nan"),
        "insufficient_rate": float(cl["m"].isna().mean()),
    }
    return cl, m


def plot(cl: pd.DataFrame, m: dict, out: Path) -> None:
    te = cl[cl["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d = m["test_detection_by_inflation"]
    if d:
        xs = sorted(d)
        axes[0].plot(xs, [d[x] for x in xs], "o-", label="Inflated claims (detected)")
    axes[0].axhline(m["test_false_alarm_true"], color="gray", ls="--", label="True claims (false alarm)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Declared quantity inflation factor")
    axes[0].set_ylabel("Flag rate (test parcels)")
    axes[0].set_title(f"Quantity check, τ = {m['tolerance_tau']:.2f}")
    axes[0].legend(fontsize=8)
    pp = te[te["label"] == "true"].dropna(subset=["m"]).groupby("ref_pid").agg(
        cover_ref=("cover_ref", "first"), cover_sat=("cover_sat", "median"))
    axes[1].scatter(pp["cover_ref"], pp["cover_sat"], s=12)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Reference raft cover (Fu et al. 2021)")
    axes[1].set_ylabel("Satellite-estimated cover (median over seasons)")
    axes[1].set_title(f"Test parcels, r = {m['test_cover_sat_vs_ref_pearson_r']:.2f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="产量核验实验")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--claims", required=True)
    ap.add_argument("--claim-geoms", required=True)
    ap.add_argument("--raft-reference", required=True)
    ap.add_argument("--activity", required=True, nargs="+")
    ap.add_argument("--metric", default=None, choices=["excess", "abs_excess", "bright_frac"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.metric:
        C.ACTIVITY_METRIC = args.metric

    files = sorted({f for p in args.activity for f in (glob.glob(p) or [p])})
    act = prepare_activity(pd.concat([pd.read_csv(p) for p in files], ignore_index=True))
    rm = geo.RasterMask(geo.load_geojson(args.raft_reference), C.STUDY_AREAS[args.area]["bbox"])
    cover = reference_cover(geo.load_geojson(args.claim_geoms), rm)
    cl, m = run(pd.read_csv(args.claims), act, cover)

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_quantity")
    od.mkdir(parents=True, exist_ok=True)
    cl.to_csv(od / "quantity_scored.csv", index=False)
    with open(od / "quantity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    plot(cl, m, od / "quantity_check.png")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("输出目录：", od)


if __name__ == "__main__":
    main()
