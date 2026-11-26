"""年度检查：每块地每年只在"检查窗口"里，用好天气影像合成判断一次。

为什么这样做（通俗说法）：
  筏架一挂就是几个月甚至全年（桑沟湾：海带 11 月→次年 4–8 月收，牡蛎/海参全年养），
  不需要逐景监测。逐景投票会被大风天"看不清"的影像拖累；改成每年选一个最好分辨的时段，
  把这段时间里海况合适的影像取中位数，一个数判断"这块海今年在不在养"。

步骤：
  1. 选窗口：逐月计算"养殖地块 vs 空海面"的可分性（AUC），取最好的连续 N 个月（默认 3）。
     只用海面背景在 [MIN_BACKGROUND_DB, MAX_BACKGROUND_DB] 之间的影像。
  2. 合成：每个多边形 × 每个"季节年"，窗口内好天气影像的指标中位数 m 和影像数 n。
  3. 标定（只用标定组地块）：
       在养门限 thr：养殖地块 vs 空海面的合成值，取 Youden 最优；
       满覆盖系数 k：m / 参考覆盖率 的中位数（产量核验用：卫星覆盖率 = m / k）。
  4. 检验（检验组地块）：在养判断的 AUC、误报率、按挪动距离的检出率。

输出（outputs/<area>_annual/）：
  window_selection.csv   逐月可分性
  annual_composite.csv   每个多边形 × 年 的合成值（后续 trace_sim 用）
  annual_metrics.json    选出的窗口、门限、k、检验组结果
  annual_check.png       (a) 逐月可分性 (b) 检出率 vs 挪动距离

用法：
  python -m rs_verify.annual --area ailun --claims data/claims/ailun_claims.csv \
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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402
from .quantity import reference_cover, split_parcels  # noqa: E402
from .verify import prepare_activity  # noqa: E402

CLEAN_OVERLAP = 0.1


# ---------------------------------------------------------------------------
# 多边形分组
# ---------------------------------------------------------------------------
def polygon_table(claims: pd.DataFrame) -> pd.DataFrame:
    """每个多边形一行：pid, ref_pid, kind(farm/shifted), shift_m, overlap_ref。"""
    cl = claims.copy()
    cl["pid"] = cl["pid"].astype(str)
    cl["ref_pid"] = cl["ref_pid"].astype(str)
    farm = cl[cl["label"] == "true"].drop_duplicates("pid")[["pid", "ref_pid"]].assign(
        kind="farm", shift_m=0.0, overlap_ref=1.0)
    sh = cl[cl["label"] == "fake_location"].drop_duplicates("pid")[["pid", "ref_pid", "shift_m", "overlap_ref"]].assign(
        kind="shifted")
    return pd.concat([farm, sh], ignore_index=True)


def is_open_sea(polys: pd.DataFrame) -> pd.Series:
    return (polys["kind"] == "shifted") & (polys["shift_m"] >= 2000) & (polys["overlap_ref"] < 0.05)


def is_gap(polys: pd.DataFrame, max_shift: float | None = None) -> pd.Series:
    """养殖区内或旁边的"空地块"：离真实养殖地块不远（≤ 1000 m），但与参考筏区几乎不重叠。
    注意参考数据较旧，其中一部分其实已有筏架——评估时另用目视真值核对。"""
    max_shift = C.GAP_MAX_SHIFT_M if max_shift is None else max_shift
    return (polys["kind"] == "shifted") & (polys["shift_m"] <= max_shift) & (polys["overlap_ref"] < 0.05)


# ---------------------------------------------------------------------------
# 窗口选择与合成
# ---------------------------------------------------------------------------
def good_scenes(act_raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_activity(act_raw, min_bg_db=C.MIN_BACKGROUND_DB, max_bg_db=C.MAX_BACKGROUND_DB)


def monthly_separability(act: pd.DataFrame, farm: set[str], sea: set[str]) -> pd.DataFrame:
    a = act[act["pid"].isin(farm | sea)].copy()
    a["year"] = a["date"].dt.year
    pm = a.groupby(["pid", "year", "month"])["metric"].median().reset_index()
    pm["is_farm"] = pm["pid"].isin(farm)
    rows = []
    for mth in range(1, 13):
        x = pm[pm["month"] == mth]
        if x["is_farm"].nunique() < 2:
            rows.append({"month": mth, "auc": np.nan, "n": len(x)})
            continue
        rows.append({"month": mth, "auc": float(roc_auc_score(x["is_farm"], x["metric"])), "n": len(x)})
    return pd.DataFrame(rows)


def select_window(sep: pd.DataFrame, length: int = C.INSPECTION_WINDOW_LENGTH) -> list[int]:
    auc = dict(zip(sep["month"], sep["auc"].fillna(0)))
    best, best_v = None, -1.0
    for start in range(1, 13):
        months = [((start - 1 + i) % 12) + 1 for i in range(length)]
        v = float(np.mean([auc[m] for m in months]))
        if v > best_v:
            best, best_v = months, v
    return best


def season_year(dates: pd.Series, window: list[int]) -> pd.Series:
    """窗口跨年（如 12、1、2 月）时，12 月算到下一年。"""
    wraps = any(window[i] > window[i + 1] for i in range(len(window) - 1))
    y = dates.dt.year
    if wraps:
        y = y + (dates.dt.month >= window[0]).astype(int)
    return y


def full_years(window: list[int], start: str = C.START_DATE, end: str = C.END_DATE) -> list[int]:
    """整个窗口都落在分析时段内的季节年（不完整的首尾年不算）。"""
    wraps = any(window[i] > window[i + 1] for i in range(len(window) - 1))
    out = []
    for y in range(int(start[:4]), int(end[:4]) + 2):
        y0 = y - 1 if wraps else y
        first = pd.Timestamp(y0, window[0], 1)
        last = pd.Timestamp(y, window[-1], 1) + pd.offsets.MonthEnd(0)
        if first >= pd.Timestamp(start) and last <= pd.Timestamp(end):
            out.append(y)
    return out


def annual_composite(act: pd.DataFrame, window: list[int]) -> pd.DataFrame:
    a = act[act["month"].isin(window)].copy()
    a["year"] = season_year(a["date"], window)
    a = a[a["year"].isin(full_years(window))]
    return a.groupby(["pid", "year"]).agg(m=("metric", "median"), n=("metric", "size")).reset_index()


def youden(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    fpr, tpr, thr = roc_curve(y, np.r_[pos, neg])
    return float(thr[int(np.argmax(tpr - fpr))])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(claims: pd.DataFrame, act_raw: pd.DataFrame, cover_ref: dict[str, float],
        window: list[int] | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    act = good_scenes(act_raw)
    polys = polygon_table(claims)
    calib, test = split_parcels(list(cover_ref))
    polys["split"] = np.where(polys["ref_pid"].isin(calib), "calib", "test")
    polys["open_sea"] = is_open_sea(polys)
    polys["gap"] = is_gap(polys)

    farm_all = set(polys.loc[polys["kind"] == "farm", "pid"])
    sea_cal = set(polys.loc[polys["open_sea"] & (polys["split"] == "calib"), "pid"])
    farm_cal = set(polys.loc[(polys["kind"] == "farm") & (polys["split"] == "calib"), "pid"])

    # 窗口只用标定组选，避免"考题泄露"
    sep = monthly_separability(act, farm_cal, sea_cal)
    if window is None:
        window = C.INSPECTION_WINDOW_MONTHS or select_window(sep)

    comp = annual_composite(act, window).merge(polys, on="pid", how="left")
    comp["cover_ref"] = comp["ref_pid"].map(cover_ref)
    ok = comp["n"] >= C.MIN_SCENES_PER_YEAR

    cal_farm = comp[ok & comp["pid"].isin(farm_cal)]
    cal_sea = comp[ok & comp["pid"].isin(sea_cal)]
    thr = youden(cal_farm["m"].to_numpy(), cal_sea["m"].to_numpy())
    k = float(np.median(cal_farm["m"] / cal_farm["cover_ref"]))
    comp["present"] = np.where(ok, comp["m"] >= thr, np.nan)
    comp["status"] = np.where(ok, "ok", "insufficient")
    comp["cover_sat"] = (comp["m"] / k).clip(lower=0, upper=1.0)

    te = comp[comp["split"] == "test"]
    te_farm = te[te["kind"] == "farm"]
    te_sea = te[te["open_sea"]]
    te_ok_f, te_ok_s = te_farm[te_farm["status"] == "ok"], te_sea[te_sea["status"] == "ok"]
    auc = float(roc_auc_score(np.r_[np.ones(len(te_ok_f)), np.zeros(len(te_ok_s))],
                              np.r_[te_ok_f["m"], te_ok_s["m"]])) if len(te_ok_f) and len(te_ok_s) else float("nan")
    # "未通过" = 判为不在养，或无法核验
    fail = lambda d: float(((d["present"] != 1.0) | (d["status"] != "ok")).mean()) if len(d) else float("nan")  # noqa: E731
    sh = te[te["kind"] == "shifted"]
    det_clean = {int(d): fail(g) for d, g in sh[sh["overlap_ref"] < CLEAN_OVERLAP].groupby("shift_m")}
    det_all = {int(d): fail(g) for d, g in sh.groupby("shift_m")}

    per_parcel = te_farm[te_farm["status"] == "ok"].groupby("pid").agg(cover_ref=("cover_ref", "first"),
                                                                       cover_sat=("cover_sat", "median"))
    r = float(np.corrcoef(per_parcel["cover_ref"], per_parcel["cover_sat"])[0, 1]) \
        if len(per_parcel) > 2 and per_parcel["cover_ref"].std() > 0 and per_parcel["cover_sat"].std() > 0 else float("nan")

    m = {
        "metric": C.ACTIVITY_METRIC,
        "sea_state_db": [C.MIN_BACKGROUND_DB, C.MAX_BACKGROUND_DB],
        "good_scenes": int(act["date"].nunique()),
        "window_months": window,
        "window_auc_calib": float(sep.set_index("month").loc[window, "auc"].mean()),
        "presence_threshold": thr,
        "k_full_cover": k,
        "n_parcels": {"calib": len(calib), "test": len(test)},
        "test_auc_farm_vs_open_sea": auc,
        "test_false_alarm_farm_years": fail(te_farm),
        "test_detection_open_sea": fail(te_sea),
        "test_detection_by_shift_clean": det_clean,
        "test_detection_by_shift_all": det_all,
        "test_insufficient_rate_farm": float((te_farm["status"] != "ok").mean()) if len(te_farm) else float("nan"),
        "test_cover_sat_vs_ref_r": r,
        "years": sorted(int(y) for y in comp["year"].unique()),
    }
    return m, sep, comp


def plot(sep: pd.DataFrame, m: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = ["tab:orange" if mm in m["window_months"] else "tab:gray" for mm in sep["month"]]
    axes[0].bar(sep["month"], sep["auc"], color=cols)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("AUC farm vs open sea (calibration)")
    axes[0].set_title(f"Inspection window = {m['window_months']}")
    for key, lab in [("test_detection_by_shift_clean", "Moved to open sea / elsewhere"),
                     ("test_detection_by_shift_all", "All shifted geofences")]:
        d = m[key]
        if d:
            xs = sorted(d)
            axes[1].plot(xs, [d[x] for x in xs], "o-", label=lab)
    axes[1].axhline(m["test_false_alarm_farm_years"], color="gray", ls="--", label="False alarm (true parcels)")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Geofence shift distance (m)")
    axes[1].set_ylabel("Flagged (annual check, test parcels)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def load_inputs(args) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    files = sorted({f for p in args.activity for f in (glob.glob(p) or [p])})
    act_raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    rm = geo.RasterMask(geo.load_geojson(args.raft_reference), C.STUDY_AREAS[args.area]["bbox"])
    cover = reference_cover(geo.load_geojson(args.claim_geoms), rm)
    return pd.read_csv(args.claims), act_raw, cover


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--claims", required=True)
    ap.add_argument("--claim-geoms", required=True)
    ap.add_argument("--raft-reference", required=True)
    ap.add_argument("--activity", required=True, nargs="+")
    ap.add_argument("--metric", default=None, choices=["excess", "abs_excess", "bright_frac"])
    ap.add_argument("--window", default=None, help="手动指定检查窗口月份，如 2,3,4（默认由数据选）")


def main() -> None:
    ap = argparse.ArgumentParser(description="年度检查窗口 + 合成判断")
    add_common_args(ap)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.metric:
        C.ACTIVITY_METRIC = args.metric
    window = [int(x) for x in args.window.split(",")] if args.window else None
    claims, act_raw, cover = load_inputs(args)
    m, sep, comp = run(claims, act_raw, cover, window)
    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_annual")
    od.mkdir(parents=True, exist_ok=True)
    sep.to_csv(od / "window_selection.csv", index=False)
    comp.to_csv(od / "annual_composite.csv", index=False)
    with open(od / "annual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    plot(sep, m, od / "annual_check.png")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("输出目录：", od)


if __name__ == "__main__":
    main()
