"""典型案例图：一块登记养殖地块逐年"在养 → 不在养"，看卫星怎么一步步把它列进两级名单。

默认案例 R0035：参考数据（Fu et al. 2021，2016–2019）里是养殖地块，2024、2025 年海带季图上已无筏架，
系统先发"整改通知"、次年列入"吊销核查"。可加一块一直在养的对照地块（--compare）。

图的内容（fig_case_<pid>.png/.pdf）：
  第 1 行（可选，需 GEE）：每个海带季最清晰的一张 Sentinel-2 真彩色小图，黄框 = 地块
  第 2 行：雷达逐景指标（点，好海况）+ 年度中位数（折线），实线 = "看到筏"门限 T，虚线 = "没看到"门限 L
  第 3 行：光学逐景纹理（点，12–5 月晴空景）+ 年度持续性值（折线），门限同上
  背景色 = 当年结论（绿 在养 / 黄 待复核 / 红 不在养），顶部标注名单级别
另存 case_<pid>.csv：逐年数值、结论、名单级别。

用法：
  python -m rs_verify.case_study --composite outputs/v4_fusion/fusion_composite.csv \
      --metrics outputs/v4_fusion/fusion_metrics.json --pid R0035 --compare R0014 \
      --activity "data/activity/ailun_activity_*.csv" --s2 "data/activity_s2/ailun_s2_*.csv" --metric abs_excess \
      [--thumbs --claim-geoms data/claims/ailun_claim_geoms.geojson --project <PROJECT>] --outdir outputs/case_study
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

from . import annual  # noqa: E402
from . import config as C  # noqa: E402
from . import fusion  # noqa: E402
from . import geo  # noqa: E402

DEC_COLOR = {"present": "#d5f0dd", "review": "#fdebc8", "absent": "#f8d0cb", "insufficient": "#eeeeee"}
LEVEL_EN = {fusion.LEVEL_NOTICE: "Level 1: notice", fusion.LEVEL_REVOKE: "Level 2: revocation review"}


def _read(paths) -> pd.DataFrame | None:
    if not paths:
        return None
    files = sorted({f for p in paths for f in (glob.glob(p) or [p])})
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def annual_table(comp: pd.DataFrame, pid: str) -> pd.DataFrame:
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    lst = fusion.screening_list(comp)[["pid", "year", "level"]]
    t = comp[comp["pid"] == pid][["pid", "year", "m_sar", "tex_s2", "decision"] +
                                 [c for c in ("absent1", "n_sar", "n_s2") if c in comp]]
    return t.merge(lst, on=["pid", "year"], how="left").sort_values("year").reset_index(drop=True)


def scene_series(act_raw: pd.DataFrame | None, s2_raw: pd.DataFrame | None, pid: str):
    sar = s2 = None
    if act_raw is not None:
        a = annual.good_scenes(act_raw)
        sar = a[a["pid"] == pid][["date", "metric"]]
    if s2_raw is not None:
        s = fusion.s2_good_scenes(s2_raw)
        s2 = s[s["pid"] == pid][["date", "tex", "year"]]
    return sar, s2


def season_span(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """海带季 year = (year−1) 年 12 月 1 日 – year 年 5 月 31 日；背景色按日历年铺满更直观。"""
    return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)


def figure(case: pd.DataFrame, th: dict, out: Path, sar=None, s2=None, comp_case: pd.DataFrame | None = None,
           comp_sar=None, comp_s2=None, thumbs: dict | None = None, pid: str = "", cmp_pid: str = "") -> None:
    years = case["year"].tolist()
    n_rows = 3 if thumbs else 2
    fig = plt.figure(figsize=(13, 7.0 if thumbs else 5.6))
    gs = fig.add_gridspec(n_rows, len(years) if thumbs else 1,
                          height_ratios=([0.8] if thumbs else []) + [1, 1], hspace=0.3)
    if thumbs:
        for j, y in enumerate(years):
            ax = fig.add_subplot(gs[0, j])
            img = thumbs.get(y)
            if img is not None:
                ax.imshow(img)
            ax.set_title(f"{y}\n{thumbs.get(('date', y), '')}", fontsize=8)
            ax.axis("off")
    ax_sar = fig.add_subplot(gs[n_rows - 2, :])
    ax_s2 = fig.add_subplot(gs[n_rows - 1, :], sharex=ax_sar)
    for ax in (ax_sar, ax_s2):
        for r in case.itertuples():
            a, b = season_span(r.year)
            ax.axvspan(a, b, color=DEC_COLOR.get(r.decision, "#ffffff"), alpha=0.7, lw=0)
    mid = [pd.Timestamp(y, 7, 1) for y in years]          # 雷达：日历年中位数，点在年中
    mid_s2 = [pd.Timestamp(y, 3, 1) for y in years]       # 光学：海带季（上年 12 月–当年 5 月），点在季中
    if sar is not None and len(sar):
        ax_sar.plot(sar["date"], sar["metric"], ".", color="#1f5fa8", ms=4, alpha=0.6, label=f"{pid} scenes")
    ax_sar.plot(mid, case["m_sar"], "-o", color="#1f5fa8", lw=2, label=f"{pid} annual median")
    if comp_case is not None:
        if comp_sar is not None and len(comp_sar):
            ax_sar.plot(comp_sar["date"], comp_sar["metric"], ".", color="#555555", ms=3, alpha=0.35)
        ax_sar.plot([pd.Timestamp(y, 7, 1) for y in comp_case["year"]], comp_case["m_sar"], "--s", color="#555555",
                    lw=1.2, ms=4, label=f"{cmp_pid} (farmed, reference)")
    ax_sar.axhline(th["T_sar"], color="k", lw=0.9, label="'present' threshold T")
    ax_sar.axhline(th["L_sar"], color="k", lw=0.9, ls=":", label="'absent' threshold L")
    ax_sar.set_ylabel("SAR excess backscatter\n(abs. excess, ×10⁻³)")
    top = max(np.nanpercentile(sar["metric"], 98) if sar is not None and len(sar) else 0,
              np.nanmax(case["m_sar"]) * 1.3, th["T_sar"] * 2)
    ax_sar.set_ylim(min(-0.5, ax_sar.get_ylim()[0]), top * 1.15)
    ax_sar.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax_sar.set_title("(a) Sentinel-1 (all year, good sea state)", loc="left", fontsize=10)
    if s2 is not None and len(s2):
        ax_s2.plot(s2["date"], s2["tex"], ".", color="#c0392b", ms=4, alpha=0.6, label=f"{pid} clear scenes (Dec–May)")
    ax_s2.plot(mid_s2, case["tex_s2"], "-o", color="#c0392b", lw=2, label=f"{pid} seasonal persistence value")
    if comp_case is not None:
        ax_s2.plot([pd.Timestamp(y, 3, 1) for y in comp_case["year"]], comp_case["tex_s2"], "--s", color="#555555",
                   lw=1.2, ms=4, label=f"{cmp_pid} (farmed, reference)")
    ax_s2.axhline(th["T_s2"], color="k", lw=0.9)
    ax_s2.axhline(th["L_s2"], color="k", lw=0.9, ls=":")
    ax_s2.set_ylabel("Sentinel-2 raft texture\n(3×3 CV)")
    ax_s2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax_s2.set_title("(b) Sentinel-2 (kelp season, best clear scenes)", loc="left", fontsize=10)
    for r in case.itertuples():
        if isinstance(r.level, str):
            ax_sar.annotate(LEVEL_EN.get(r.level, r.level).replace(": ", ":\n"), (pd.Timestamp(r.year, 7, 1), 0.95),
                            xycoords=("data", "axes fraction"), ha="center", va="top",
                            fontsize=7.5, weight="bold", color="#7b241c")
    for ax in (ax_sar, ax_s2):
        ax.grid(alpha=0.25)
    plt.setp(ax_sar.get_xticklabels(), visible=False)
    fig.suptitle(f"Registered parcel {pid}: annual satellite screening "
                 "(background: green = present, amber = review, red = absent)", fontsize=10.5)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fetch_thumbs(pid: str, years: list[int], s2_raw: pd.DataFrame, geoms_path: str, project: str) -> dict:
    from .visual_sample import best_dates, make_fetcher

    gj = geo.load_geojson(geoms_path)
    geom = next(f["geometry"] for f in gj["features"] if str(f["properties"]["pid"]) == pid)
    dates = best_dates(s2_raw)
    dates = dates[(dates["pid"] == pid) & dates["year"].isin(years)]
    fetch = make_fetcher(project)
    out = {}
    for r in dates.itertuples():
        d = pd.Timestamp(r.date)
        out[int(r.year)] = fetch(geom, d)
        out[("date", int(r.year))] = d.strftime("%Y-%m-%d")
        print(f"  {pid} {r.year}: {d.date()}", "ok" if out[int(r.year)] is not None else "失败")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="典型案例图")
    ap.add_argument("--composite", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--pid", default="R0035")
    ap.add_argument("--compare", default=None, help="一块一直在养的对照地块，如 R0014")
    ap.add_argument("--activity", nargs="+", default=None)
    ap.add_argument("--s2", nargs="+", default=None)
    ap.add_argument("--metric", default="abs_excess", choices=["excess", "abs_excess", "bright_frac"])
    ap.add_argument("--thumbs", action="store_true", help="用 GEE 取每年最清晰的光学小图（需 --claim-geoms --project）")
    ap.add_argument("--claim-geoms", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--outdir", default="outputs/case_study")
    args = ap.parse_args()
    C.ACTIVITY_METRIC = args.metric
    comp = pd.read_csv(args.composite)
    th = json.load(open(args.metrics, encoding="utf-8"))["thresholds"]
    act_raw, s2_raw = _read(args.activity), _read(args.s2)
    case = annual_table(comp, args.pid)
    if case.empty:
        raise SystemExit(f"composite 里没有 {args.pid}")
    sar, s2 = scene_series(act_raw, s2_raw, args.pid)
    cmp_case = cmp_sar = cmp_s2 = None
    if args.compare:
        cmp_case = annual_table(comp, args.compare)
        cmp_sar, cmp_s2 = scene_series(act_raw, s2_raw, args.compare)
    thumbs = None
    if args.thumbs and s2_raw is not None and args.claim_geoms and args.project:
        try:
            thumbs = fetch_thumbs(args.pid, case["year"].tolist(), s2_raw, args.claim_geoms, args.project)
        except Exception as ex:  # noqa: BLE001  GEE 配额受限等情况：不出小图，其余照常
            print("取小图失败，改为不带小图：", ex)
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    case.to_csv(od / f"case_{args.pid}.csv", index=False, encoding="utf-8-sig")
    figure(case, th, od / f"fig_case_{args.pid}", sar, s2, cmp_case, cmp_sar, cmp_s2, thumbs, args.pid,
           args.compare or "")
    print(case.to_string())
    print("输出目录：", od)


if __name__ == "__main__":
    main()
