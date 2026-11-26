"""取舍表：在"少冤枉诚实渔民"和"多抓造假"之间，给决策者一张可以挑的表和一张图。

两个旋钮：
  旋钮 A（位置/在养判断有多严）：判"没看到筏"的门限 = 真实养殖地块指标的 q 分位。
         q 越大 → 越容易判"不在养" → 抓空证/冒用越多，但诚实渔民被冤枉的也越多。
  旋钮 B（产量容差）：诚实养殖户产量比的 (1 − p) 分位作为报警线，p = 预设误报率。
         p 越大 → 报警线越低 → 抓洗货越多，诚实渔民被冤枉的也越多。

每一组 (q, p) 跑一遍完整流程（同一个登记库、同一批批次，只换门限），记录：
  诚实户被判"可疑"的比例（冤枉率）、"待复核"比例；
  各类造假被判"可疑"的比例（检出率），洗货按放大倍数分开。
  旋钮 C（荒芜认定季数）：1 = 一季没看到就进筛查名单；2 = 连续两季没看到才进（冤枉更少，但晚一年发现）。
另外给三个推荐档位：宽松（几乎不冤枉）/ 均衡 / 严格（多抓），每种荒芜规则各一套。
如给 --visual-labels，另按目视真值统计"图上有筏却进名单"（真冤枉）和"图上无筏进了名单"（真检出）。
--s2-agg median|persist 切换光学多景合成方式；--idle-place gap|open_sea 切换空挂证模拟位置。

输出（outputs/<area>_tradeoff/）：
  tradeoff_table.csv     全部组合（中文列名，可直接放论文附表）
  tradeoff_presets.csv   三个推荐档位
  tradeoff.png           (a) 冤枉率 vs 检出率（每类造假一条线）(b) 档位对比柱状图

用法（参数同 fusion.py）：
  python -m rs_verify.tradeoff --area ailun --claims ... --claim-geoms ... --raft-reference ... \
      --activity data/activity/ailun_activity_*.csv --s2 data/activity_s2/ailun_s2_*.csv --metric abs_excess
"""
from __future__ import annotations

import argparse
import glob
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
from . import trace_sim  # noqa: E402

Q_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]      # 旋钮 A
P_GRID = [0.005, 0.01, 0.02, 0.05, 0.10]             # 旋钮 B
PRESETS = {"宽松（几乎不冤枉）": (0.01, 0.005), "均衡": (0.05, 0.02), "严格（多抓）": (0.20, 0.10)}

SCEN_CN = {"honest": "诚实户", "S1_outside": "区外冒充", "S2_idle": "有证不养", "S3_neighbour": "冒用邻居地块",
           "S4_launder": "洗货", "S5_collusion": "借账号"}


def visual_truth(f: pd.DataFrame, visual: pd.DataFrame | None) -> pd.DataFrame | None:
    """目视真值（只用海带季图片上的 yes/no）与本次判定对齐。"""
    if visual is None or not len(visual):
        return None
    v = visual.copy()
    v["pid"] = v["pid"].astype(str)
    v = v.drop(columns=[c for c in ("decision",) if c in v.columns])
    if "date" in v.columns:
        v = v[pd.to_datetime(v["date"]).dt.month.isin(C.S2_MONTHS)]
    v = v[v["visual"].isin(["yes", "no"])]
    return v.merge(f[["pid", "year", "decision"]], on=["pid", "year"], how="inner")


def one_setting(f0: pd.DataFrame, am: dict, claims, cover, claim_geoms, area, q: float, p: float,
                seasons: int = 1, visual: pd.DataFrame | None = None) -> dict:
    th = fusion.calibrate(f0, low_q=q)
    f = fusion.apply_idle_rule(fusion.decide(f0, th), seasons)
    old = C.QTY_FALSE_ALARM_TARGET
    C.QTY_FALSE_ALARM_TARGET = p
    try:
        tm, _, _, _ = trace_sim.run_with_comp(f, am, claims, cover, claim_geoms, area)
    finally:
        C.QTY_FALSE_ALARM_TARGET = old
    te = f[f["split"] == "test"]
    row = {
        "idle_seasons": seasons, "q_absent": q, "p_quantity": p,
        "farm_years_absent": float((te.loc[te["kind"] == "farm", "decision"] == "absent").mean()),
        "gap_years_absent": float((te.loc[te["gap"], "decision"] == "absent").mean()),
        "open_sea_years_absent": float((te.loc[te["open_sea"], "decision"] == "absent").mean()),
        "honest_flagged": tm["honest"]["flagged_suspicious"],
        "honest_review": tm["honest"]["caught_by"]["review"],
    }
    for sc in ("S1_outside", "S2_idle", "S3_neighbour", "S4_launder", "S5_collusion"):
        if sc in tm:
            row[f"{sc}_flagged"] = tm[sc]["flagged_suspicious"]
    for k in ("S4_launder",):
        for r, v in tm.get(k, {}).get("flagged_by_factor", {}).items():
            row[f"{k}_x{r:g}_flagged"] = v
    vt = visual_truth(f, visual)
    if vt is not None and len(vt):
        yes, no = vt[vt["visual"] == "yes"], vt[vt["visual"] == "no"]
        row["visual_farmed_listed"] = float((yes["decision"] == "absent").mean()) if len(yes) else float("nan")
        row["visual_empty_listed"] = float((no["decision"] == "absent").mean()) if len(no) else float("nan")
        listed = vt[vt["decision"] == "absent"]
        row["visual_list_truly_empty"] = float((listed["visual"] == "no").mean()) if len(listed) else float("nan")
    return row


def to_cn(df: pd.DataFrame) -> pd.DataFrame:
    ren = {"idle_seasons": "旋钮C_荒芜认定季数", "q_absent": "旋钮A_判没筏分位q", "p_quantity": "旋钮B_产量预设误报p",
           "farm_years_absent": "养殖地块年_判不在养", "gap_years_absent": "区内空地块年_判不在养",
           "open_sea_years_absent": "开阔海年_判不在养",
           "visual_farmed_listed": "目视有筏_进名单(真冤枉)", "visual_empty_listed": "目视无筏_进名单(真检出)",
           "visual_list_truly_empty": "名单中目视确为空",
           "honest_flagged": "诚实户_被判可疑(冤枉率)", "honest_review": "诚实户_待复核"}
    for sc, cn in SCEN_CN.items():
        if sc != "honest":
            ren[f"{sc}_flagged"] = f"{cn}_检出率"
    for c in df.columns:
        if c.startswith("S4_launder_x"):
            ren[c] = "洗货" + c.split("_x")[1].replace("_flagged", "") + "倍_检出率"
    out = df.rename(columns=ren)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out


def preset_en(name: str) -> str:
    en = {"宽松（几乎不冤枉）": "Lenient", "均衡": "Balanced", "严格（多抓）": "Strict"}
    base, _, sn = name.partition("｜")
    return f"{en.get(base, base)}, {sn.replace('季', '-season')}" if sn else en.get(base, base)


def plot(df: pd.DataFrame, presets: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    lines = [("S2_idle_flagged", "Idle permit"), ("S4_launder_x1.5_flagged", "Laundering ×1.5"),
             ("S4_launder_x2_flagged", "Laundering ×2"), ("S4_launder_x3_flagged", "Laundering ×3")]
    marker = {1: "o", 2: "s"}
    for j, (col, lab) in enumerate(lines):
        if col not in df:
            continue
        for sn, d in df.groupby("idle_seasons"):
            d = d.sort_values("honest_flagged")
            axes[0].plot(d["honest_flagged"] * 100, d[col] * 100, marker.get(int(sn), "^"), ms=4, alpha=0.6,
                         color=f"C{j}", label=f"{lab} ({int(sn)}-season rule)")
    for i, (name, r) in enumerate(presets.iterrows()):
        axes[0].axvline(r["honest_flagged"] * 100, color="gray", ls=":", lw=0.8)
        axes[0].text(r["honest_flagged"] * 100, 101 + 3.5 * i, preset_en(name), fontsize=7, ha="left")
    axes[0].set_xlabel("Honest farmers flagged as suspicious (%)")
    axes[0].set_ylabel("Fraud flagged as suspicious (%)")
    axes[0].set_ylim(0, 125)
    axes[0].set_title("Trade-off: each dot = one threshold setting")
    axes[0].legend(fontsize=8)
    cols = [c for c, _ in [("honest_flagged", ""), ("S2_idle_flagged", ""), ("S4_launder_x2_flagged", ""),
                           ("S4_launder_x3_flagged", "")] if c in presets]
    labs = {"honest_flagged": "Honest\n(false alarm)", "S2_idle_flagged": "Idle\npermit",
            "S4_launder_x2_flagged": "Launder\n×2", "S4_launder_x3_flagged": "Launder\n×3"}
    x = np.arange(len(cols))
    w = 0.8 / max(len(presets), 1)
    for i, (name, r) in enumerate(presets.iterrows()):
        axes[1].bar(x + i * w, [r[c] * 100 for c in cols], w, label=preset_en(name))
    axes[1].set_xticks(x + w * (len(presets) - 1) / 2)
    axes[1].set_xticklabels([labs[c] for c in cols], fontsize=8)
    axes[1].set_ylabel("Flagged (%)")
    axes[1].set_title("Preset operating points")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run(claims, act_raw, cover, s2_raw, claim_geoms, area, q_grid=Q_GRID, p_grid=P_GRID,
        seasons_grid=(1, 2), visual=None, s2_agg=None):
    f0, _, am = fusion.build(claims, act_raw, cover, s2_raw, s2_agg=s2_agg, idle_seasons=1)
    f0 = f0.drop(columns=["decision", "present", "status", "evidence"])
    rows = []
    grid = sorted({(q, p) for q in q_grid for p in p_grid} | set(PRESETS.values()))
    for sn in seasons_grid:
        for q, p in grid:
            rows.append(one_setting(f0, am, claims, cover, claim_geoms, area, q, p, sn, visual))
            print(f"季数={sn} q={q:<6} p={p:<6} 冤枉率={rows[-1]['honest_flagged']:.3f}")
    df = pd.DataFrame(rows)
    presets = pd.DataFrame({f"{name}｜{sn}季": df[(df.idle_seasons == sn) & (df.q_absent == q) &
                                                (df.p_quantity == p)].iloc[0]
                            for sn in seasons_grid for name, (q, p) in PRESETS.items()}).T
    return df, presets


def main() -> None:
    ap = argparse.ArgumentParser(description="冤枉率 vs 检出率 取舍表")
    annual.add_common_args(ap)
    ap.add_argument("--s2", required=True, nargs="+")
    ap.add_argument("--s2-agg", default=None, choices=["median", "persist"])
    ap.add_argument("--idle-seasons", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument("--idle-place", default=None, choices=["gap", "open_sea"])
    ap.add_argument("--visual-labels", default=None, nargs="+")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.metric:
        C.ACTIVITY_METRIC = args.metric
    if args.idle_place:
        C.IDLE_PLACEMENT = args.idle_place
    visual = fusion.load_visual(args.visual_labels) if args.visual_labels else None
    claims, act_raw, cover = annual.load_inputs(args)
    files = sorted({x for p in args.s2 for x in (glob.glob(p) or [p])})
    s2_raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df, presets = run(claims, act_raw, cover, s2_raw, geo.load_geojson(args.claim_geoms), args.area,
                      seasons_grid=tuple(args.idle_seasons), visual=visual, s2_agg=args.s2_agg)

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_tradeoff")
    od.mkdir(parents=True, exist_ok=True)
    to_cn(df).to_csv(od / "tradeoff_table.csv", index=False, encoding="utf-8-sig")
    to_cn(presets.reset_index(names="档位")).to_csv(od / "tradeoff_presets.csv", index=False, encoding="utf-8-sig")
    plot(df, presets, od / "tradeoff.png")
    print(to_cn(presets.reset_index(names="档位")).T.to_string())
    print("输出目录：", od)


if __name__ == "__main__":
    main()
