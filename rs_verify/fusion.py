"""雷达 + 光学融合的年度"在养"判定——精度优先（宁可待复核，不冤枉诚实渔民）。

两个证人：
  雷达 S1（全天候，但会漏看）：全年好海况影像的指标中位数 m_sar（annual.py，窗口 = 全年）。
  光学 S2（看得清，但怕云雾）：每个海带季（12–5 月）只挑几张"好图"——多边形内晴空 ≥ 95%、
                               雾霾最轻（蓝波段最低）的最多 K 张（默认 5），取筏格纹理的中位数 tex_s2。
                               6–11 月海带收完后筏架在光学图上几乎看不见，不用。

判定（每个地块 × 每年，三种结论）：
  在养（通过）      任一证人明确看到有筏：m_sar ≥ T_sar 或 tex_s2 ≥ T_s2
  不在养（可疑）    两个证人都明确没看到：有好光学图，且 tex_s2 < L_s2 且 m_sar < L_sar
  待复核            其余情况（证据不足或互相矛盾）——系统不下"可疑"结论，提示补图/人工核查

门限只用标定组地块定：
  T（判"在养"）= 养殖 vs 空海面的 Youden 最优点；
  L（判"没有"）= 养殖地块的低分位（默认 2%）——只有比 98% 的真实养殖地块都"更空"才敢说没有。
所以诚实渔民被判"可疑"的概率 ≈ 两个证人同时极端漏看，被压到很低。

评估：检验组地块上，按参考标签（养殖 / 空海面）统计三种结论的比例；
      如提供目视判读表（--visual-labels，列 pid, year, visual = yes/no/unclear），另按目视真值统计。
可选 --trace：把融合结论接入完整流程模拟（登记 + 批次 + 三层核验）。

输出（outputs/<area>_fusion/）：
  fusion_metrics.json, fusion_composite.csv（每个多边形 × 年的两证人数值与结论）, fusion_check.png
  （--trace 时另有 trace_* 文件，同 trace_sim.py）

用法：
  python -m rs_verify.fusion --area ailun --claims data/claims/ailun_claims.csv \
      --claim-geoms data/claims/ailun_claim_geoms.geojson \
      --raft-reference data/reference/ailun_fu2021_reference.geojson \
      --activity data/activity/ailun_activity_*.csv --s2 data/activity_s2/ailun_s2_*.csv \
      --metric abs_excess --trace
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
from . import geo  # noqa: E402
from . import trace_sim  # noqa: E402

PIXEL_AREA_M2 = 100.0
FULL_YEAR = list(range(1, 13))


# ---------------------------------------------------------------------------
# 光学：挑好图
# ---------------------------------------------------------------------------
def s2_good_scenes(s2_raw: pd.DataFrame, months: list[int] | None = None, min_clear: float = C.S2_MIN_CLEAR,
                   min_cov: float = 0.8) -> pd.DataFrame:
    """海带季内、多边形晴空且覆盖完整的光学景；year = 海带季的"结束年"（12 月算下一年）。"""
    months = months or C.S2_MONTHS
    a = s2_raw.copy()
    a["pid"] = a["pid"].astype(str)
    a["date"] = pd.to_datetime(a["date"])
    a = a[a["date"].dt.month.isin(months)]
    a["year"] = annual.season_year(a["date"], months)
    for c in ("tex", "clear", "b2", "n_pix", "area_m2"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["cov"] = a["n_pix"] * PIXEL_AREA_M2 / a["area_m2"]
    a = a[(a["clear"] >= min_clear) & (a["cov"] >= min_cov)].dropna(subset=["tex", "b2"])
    return a.sort_values("n_pix", ascending=False).drop_duplicates(["pid", "date"])


def s2_annual(s2_raw: pd.DataFrame, years: list[int], k_best: int = C.S2_BEST_SCENES,
              min_clear: float = C.S2_MIN_CLEAR, min_cov: float = 0.8, months: list[int] | None = None) -> pd.DataFrame:
    a = s2_good_scenes(s2_raw, months, min_clear, min_cov)
    a = a[a["year"].isin(years)]
    return _best_k(a, k_best)


def _best_k(a: pd.DataFrame, k_best: int) -> pd.DataFrame:
    if a.empty:
        return pd.DataFrame(columns=["pid", "year", "tex_s2", "n_s2", "b2_s2"])
    # 每个多边形每年：雾霾最轻（b2 最低）的 K 张
    a = a.sort_values("b2").groupby(["pid", "year"]).head(k_best)
    return a.groupby(["pid", "year"]).agg(tex_s2=("tex", "median"), n_s2=("tex", "size"),
                                          b2_s2=("b2", "median")).reset_index()


# ---------------------------------------------------------------------------
# 融合判定
# ---------------------------------------------------------------------------
def youden(pos, neg) -> float:
    return annual.youden(np.asarray(pos, float), np.asarray(neg, float))


def calibrate(f: pd.DataFrame, low_q: float = C.FUSION_LOW_QUANTILE) -> dict:
    cal = f[f["split"] == "calib"]
    farm, sea = cal[cal["kind"] == "farm"], cal[cal["open_sea"]]
    fs, ss = farm.dropna(subset=["m_sar"]), sea.dropna(subset=["m_sar"])
    fo, so = farm.dropna(subset=["tex_s2"]), sea.dropna(subset=["tex_s2"])
    th = {
        "T_sar": youden(fs["m_sar"], ss["m_sar"]),
        "L_sar": float(np.quantile(fs["m_sar"], low_q)),
        "T_s2": youden(fo["tex_s2"], so["tex_s2"]) if len(fo) and len(so) else float("nan"),
        "L_s2": float(np.quantile(fo["tex_s2"], low_q)) if len(fo) else float("nan"),
        "low_quantile": low_q,
    }
    # "没看到"的门限不能高于"看到了"的门限
    th["L_sar"] = min(th["L_sar"], th["T_sar"])
    th["L_s2"] = min(th["L_s2"], th["T_s2"])
    return th


def decide(f: pd.DataFrame, th: dict) -> pd.DataFrame:
    f = f.copy()
    has_sar = f["m_sar"].notna()
    has_s2 = f["tex_s2"].notna()
    sar_yes = has_sar & (f["m_sar"] >= th["T_sar"])
    s2_yes = has_s2 & (f["tex_s2"] >= th["T_s2"])
    sar_no = has_sar & (f["m_sar"] < th["L_sar"])
    s2_no = has_s2 & (f["tex_s2"] < th["L_s2"])
    f["decision"] = np.select([sar_yes | s2_yes, sar_no & s2_no, ~has_sar & ~has_s2],
                              ["present", "absent", "insufficient"], default="review")
    f["present"] = np.select([f["decision"] == "present", f["decision"] == "absent"], [1.0, 0.0], default=np.nan)
    f["status"] = np.select([f["decision"].isin(["present", "absent"]), f["decision"] == "review"],
                            ["ok", "review"], default="insufficient")
    f["evidence"] = np.select([sar_yes & s2_yes, sar_yes, s2_yes, sar_no & s2_no],
                              ["雷达+光学均见筏", "雷达见筏", "光学见筏", "雷达+光学均未见筏"], default="证据不足/矛盾")
    return f


def rates(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {"n": 0}
    return {"n": int(n), **{k: float((d["decision"] == k).mean()) for k in ["present", "absent", "review", "insufficient"]}}


def evaluate(f: pd.DataFrame, visual: pd.DataFrame | None = None) -> dict:
    te = f[f["split"] == "test"]
    sh = te[te["kind"] == "shifted"]
    m = {
        "test_farm": rates(te[te["kind"] == "farm"]),
        "test_open_sea": rates(te[te["open_sea"]]),
        "test_shift_clean_absent_rate": {int(d): float((g["decision"] == "absent").mean())
                                         for d, g in sh[sh["overlap_ref"] < 0.1].groupby("shift_m")},
        "test_shift_clean_not_passed": {int(d): float((g["decision"] != "present").mean())
                                        for d, g in sh[sh["overlap_ref"] < 0.1].groupby("shift_m")},
        "s2_coverage_farm_years": float(te.loc[te["kind"] == "farm", "tex_s2"].notna().mean()),
        "sar_only": {
            "farm_absent": float((te.loc[te["kind"] == "farm", "m_sar"] < f.attrs["th"]["T_sar"]).mean()),
            "sea_present": float((te.loc[te["open_sea"], "m_sar"] >= f.attrs["th"]["T_sar"]).mean()),
        },
    }
    if visual is not None and len(visual):
        v = visual.copy()
        v["pid"] = v["pid"].astype(str)
        v = v.drop(columns=[c for c in ("decision", "m_sar", "tex_s2") if c in v.columns])  # 以本次判定为准
        v = v.merge(f[["pid", "year", "decision", "m_sar", "tex_s2"]], on=["pid", "year"], how="inner")
        if "date" in v.columns:  # 只用海带季图片上的判读（夏秋图上筏架看不见，"no" 不可信）
            v = v[pd.to_datetime(v["date"]).dt.month.isin(C.S2_MONTHS)]
        v = v[v["visual"].isin(["yes", "no"])]
        m["visual"] = {"n_labelled": int(len(v)),
                       "by_decision": {k: {"n": int(len(g)), "raft_yes_share": float((g["visual"] == "yes").mean())}
                                       for k, g in v.groupby("decision")},
                       "absent_but_raft_visible": int(((v["decision"] == "absent") & (v["visual"] == "yes")).sum()),
                       "raft_yes": rates(v[v["visual"] == "yes"]),
                       "raft_no": rates(v[v["visual"] == "no"])}
        th = f.attrs["th"]
        yes, no = v[v["visual"] == "yes"], v[v["visual"] == "no"]
        m["visual"]["sar_only_miss_rate"] = float((yes["m_sar"] < th["T_sar"]).mean()) if len(yes) else float("nan")
        m["visual"]["s2_only_miss_rate"] = float((yes["tex_s2"] < th["T_s2"]).mean()) if len(yes) else float("nan")
    return m


def plot(f: pd.DataFrame, th: dict, m: dict, out: Path) -> None:
    te = f[f["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, d, c in [("Farm parcels", te[te["kind"] == "farm"], "tab:green"),
                       ("Open sea", te[te["open_sea"]], "tab:blue")]:
        d = d.dropna(subset=["m_sar", "tex_s2"])
        axes[0].scatter(d["m_sar"], d["tex_s2"], s=6, alpha=0.5, color=c, label=name)
    for v, ls in [(th["T_sar"], "-"), (th["L_sar"], ":")]:
        axes[0].axvline(v, color="k", ls=ls, lw=0.8)
    for v, ls in [(th["T_s2"], "-"), (th["L_s2"], ":")]:
        axes[0].axhline(v, color="k", ls=ls, lw=0.8)
    axes[0].set_xlabel("SAR annual median (abs. excess)")
    axes[0].set_ylabel("S2 raft texture (best scenes)")
    axes[0].set_title("Solid: 'present' thresholds; dotted: 'absent' thresholds")
    axes[0].legend(fontsize=8)
    cats = ["present", "review", "insufficient", "absent"]
    colors = {"present": "#dddddd", "review": "khaki", "insufficient": "tab:gray", "absent": "tab:red"}
    groups = {"Farm": m["test_farm"], "Open sea": m["test_open_sea"]}
    if "visual" in m:
        groups.update({"Visual: raft": m["visual"]["raft_yes"], "Visual: none": m["visual"]["raft_no"]})
    bottom = np.zeros(len(groups))
    for cat in cats:
        vals = np.array([g.get(cat, 0.0) for g in groups.values()])
        axes[1].bar(list(groups), vals, bottom=bottom, color=colors[cat], label=cat)
        bottom += vals
    axes[1].set_ylabel("Share of parcel-years (test)")
    axes[1].set_title("Fused decision")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build(claims, act_raw, cover, s2_raw, visual=None):
    am, sep, comp = annual.run(claims, act_raw, cover, window=FULL_YEAR)
    s2 = s2_annual(s2_raw, am["years"])
    f = comp.rename(columns={"m": "m_sar", "n": "n_sar"})
    f["m_sar"] = np.where(f["status"] == "ok", f["m_sar"], np.nan)
    f = f.drop(columns=["present", "status"]).merge(s2, on=["pid", "year"], how="left")
    # 没有雷达行、只有光学的多边形-年（少见）也补进来
    th = calibrate(f)
    f = decide(f, th)
    f.attrs["th"] = th
    m = evaluate(f, visual)
    m = {"thresholds": th, "sar_annual": am, **m}
    return f, m, am


def load_visual(paths) -> pd.DataFrame:
    """读入多轮目视判读表并合并；同一 pid-year 重复时保留后一轮。"""
    vs = []
    for i, p in enumerate(paths):
        v = pd.read_csv(p)
        v["round"] = i + 1
        vs.append(v)
    v = pd.concat(vs, ignore_index=True)
    v["pid"] = v["pid"].astype(str)
    return v.drop_duplicates(["pid", "year"], keep="last").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="雷达 + 光学融合判定（精度优先）")
    annual.add_common_args(ap)
    ap.add_argument("--s2", required=True, nargs="+", help="gee_s2_activity 导出的 CSV（可用 *）")
    ap.add_argument("--visual-labels", default=None, nargs="+",
                    help="目视判读表（pid, year, visual=yes/no/unclear），可给多轮，同一 pid-year 以后一轮为准")
    ap.add_argument("--trace", action="store_true", help="接入完整流程模拟")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.metric:
        C.ACTIVITY_METRIC = args.metric
    claims, act_raw, cover = annual.load_inputs(args)
    files = sorted({f for p in args.s2 for f in (glob.glob(p) or [p])})
    s2_raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    visual = load_visual(args.visual_labels) if args.visual_labels else None
    f, m, am = build(claims, act_raw, cover, s2_raw, visual)

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_fusion")
    od.mkdir(parents=True, exist_ok=True)
    f.to_csv(od / "fusion_composite.csv", index=False, encoding="utf-8-sig")
    with open(od / "fusion_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=2)
    plot(f, m["thresholds"], m, od / "fusion_check.png")
    print(json.dumps({k: v for k, v in m.items() if k != "sar_annual"}, ensure_ascii=False, indent=2))

    if args.trace:
        tm, tab, reg, v = trace_sim.run_with_comp(f, am, claims, cover, geo.load_geojson(args.claim_geoms), args.area)
        reg.to_csv(od / "registry.csv", index=False, encoding="utf-8-sig")
        v.to_csv(od / "batches.csv", index=False, encoding="utf-8-sig")
        tab.rename(columns=trace_sim.LAYER_CN).to_csv(od / "trace_scenario_table.csv", encoding="utf-8-sig")
        with open(od / "trace_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(tm, fh, ensure_ascii=False, indent=2)
        trace_sim.plot(tab, tm, od / "trace_layers.png")
        print(tab.round(3).to_string())
    print("输出目录：", od)


if __name__ == "__main__":
    main()
