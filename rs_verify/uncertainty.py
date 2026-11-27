"""关键指标的 95% 置信区间（按地块整群自助抽样，cluster bootstrap）。

为什么按地块整群：同一个地块不同年份的结果高度相关（筏架今年在、明年多半也在），
如果把"地块-年"当成相互独立的样本，区间会被算得过窄。这里每次抽样以参考地块（ref_pid）为单位，
抽中一个地块就带上它的全部年份和它的所有偏移多边形，再算指标；重复 N 次取 2.5%–97.5% 分位。

输入（全部来自 fusion 的一次运行，不需要重跑卫星数据）：
  --composite  fusion_composite.csv（含 decision、absent1、m_sar、tex_s2、kind、gap、open_sea、split、ref_pid）
  --metrics    fusion_metrics.json（取门限 T_sar、T_s2，算"单用雷达 / 单用光学漏看率"）
  --batches    batches.csv（--trace 的批次核验结果，算模拟造假的检出率与诚实户被判可疑率）
  --visual-labels 目视判读表（可多轮）

输出：
  uncertainty.csv   指标、口径、估计值、95% 区间、分子/分母、地块数（中英文名都有，可直接进论文表格）
  uncertainty.md    同一张表的 Markdown 版

用法：
  python -m rs_verify.uncertainty --composite outputs/v4_fusion/fusion_composite.csv \
      --metrics outputs/v4_fusion/fusion_metrics.json --batches outputs/v4_fusion/batches.csv \
      --visual-labels data/visual/ailun_visual_round1.csv data/visual/ailun_visual_round2.csv \
      --outdir outputs/v4_uncertainty
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fusion import LEVEL_REVOKE, load_visual, screening_list
from .trace_sim import FLAG_LAYERS

N_BOOT = 2000


def ref_of(pid: pd.Series) -> pd.Series:
    """多边形 → 参考地块编号（偏移多边形 R0014_S500_B056 → R0014）。"""
    return pid.astype(str).str.split("_S").str[0]


def cluster_bootstrap(df: pd.DataFrame, stat, cluster: str = "cluster", n: int = N_BOOT,
                      seed: int = C.RANDOM_SEED) -> tuple[float, float, float]:
    """stat(df) -> (num, den)；返回 (估计值, 2.5%, 97.5%)。"""
    num, den = stat(df)
    est = num / den if den else float("nan")
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    if len(keys) < 2 or not den:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        b = pd.concat([groups[k] for k in pick], ignore_index=True)
        a, d = stat(b)
        if d:
            vals.append(a / d)
    if not vals:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)，X ~ Binomial(n, p)；对数空间求和，n 上千也不溢出。"""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
            for i in range(k + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """精确二项区间（Clopper-Pearson）。估计为 0% 或 100% 时自助法区间会退化成一个点（0.0–0.0%），
    这时报告这个区间：例如 0/114 → 0–3.2%，16/16 → 79.4–100%。"""
    if n <= 0:
        return float("nan"), float("nan")

    def upper(kk: int) -> float:   # 使 P(X <= kk) = alpha/2 的 p；P 随 p 单调递减，二分
        if kk >= n:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _binom_cdf(kk, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return (0.0 if k == 0 else 1.0 - upper(n - k)), upper(k)


def ratio(mask_num, mask_den):
    """生成 stat 函数：分子 = mask_num & mask_den 的行数，分母 = mask_den 的行数。"""
    def f(d):
        den = mask_den(d)
        return int((mask_num(d) & den).sum()), int(den.sum())
    return f


def visual_table(comp: pd.DataFrame, visual: pd.DataFrame, th: dict | None) -> pd.DataFrame:
    v = visual.copy()
    v["pid"] = v["pid"].astype(str)
    if "date" in v.columns:
        v = v[pd.to_datetime(v["date"]).dt.month.isin(C.S2_MONTHS)]
    v = v[v["visual"].isin(["yes", "no"])][["pid", "year", "visual"]]
    lst = screening_list(comp)[["pid", "year", "level"]]
    cols = ["pid", "year", "decision", "absent1", "m_sar", "tex_s2"]
    out = v.merge(comp[[c for c in cols if c in comp]], on=["pid", "year"], how="inner")
    out = out.merge(lst, on=["pid", "year"], how="left")
    out["revoke"] = out["level"] == LEVEL_REVOKE
    out["absent1"] = out["absent1"].astype(bool)
    if th:
        out["sar_miss"] = out["m_sar"] < th["T_sar"]
        out["s2_miss"] = out["tex_s2"] < th["T_s2"]
    out["cluster"] = ref_of(out["pid"])
    return out


def run(comp: pd.DataFrame, visual: pd.DataFrame | None = None, th: dict | None = None,
        batches: pd.DataFrame | None = None, n: int = N_BOOT) -> pd.DataFrame:
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    for c in ("gap", "open_sea", "absent1"):
        if c in comp:
            comp[c] = comp[c].astype(str).str.lower().isin(["true", "1", "1.0"]) if comp[c].dtype == object \
                else comp[c].astype(bool)
    comp["cluster"] = ref_of(comp["ref_pid"] if "ref_pid" in comp else comp["pid"])
    rows = []

    def add(key, cn, basis, df, stat):
        est, lo, hi = cluster_bootstrap(df, stat, n=n)
        num, den = stat(df)
        ex_lo, ex_hi = clopper_pearson(num, den)
        rows.append({"metric": key, "指标": cn, "basis": basis, "estimate": est, "ci_low": lo, "ci_high": hi,
                     "exact_low": ex_lo, "exact_high": ex_hi,
                     "numerator": num, "denominator": den, "n_clusters": int(df["cluster"].nunique())})

    yes = lambda d: d["visual"] == "yes"  # noqa: E731
    no = lambda d: d["visual"] == "no"    # noqa: E731
    if visual is not None and len(visual):
        vt = visual_table(comp, visual, th)
        add("farmed_listed_1season", "真有筏→进名单（一季规则，冤枉率）", "visual", vt,
            ratio(lambda d: d["absent1"], yes))
        add("empty_listed_1season", "真无筏→进名单（一季规则，查出率）", "visual", vt,
            ratio(lambda d: d["absent1"], no))
        add("list_purity_1season", "名单中确为空（一季规则）", "visual", vt, ratio(no, lambda d: d["absent1"]))
        add("farmed_listed_2season", "真有筏→吊销核查（两季规则，冤枉率）", "visual", vt,
            ratio(lambda d: d["revoke"], yes))
        add("empty_listed_2season", "真无筏→吊销核查（两季规则，查出率）", "visual", vt,
            ratio(lambda d: d["revoke"], no))
        add("list_purity_2season", "吊销核查名单中确为空", "visual", vt, ratio(no, lambda d: d["revoke"]))
        add("raft_judged_present", "真有筏→判在养", "visual", vt, ratio(lambda d: d["decision"] == "present", yes))
        add("empty_judged_present", "真无筏→判在养（放过）", "visual", vt,
            ratio(lambda d: d["decision"] == "present", no))
        if th:
            add("sar_only_miss", "单用雷达漏看真筏", "visual", vt, ratio(lambda d: d["sar_miss"], yes))
            add("s2_only_miss", "单用光学漏看真筏", "visual", vt, ratio(lambda d: d["s2_miss"], yes))

    te = comp[comp["split"] == "test"]
    farm = lambda d: d["kind"] == "farm"  # noqa: E731
    add("farm_years_listed", "参考养殖地块年→进名单", "reference(test)", te, ratio(lambda d: d["absent1"], farm))
    if "gap" in te:
        add("gap_years_listed", "区内空地块年→进名单", "reference(test)", te,
            ratio(lambda d: d["absent1"], lambda d: d["gap"]))
    add("open_sea_years_listed", "开阔海地块年→进名单", "reference(test)", te,
        ratio(lambda d: d["absent1"], lambda d: d["open_sea"]))

    if batches is not None and len(batches):
        b = batches[batches["split"] == "test"].copy()
        first = b[b["layer"] != "passed"].sort_values("seq").groupby("case")["layer"].first()
        cases = b.groupby("case").agg(scenario=("scenario", "first"), factor=("factor", "first"),
                                      pid=("pid", "first")).reset_index()
        cases["flagged"] = cases["case"].map(first).isin(FLAG_LAYERS)
        cases["cluster"] = ref_of(cases["pid"])
        for sc, cn in [("honest", "诚实户被判可疑（模拟）"), ("S2_idle", "有证不养检出（模拟）")]:
            add(f"{sc}_flagged", cn, "simulation(test)", cases,
                ratio(lambda d: d["flagged"], lambda d, s=sc: d["scenario"] == s))
        for f in (2.0, 3.0):
            add(f"launder_x{f:g}_flagged", f"洗货×{f:g}检出（模拟）", "simulation(test)", cases,
                ratio(lambda d: d["flagged"], lambda d, f=f: (d["scenario"] == "S4_launder") & (d["factor"] == f)))
    return pd.DataFrame(rows)


def degenerate(r) -> bool:
    """自助法区间退化（估计 0% / 100%，或上下限相同）→ 改报精确二项区间。"""
    return bool(np.isnan(r.ci_low) or r.ci_high - r.ci_low < 1e-9)


def to_markdown(t: pd.DataFrame) -> str:
    lines = ["| 指标 | 口径 | 估计 | 95% 区间 | 分子/分母 | 地块数 |", "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        if degenerate(r):
            ci = "—" if np.isnan(r.exact_low) else f"{r.exact_low * 100:.1f}–{r.exact_high * 100:.1f}%†"
        else:
            ci = f"{r.ci_low * 100:.1f}–{r.ci_high * 100:.1f}%"
        lines.append(f"| {r.指标} | {r.basis} | {r.estimate * 100:.1f}% | {ci} | {r.numerator}/{r.denominator} | "
                     f"{r.n_clusters} |")
    lines += ["", "区间：按参考地块整群自助抽样（同一地块不同年份不当作独立样本）。",
              "† 估计为 0% 或 100% 时自助法区间退化成一个点，改报精确二项区间（Clopper-Pearson）。"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="关键指标的整群自助置信区间")
    ap.add_argument("--composite", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--batches", default=None)
    ap.add_argument("--visual-labels", default=None, nargs="+")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--outdir", default="outputs/uncertainty")
    args = ap.parse_args()
    comp = pd.read_csv(args.composite)
    th = json.load(open(args.metrics, encoding="utf-8"))["thresholds"] if args.metrics else None
    batches = pd.read_csv(args.batches) if args.batches else None
    visual = load_visual(args.visual_labels) if args.visual_labels else None
    t = run(comp, visual, th, batches, n=args.n_boot)
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    t.to_csv(od / "uncertainty.csv", index=False, encoding="utf-8-sig")
    md = to_markdown(t)
    (od / "uncertainty.md").write_text(md, encoding="utf-8")
    print(md)
    print("输出目录：", od)


if __name__ == "__main__":
    main()
