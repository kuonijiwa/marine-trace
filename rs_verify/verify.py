"""核验算法：给一份产地声明打分。

三项检查（通俗说法）：
  空间分 S   声明的那块海，在声明时段里"有设施"的影像占比。
             挪到空海面的假围栏 → 几乎每景都没设施 → S≈0。
  时间分 T   声明的月份，是不是这块海"平时有设施"的月份。
             拿这块海多年的"月度活动规律"来比，错报季节 → T 低。
  产量比 Q   声明数量 ÷ 卫星看到的有效养殖规模。
             Q>1 说明卖出去的比这片海能产的还多 → 可能把外面的货洗成牧场货。

产地可信度 = w*S + (1-w)*T；产量比单独作为第三道检查。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PIXEL_AREA_M2 = 100.0  # 10 m × 10 m


def activity_metric(act: pd.DataFrame, metric: str | None = None) -> pd.Series:
    """每景每个多边形的"活动指标"。metric 为 None 时用 config.ACTIVITY_METRIC（运行时读取，命令行可改）。

    excess      = lin_ratio − 1：多边形平均雷达亮度比海面高出多少（相对值）。空海面≈0。
                  缺点：浮筏回波大致固定，海面回波随风浪变，相对值在平静海面被放大、在大风天被压扁。
    abs_excess  = (lin_ratio − 1) × 海面背景（线性）× 1000 = 多边形平均回波 − 海面回波（绝对值，×10⁻³）。
                  混合模型：≈ 覆盖率 × (浮筏回波 − 海面回波)；海面回波远小于浮筏时基本不随风浪变。
    bright_frac = 多边形里"比背景亮 6 dB"的像元占比。（适合网箱等强回波结构）
    数据里没有 lin_ratio 列时（旧版导出），自动退回 bright_frac。
    """
    metric = metric or C.ACTIVITY_METRIC
    if metric in ("excess", "abs_excess") and "lin_ratio" in act.columns:
        ex = pd.to_numeric(act["lin_ratio"], errors="coerce") - 1.0
        if metric == "excess":
            return ex
        bg_lin = 10 ** (pd.to_numeric(act["bg_db"], errors="coerce") / 10)
        return ex * bg_lin * 1000
    return pd.to_numeric(act["bright_frac"], errors="coerce")


def prepare_activity(
    act: pd.DataFrame,
    active_threshold: float = C.ACTIVE_FRAC_THRESHOLD,
    min_coverage: float = 0.8,
    min_bg_db: float | None = C.MIN_BACKGROUND_DB,
) -> pd.DataFrame:
    """清洗活动表：去掉多边形只被影像部分覆盖（或大半在陆地上）的景、海面太平静（接近噪声底）的景，
    标记"有设施"景。"""
    a = act.copy()
    if min_bg_db is not None and "bg_db" in a.columns:
        a = a[pd.to_numeric(a["bg_db"], errors="coerce") >= min_bg_db]
    a["pid"] = a["pid"].astype(str)
    a["date"] = pd.to_datetime(a["date"])
    a["metric"] = activity_metric(a)
    a = a.dropna(subset=["metric"])
    a["coverage"] = a["n_pix"] * PIXEL_AREA_M2 / a["area_m2"]
    a = a[a["coverage"] >= min_coverage]
    # 同一轨道相邻两景（同一天、前后衔接）的重叠带会让同一多边形同一天出现两行：只留覆盖最全的一行
    a = a.sort_values("n_pix", ascending=False).drop_duplicates(["pid", "date"]).sort_values(["pid", "date"])
    a["active"] = a["metric"] >= active_threshold
    a["month"] = a["date"].dt.month
    return a


def monthly_profile(act: pd.DataFrame) -> pd.DataFrame:
    """每个多边形的"月度活动规律"：各月份有设施的影像占比（多年平均）。"""
    return act.groupby(["pid", "month"])["active"].mean().unstack("month").reindex(columns=range(1, 13))


def full_stock_fraction(act: pd.DataFrame, claims: pd.DataFrame, q: float = 0.9) -> float:
    """标定"满产时亮像元占比"：真实声明窗口内亮像元占比中位数的 90% 分位。"""
    vals = []
    a_by = dict(tuple(act.groupby("pid")))
    for c in claims[claims["label"] == "true"].itertuples():
        a = a_by.get(c.pid)
        if a is None:
            continue
        w = a[(a["date"] >= c.start) & (a["date"] <= c.end)]
        if len(w):
            vals.append(w["metric"].median())
    return float(np.quantile(vals, q)) if vals else float("nan")


def window_months(start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    return sorted(set(pd.date_range(start.replace(day=1), end, freq="MS").month))


def verify_claims(
    act: pd.DataFrame,
    claims: pd.DataFrame,
    spatial_weight: float = C.SPATIAL_WEIGHT,
    min_scenes: int = C.MIN_SCENES_IN_WINDOW,
    f_full: float | None = None,
) -> pd.DataFrame:
    """对每份声明打分。act 须先经过 prepare_activity。"""
    cl = claims.copy()
    cl["pid"] = cl["pid"].astype(str)
    cl["start"] = pd.to_datetime(cl["start"])
    cl["end"] = pd.to_datetime(cl["end"])
    if f_full is None:
        f_full = full_stock_fraction(act, cl)

    prof = monthly_profile(act)
    prof_max = prof.max(axis=1)
    a_by = dict(tuple(act.groupby("pid")))

    out = []
    for c in cl.itertuples(index=False):
        a = a_by.get(c.pid)
        w = a[(a["date"] >= c.start) & (a["date"] <= c.end)] if a is not None else None
        n = 0 if w is None else len(w)
        rec = {"claim_id": c.claim_id, "n_scenes": n}
        if n < min_scenes:
            rec.update(spatial=np.nan, temporal=np.nan, confidence=np.nan, eff_frac=np.nan,
                       qty_ratio=np.nan, status="insufficient")
            out.append(rec)
            continue
        S = float(w["active"].mean())
        months = window_months(c.start, c.end)
        pm = prof.loc[c.pid, months] if c.pid in prof.index else pd.Series(dtype=float)
        pmax = prof_max.get(c.pid, np.nan)
        T = float(pm.mean() / pmax) if (pmax and pmax > 0 and pm.notna().any()) else 0.0
        conf = spatial_weight * S + (1 - spatial_weight) * T
        eff = float(np.clip(w["metric"].median() / f_full, 0, 1)) if f_full and f_full > 0 else np.nan
        qr = float(c.quantity_rel / eff) if eff and eff > 0 else np.inf
        rec.update(spatial=S, temporal=T, confidence=conf, eff_frac=eff, qty_ratio=qr, status="ok")
        out.append(rec)

    res = pd.DataFrame(out)
    return cl.merge(res, on="claim_id", how="left")


def calibrate_active_threshold(act_raw: pd.DataFrame, claims: pd.DataFrame) -> float:
    """用数据自己选"有设施"阈值：
    正样本 = 参考多边形在真实养殖季的景；负样本 = 平移 ≥2 km 且不压别人养殖区的景。
    取让 (检出率 − 误报率) 最大的阈值（Youden 指数）。"""
    a = prepare_activity(act_raw, active_threshold=0.0)
    cl = claims.copy()
    cl["start"] = pd.to_datetime(cl["start"])
    cl["end"] = pd.to_datetime(cl["end"])
    a_by = dict(tuple(a.groupby("pid")))

    def collect(rows):
        vals = []
        for c in rows.itertuples():
            x = a_by.get(str(c.pid))
            if x is not None:
                vals.append(x[(x["date"] >= c.start) & (x["date"] <= c.end)]["metric"].to_numpy())
        return np.concatenate(vals) if vals else np.array([])

    pos = collect(cl[cl["label"] == "true"])
    neg = collect(cl[(cl["label"] == "fake_location") & (cl["shift_m"] >= 2000) & (cl["overlap_ref"] < 0.05)])
    if len(pos) == 0 or len(neg) == 0:
        return C.ACTIVE_FRAC_THRESHOLD
    # 候选阈值取正负样本合并后的分位点（指标量纲不同，固定网格不通用）
    grid = np.unique(np.quantile(np.r_[pos, neg], np.linspace(0.01, 0.99, 197)))
    j = [(pos >= t).mean() - (neg >= t).mean() for t in grid]
    return float(grid[int(np.argmax(j))])
