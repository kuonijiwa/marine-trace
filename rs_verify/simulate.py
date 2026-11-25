"""合成数据：在没有 GEE 结果之前，用来跑通整条流程、检查代码有没有错。

⚠️ 这里生成的全是假数据，只用于测试程序，绝不能当作论文结果。

模拟设定（尽量贴近浮筏海带的情况）：
  - 若干矩形筏区（约 400 m × 150 m），成片排列，中间有航道；
  - 海带季（11 月→次年 5 月）亮像元占比约 0.45，淡季约 0.05；
  - 空海面亮像元占比很低，偶尔有船经过（突然变亮）；
  - 平移后的假围栏：亮像元占比 = 与参考筏区的重叠比例 × 筏区值 + 其余 × 海面值。

用法：
  python -m rs_verify.simulate --outdir data/sim
  python -m rs_verify.experiment --area ailun --claims data/sim/ailun_claims.csv \
      --activity data/sim/ailun_activity.csv --calibrate --outdir outputs/sim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import geo
from .claims import build_claims


def synthetic_reference(n_rows: int = 4, n_cols: int = 5, lon0: float = 122.62, lat0: float = 37.18) -> dict:
    """一片 n_rows × n_cols 的矩形筏区。"""
    dlon, dlat = geo._deg_per_m(lat0)
    w, h, gap_x, gap_y = 400.0, 150.0, 120.0, 200.0
    feats = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = lon0 + c * (w + gap_x) * dlon
            y0 = lat0 + r * (h + gap_y) * dlat
            x1, y1 = x0 + w * dlon, y0 + h * dlat
            ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"product": "kelp"}})
    return geo.feature_collection(feats)


def in_kelp_season(month: np.ndarray) -> np.ndarray:
    m0, m1 = C.PRODUCT_CALENDAR["kelp"]
    return (month >= m0) | (month <= m1)


def synthetic_activity(geoms: dict, seed: int = C.RANDOM_SEED, revisit_days: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(C.START_DATE, C.END_DATE, freq=f"{revisit_days}D")
    months = dates.month.to_numpy()
    season = in_kelp_season(months)

    # 每个参考筏区的"放养程度"（满产=1）
    ref_ids = [f["properties"]["pid"] for f in geoms["features"] if f["properties"]["kind"] == "reference"]
    stocking = {pid: rng.uniform(0.85, 1.0) for pid in ref_ids}

    def raft_series(pid):
        base = np.where(season, 0.45 * stocking[pid], 0.05)
        return np.clip(base + rng.normal(0, 0.06, len(dates)), 0, 1)

    def sea_series():
        s = rng.beta(1, 40, len(dates))
        ships = rng.random(len(dates)) < 0.03
        return np.clip(s + ships * rng.uniform(0.1, 0.3, len(dates)), 0, 1)

    rows = []
    for f in geoms["features"]:
        p = f["properties"]
        pid, area = p["pid"], p["area_m2"]
        if p["kind"] == "reference":
            frac = raft_series(pid)
        else:
            ov = p.get("overlap_ref", 0.0)
            # 落在"某个"参考筏区上：借用其所属参考筏区的季节曲线近似
            frac = ov * raft_series(p["ref_pid"]) + (1 - ov) * sea_series()
        full_pix = area / 100.0
        cover = np.where(rng.random(len(dates)) < 0.05, rng.uniform(0.3, 0.7, len(dates)), 1.0)
        keep = rng.random(len(dates)) > 0.03  # 少量缺景
        for d, fr, cv, k in zip(dates, frac, cover, keep):
            if k:
                rows.append({"pid": pid, "date": d.strftime("%Y-%m-%d"), "bright_frac": round(float(fr), 4),
                             "n_pix": int(full_pix * cv), "orbit_pass": "ASCENDING", "rel_orbit": 0,
                             "platform": "A", "water_frac": 1.0, "area_m2": area})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成测试数据（仅测试用）")
    ap.add_argument("--outdir", default=str(Path(C.DATA_DIR) / "sim"))
    ap.add_argument("--n-bearings", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ref = synthetic_reference()
    geo.save_geojson(ref, out / "ailun_reference.geojson")
    claims, geoms = build_claims(ref, "ailun", n_bearings=args.n_bearings)
    claims.to_csv(out / "ailun_claims.csv", index=False)
    geo.save_geojson(geoms, out / "ailun_claim_geoms.geojson")
    act = synthetic_activity(geoms)
    act.to_csv(out / "ailun_activity.csv", index=False)
    print(f"参考多边形 {len(ref['features'])} 个，声明 {len(claims)} 份，活动记录 {len(act)} 行 → {out}")


if __name__ == "__main__":
    main()
