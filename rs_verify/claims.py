"""生成"产地声明"：真实声明 + 三类人为构造的虚假声明。

一份声明 = "这批产品出自多边形 X，养殖期是 A 到 B，数量是 Q"。

  true           真实：参考多边形 + 正常养殖季 + 合理数量
  fake_location  位置冒用：把多边形平移 d 米（空海面或别处）
  fake_time      时段错报：养殖季整体平移 k 个月
  fake_quantity  产量冒用：数量放大 r 倍（把外面的货洗成牧场货）

输出：
  data/claims/<area>_claims.csv          每行一份声明
  data/claims/<area>_claim_geoms.geojson 需要在 GEE 里算活动曲线的所有多边形

用法：
  python -m rs_verify.claims --area ailun --reference data/reference/ailun_reference.geojson
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import geo


def season_windows(product: str, start_year: int, end_year: int) -> list[tuple[str, str]]:
    """按产品日历生成每一季的 (起始日, 结束日)。跨年的季按"结束年"计。"""
    m0, m1 = C.PRODUCT_CALENDAR[product]
    out = []
    for y_end in range(start_year, end_year + 1):
        y_start = y_end - 1 if m0 > m1 else y_end
        last_day = calendar.monthrange(y_end, m1)[1]
        out.append((f"{y_start}-{m0:02d}-01", f"{y_end}-{m1:02d}-{last_day:02d}"))
    return out


def shift_window(start: str, end: str, months: int) -> tuple[str, str]:
    s = (pd.Timestamp(start) + pd.DateOffset(months=months)).replace(day=1)
    e = pd.Timestamp(end) + pd.DateOffset(months=months)
    e = e.replace(day=calendar.monthrange(e.year, e.month)[1])
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


def build_claims(
    ref_fc: dict,
    area_key: str,
    max_refs: int | None = None,
    n_bearings: int = 4,
    seed: int = C.RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    farm_type = C.STUDY_AREAS[area_key]["farm_type"]
    start_year = int(C.START_DATE[:4]) + 1  # 第一年做"背景"，从第二年开始出声明
    end_year = int(C.END_DATE[:4])

    feats = ref_fc["features"]
    if max_refs and len(feats) > max_refs:
        idx = rng.choice(len(feats), size=max_refs, replace=False)
        feats = [feats[i] for i in sorted(idx)]

    ref_geoms = [f["geometry"] for f in feats]
    geom_feats: list[dict] = []
    rows: list[dict] = []
    cid = 0

    def add_claim(**kw):
        nonlocal cid
        cid += 1
        rows.append({"claim_id": f"C{cid:06d}", "ranch": area_key, **kw})

    for i, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        product = props.get("product") or C.DEFAULT_PRODUCT[farm_type]
        pid = f"R{i + 1:04d}"
        area = geo.geometry_area_m2(f["geometry"])
        geom_feats.append(
            {"type": "Feature", "geometry": f["geometry"], "properties": {"pid": pid, "area_m2": area, "kind": "reference"}}
        )
        seasons = season_windows(product, start_year, end_year)

        # 1) 真实声明 + 3) 产量冒用
        for s, e in seasons:
            q = float(rng.uniform(0.5, 0.95))  # 以"该多边形满产产能"为单位
            add_claim(pid=pid, ref_pid=pid, product=product, start=s, end=e, label="true",
                      quantity_rel=q, shift_m=0, bearing=np.nan, time_shift_months=0, inflation=1.0, overlap_ref=1.0)
            for r in C.QUANTITY_INFLATION:
                add_claim(pid=pid, ref_pid=pid, product=product, start=s, end=e, label="fake_quantity",
                          quantity_rel=q * r, shift_m=0, bearing=np.nan, time_shift_months=0, inflation=r, overlap_ref=1.0)

        # 2) 时段错报
        for k in C.TIME_SHIFT_MONTHS:
            for s, e in seasons:
                s2, e2 = shift_window(s, e, k)
                if pd.Timestamp(e2) > pd.Timestamp(C.END_DATE):
                    continue
                add_claim(pid=pid, ref_pid=pid, product=product, start=s2, end=e2, label="fake_time",
                          quantity_rel=float(rng.uniform(0.5, 0.95)), shift_m=0, bearing=np.nan,
                          time_shift_months=k, inflation=1.0, overlap_ref=1.0)

        # 4) 位置冒用
        base_bearing = float(rng.uniform(0, 360))
        for d in C.SHIFT_DISTANCES_M:
            for b in range(n_bearings):
                bearing = (base_bearing + b * 360.0 / n_bearings) % 360
                g2 = geo.shift_geometry(f["geometry"], d, bearing)
                spid = f"{pid}_S{d}_B{int(round(bearing)):03d}"
                ov = geo.overlap_fraction(g2, ref_geoms)
                geom_feats.append(
                    {"type": "Feature", "geometry": g2,
                     "properties": {"pid": spid, "area_m2": area, "kind": "shifted", "ref_pid": pid,
                                    "shift_m": d, "bearing": bearing, "overlap_ref": ov}}
                )
                for s, e in seasons:
                    add_claim(pid=spid, ref_pid=pid, product=product, start=s, end=e, label="fake_location",
                              quantity_rel=float(rng.uniform(0.5, 0.95)), shift_m=d, bearing=bearing,
                              time_shift_months=0, inflation=1.0, overlap_ref=ov)

    return pd.DataFrame(rows), geo.feature_collection(geom_feats)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成真实/虚假产地声明")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--reference", required=True, help="人工核对后的参考边界 GeoJSON")
    ap.add_argument("--max-refs", type=int, default=None, help="参考多边形太多时随机抽样")
    ap.add_argument("--n-bearings", type=int, default=4)
    ap.add_argument("--outdir", default=str(Path(C.DATA_DIR) / "claims"))
    args = ap.parse_args()

    claims, geoms = build_claims(geo.load_geojson(args.reference), args.area, args.max_refs, args.n_bearings)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    claims.to_csv(out / f"{args.area}_claims.csv", index=False)
    geo.save_geojson(geoms, out / f"{args.area}_claim_geoms.geojson")
    print(claims["label"].value_counts().to_string())
    print("多边形数（送 GEE）：", len(geoms["features"]))


if __name__ == "__main__":
    main()
