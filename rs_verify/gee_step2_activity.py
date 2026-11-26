"""第 2 步：给每个多边形画"心电图"（活动曲线）。

输入：一个 GeoJSON，每个要素必须有属性 pid（多边形编号）和 area_m2。
      通常是 claims.py 生成的 data/claims/<area>_claim_geoms.geojson
      （里面同时包含真实参考边界和平移后的假边界）。
输出：一张 CSV（导出到 Google Drive），每行 = 一个多边形 × 一景影像：
      pid, date,
      lin_ratio（多边形平均雷达亮度 / 这一景海面背景，线性功率；空海面≈1，主指标）,
      bright_frac（多边形内"比背景亮 6 dB"像元占比；适合网箱）, bg_db（背景, dB）,
      n_pix（有效像元数）, orbit_pass, rel_orbit, platform, water_frac, area_m2
  水体掩膜已向海收缩 100 m（离岸太近的像元不计）。

用法：
  python -m rs_verify.gee_step2_activity --area ailun \
      --geojson data/claims/ailun_claim_geoms.geojson --project <你的GEE项目ID>
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G
from . import geo

SELECTORS = ["pid", "date", "lin_ratio", "bright_frac", "bg_db", "n_pix", "orbit_pass", "rel_orbit", "platform",
             "water_frac", "area_m2"]


def to_ee_fc(fc: dict) -> ee.FeatureCollection:
    feats = []
    for f in fc["features"]:
        p = f["properties"]
        feats.append(ee.Feature(ee.Geometry(f["geometry"]), {"pid": str(p["pid"]), "area_m2": float(p["area_m2"])}))
    return ee.FeatureCollection(feats)


def add_water_fraction(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    w = G.water_mask().unmask(0)
    out = w.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
    return out.map(lambda f: f.set("water_frac", f.get("mean")))


def stats_region(gj: dict, pad_m: float = 2000.0) -> ee.Geometry:
    """背景统计范围：所有多边形外包框再往外扩 2 km，保证里面有足够的空海面。
    在本地算外包框（几千个多边形在 GEE 上求并集很慢）。"""
    bounds = [geo.geometry_bounds(f["geometry"]) for f in gj["features"]]
    w, s = min(b[0] for b in bounds), min(b[1] for b in bounds)
    e, n = max(b[2] for b in bounds), max(b[3] for b in bounds)
    dlon, dlat = geo._deg_per_m((s + n) / 2)
    return ee.Geometry.Rectangle([w - pad_m * dlon, s - pad_m * dlat, e + pad_m * dlon, n + pad_m * dlat],
                                 proj="EPSG:4326", geodesic=False)


def pick_orbit(region: ee.Geometry, start: str, end: str) -> tuple[int, dict]:
    """覆盖研究区次数最多的相对轨道。"""
    hist = G.s1_collection(region, start, end).aggregate_histogram("relativeOrbitNumber_start").getInfo()
    best = max(hist, key=lambda k: hist[k])
    return int(float(best)), hist


def activity_table(fc: ee.FeatureCollection, start: str, end: str, region: ee.Geometry,
                   rel_orbit: int | None = None) -> ee.FeatureCollection:
    col = G.s1_collection(region, start, end, rel_orbit).map(G.make_bright_fn(region))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def per_image(img):
        img = ee.Image(img)
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        bg_lin = ee.Number(img.get("bg_lin"))
        # 两个波段 × (mean, count) → 属性名 bright_mean / bright_count / vv_lin_mean / vv_lin_count
        stats = img.select(["bright", "vv_lin"]).reduceRegions(collection=fc, reducer=reducer, scale=10)
        # 影像没覆盖到的多边形 count=0、mean=null，先去掉再做除法
        stats = stats.filter(ee.Filter.gt("vv_lin_count", 0))
        return stats.map(
            lambda f: f.set(
                {
                    "date": date,
                    "bright_frac": f.get("bright_mean"),
                    # 亮度比 = 多边形平均线性功率 / 这一景海面背景；空海面 ≈ 1
                    "lin_ratio": ee.Number(f.get("vv_lin_mean")).divide(bg_lin),
                    "bg_db": img.get("bg_db"),
                    "n_pix": f.get("vv_lin_count"),
                    "orbit_pass": img.get("orbitProperties_pass"),
                    "rel_orbit": img.get("relativeOrbitNumber_start"),
                    "platform": img.get("platform_number"),
                }
            )
        )

    return col.map(per_image).flatten()


def main() -> None:
    ap = argparse.ArgumentParser(description="第2步：活动曲线")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--tag", default="activity", help="导出文件名后缀")
    ap.add_argument("--rel-orbit", default="auto",
                    help="只用一条相对轨道：auto = 自动选覆盖最多的那条；all = 不限；或填轨道号")
    ap.add_argument("--by-year", action="store_true", help="每年提交一个导出任务（数据量大时避免超时）")
    args = ap.parse_args()

    G.init(args.project)
    gj = geo.load_geojson(args.geojson)
    fc = add_water_fraction(to_ee_fc(gj))
    region = stats_region(gj)
    print("多边形数：", fc.size().getInfo())

    if args.rel_orbit == "all":
        orbit = None
    elif args.rel_orbit == "auto":
        orbit, hist = pick_orbit(region, args.start, args.end)
        print("各相对轨道的影像数：", {int(float(k)): v for k, v in hist.items()}, "→ 选用", orbit)
    else:
        orbit = int(args.rel_orbit)

    if args.by_year:
        y0, y1 = int(args.start[:4]), int(args.end[:4])
        periods = [(f"{y}-01-01", f"{y + 1}-01-01", f"_{y}") for y in range(y0, y1 + 1)]
    else:
        periods = [(args.start, args.end, "")]

    for start, end, suffix in periods:
        n = G.s1_collection(region, start, end, orbit).size().getInfo()
        task = ee.batch.Export.table.toDrive(
            collection=activity_table(fc, start, end, region, orbit),
            description=f"{args.area}_{args.tag}{suffix}",
            folder=args.drive_folder,
            fileFormat="CSV",
            selectors=SELECTORS,
        )
        task.start()
        print(f"已提交导出任务 {args.area}_{args.tag}{suffix}（{start} ~ {end}，{n} 景）：", task.id)
    print("完成后把 CSV 放到 data/activity/ 下，再运行 experiment.py（可一次给多个 CSV）。")


if __name__ == "__main__":
    main()
