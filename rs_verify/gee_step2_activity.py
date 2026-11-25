"""第 2 步：给每个多边形画"心电图"（活动曲线）。

输入：一个 GeoJSON，每个要素必须有属性 pid（多边形编号）和 area_m2。
      通常是 claims.py 生成的 data/claims/<area>_claim_geoms.geojson
      （里面同时包含真实参考边界和平移后的假边界）。
输出：一张 CSV（导出到 Google Drive），每行 = 一个多边形 × 一景影像：
      pid, date, bright_frac（多边形内设施像元占比）, n_pix（有效像元数）,
      orbit_pass, rel_orbit, platform, water_frac

用法：
  python -m rs_verify.gee_step2_activity --area ailun \
      --geojson data/claims/ailun_claim_geoms.geojson --project <你的GEE项目ID>
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G
from .geo import load_geojson

SELECTORS = ["pid", "date", "bright_frac", "n_pix", "orbit_pass", "rel_orbit", "platform", "water_frac", "area_m2"]


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


def activity_table(fc: ee.FeatureCollection, start: str, end: str) -> ee.FeatureCollection:
    # 背景统计范围：所有多边形外包框再往外扩 2 km，保证里面有足够的空海面
    region = fc.geometry().bounds().buffer(2000).bounds()
    col = G.s1_collection(region, start, end).map(G.make_bright_fn(region))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def per_image(img):
        img = ee.Image(img)
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        stats = img.select("bright").reduceRegions(collection=fc, reducer=reducer, scale=10)
        return stats.map(
            lambda f: f.set(
                {
                    "date": date,
                    "bright_frac": f.get("mean"),
                    "n_pix": f.get("count"),
                    "orbit_pass": img.get("orbitProperties_pass"),
                    "rel_orbit": img.get("relativeOrbitNumber_start"),
                    "platform": img.get("platform_number"),
                }
            )
        )

    table = col.map(per_image).flatten()
    # 影像没覆盖到的多边形 count=0、mean=null，去掉
    return table.filter(ee.Filter.gt("n_pix", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description="第2步：活动曲线")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--tag", default="activity", help="导出文件名后缀")
    args = ap.parse_args()

    G.init(args.project)
    fc = add_water_fraction(to_ee_fc(load_geojson(args.geojson)))
    print("多边形数：", fc.size().getInfo())

    task = ee.batch.Export.table.toDrive(
        collection=activity_table(fc, args.start, args.end),
        description=f"{args.area}_{args.tag}",
        folder=args.drive_folder,
        fileFormat="CSV",
        selectors=SELECTORS,
    )
    task.start()
    print("已提交导出任务：", task.id)
    print("完成后把 CSV 放到 data/activity/ 下，再运行 experiment.py。")


if __name__ == "__main__":
    main()
