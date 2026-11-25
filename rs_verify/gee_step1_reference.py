"""第 1 步：建参考边界（"真实答案"）。

做三件事，结果都导出到你的 Google Drive：
  1. 设施"持续度"栅格：每个像元在所有 S1 影像里被判为"亮"的比例（0~1）。
     浮筏/网箱长期在海上 → 持续度高；船只偶尔经过 → 持续度低。
  2. 把持续度 ≥ 阈值的区域自动矢量化成多边形（GeoJSON）。这就是参考边界的初稿。
  3. 一张 Sentinel-2 真彩色合成图，用来人工核对第 2 项的多边形对不对。

人工核对（在 QGIS 里打开 GeoJSON + 真彩色图）：
  - 删掉明显不是养殖设施的多边形（码头、风机、岛礁边缘）；
  - 核对后把文件放到 data/reference/<area>_reference.geojson。

用法：
  python -m rs_verify.gee_step1_reference --area ailun --project <你的GEE项目ID>
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G


def persistence_image(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = G.s1_collection(region, start, end).map(G.add_bright_band)
    n = col.size()
    pers = col.select("bright").mean().rename("persistence")
    return pers.set("n_scenes", n)


def structure_polygons(pers: ee.Image, region: ee.Geometry, area_key: str) -> ee.FeatureCollection:
    mask = pers.gte(C.PERSISTENCE_THRESHOLD)
    vectors = mask.selfMask().reduceToVectors(
        geometry=region,
        scale=10,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="label",
        maxPixels=1e10,
        bestEffort=False,
    )

    def add_props(f):
        geom = f.geometry()
        area = geom.area(maxError=1)
        mean_pers = pers.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("persistence")
        return f.set({"area_m2": area, "mean_persistence": mean_pers, "area_key": area_key})

    return vectors.map(add_props).filter(ee.Filter.gte("area_m2", C.MIN_STRUCTURE_AREA_M2))


def s2_truecolor(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    return col.median().select(["B4", "B3", "B2"]).divide(10000).clamp(0, 0.3).multiply(255 / 0.3).uint8()


def main() -> None:
    ap = argparse.ArgumentParser(description="第1步：参考边界")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--project", default=None, help="GEE Cloud 项目 ID")
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--s2-start", default="2025-02-01", help="真彩色图时段起（建议选浮筏都在海上的季节）")
    ap.add_argument("--s2-end", default="2025-04-30")
    args = ap.parse_args()

    G.init(args.project)
    info = C.STUDY_AREAS[args.area]
    region = G.region_from_bbox(info["bbox"])

    pers = persistence_image(region, args.start, args.end)
    print(f"[{info['name']}] S1 影像数：", pers.get("n_scenes").getInfo())

    tasks = [
        ee.batch.Export.image.toDrive(
            image=pers.toFloat(),
            description=f"{args.area}_s1_persistence",
            folder=args.drive_folder,
            region=region,
            scale=10,
            maxPixels=1e10,
        ),
        ee.batch.Export.table.toDrive(
            collection=structure_polygons(pers, region, args.area),
            description=f"{args.area}_reference_draft",
            folder=args.drive_folder,
            fileFormat="GeoJSON",
        ),
        ee.batch.Export.image.toDrive(
            image=s2_truecolor(region, args.s2_start, args.s2_end),
            description=f"{args.area}_s2_truecolor",
            folder=args.drive_folder,
            region=region,
            scale=10,
            maxPixels=1e10,
        ),
    ]
    for t in tasks:
        t.start()
        print("已提交导出任务：", t.id)
    print("到 https://code.earthengine.google.com/tasks 查看进度；完成后文件在 Drive 的", args.drive_folder, "文件夹。")


if __name__ == "__main__":
    main()
