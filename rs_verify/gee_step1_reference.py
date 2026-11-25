"""第 1 步：建参考边界（"真实答案"）。分两个阶段运行。

阶段 A（--stage persistence）：
  把 2025 年的 S1 影像叠加成"设施持续度"图（每个像元被判为"亮"的比例，0~1）：
  浮筏/网箱长期在海上 → 持续度高；船只偶尔经过 → 持续度低。
  只存到 GEE 云端资产（只算一次）。
  同时导出一张 Sentinel-2 真彩色图，供人工核对（已导出过可加 --no-s2 跳过）。

阶段 B（--stage vectors，等阶段 A 的资产任务完成后再运行）：
  读取存好的持续度图：
    - 导出一份 GeoTIFF 到 Drive（给你看）；
    - 把持续度 ≥ 阈值的区域勾成多边形（GeoJSON，导出到 Drive）。
  这一步只是"在现成的图上画线"，很快。

人工核对（在 QGIS 里打开 GeoJSON + 真彩色图）：
  - 删掉明显不是养殖设施的多边形（码头、风机、岛礁边缘）；
  - 核对后把文件放到 data/reference/<area>_reference.geojson。

用法：
  python -m rs_verify.gee_step1_reference --area ailun --project <项目ID> --stage persistence --no-s2
  （到 GEE 任务页面等 <area>_persistence_asset 完成）
  python -m rs_verify.gee_step1_reference --area ailun --project <项目ID> --stage vectors
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G

# 参考边界只需要一年的数据；全时段（2019-2025）留给第 2 步的活动曲线
REF_START = "2025-01-01"
REF_END = "2025-12-31"


def asset_id(project: str, area: str) -> str:
    return f"projects/{project}/assets/{area}_s1_persistence"


def persistence_image(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = G.s1_collection(region, start, end).map(G.make_bright_fn(region))
    return col.select("bright").mean().rename("persistence").toFloat()


def structure_polygons(pers: ee.Image, region: ee.Geometry, area_key: str) -> ee.FeatureCollection:
    mask = pers.gte(C.PERSISTENCE_THRESHOLD)
    vectors = mask.selfMask().reduceToVectors(
        geometry=region,
        scale=10,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="label",
        maxPixels=1e10,
        tileScale=4,
    )
    vectors = vectors.map(lambda f: f.set({"area_m2": f.geometry().area(maxError=1), "area_key": area_key}))
    vectors = vectors.filter(ee.Filter.gte("area_m2", C.MIN_STRUCTURE_AREA_M2))
    # 平均持续度：一次性对所有多边形统计（读的是存好的资产，很便宜）
    return pers.reduceRegions(collection=vectors, reducer=ee.Reducer.mean().setOutputs(["mean_persistence"]),
                              scale=10, tileScale=4)


def s2_truecolor(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    return col.median().select(["B4", "B3", "B2"]).divide(10000).clamp(0, 0.3).multiply(255 / 0.3).uint8()


def main() -> None:
    ap = argparse.ArgumentParser(description="第1步：参考边界（分两阶段）")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--project", required=True, help="GEE Cloud 项目 ID")
    ap.add_argument("--stage", required=True, choices=["persistence", "vectors"])
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=REF_START)
    ap.add_argument("--end", default=REF_END)
    ap.add_argument("--no-s2", action="store_true", help="不再导出真彩色图（已导出过时用）")
    ap.add_argument("--s2-start", default="2025-02-01", help="真彩色图时段起（建议选浮筏都在海上的季节）")
    ap.add_argument("--s2-end", default="2025-04-30")
    args = ap.parse_args()

    G.init(args.project)
    info = C.STUDY_AREAS[args.area]
    region = G.region_from_bbox(info["bbox"])
    aid = asset_id(args.project, args.area)
    tasks = []

    if args.stage == "persistence":
        n = G.s1_collection(region, args.start, args.end).size().getInfo()
        print(f"[{info['name']}] {args.start} ~ {args.end} S1 影像数：{n}")
        tasks.append(ee.batch.Export.image.toAsset(
            image=persistence_image(region, args.start, args.end),
            description=f"{args.area}_persistence_asset", assetId=aid,
            region=region, scale=10, maxPixels=1e10,
        ))
        if not args.no_s2:
            tasks.append(ee.batch.Export.image.toDrive(
                image=s2_truecolor(region, args.s2_start, args.s2_end), description=f"{args.area}_s2_truecolor",
                folder=args.drive_folder, region=region, scale=10, maxPixels=1e10,
            ))
    else:
        try:
            pers = ee.Image(aid)
            pers.bandNames().getInfo()
        except ee.EEException:
            raise SystemExit(f"找不到资产 {aid}：请先运行 --stage persistence，并等 {args.area}_persistence_asset 任务完成。")
        tasks.append(ee.batch.Export.image.toDrive(
            image=pers, description=f"{args.area}_s1_persistence", folder=args.drive_folder,
            region=region, scale=10, maxPixels=1e10,
        ))
        tasks.append(ee.batch.Export.table.toDrive(
            collection=structure_polygons(pers, region, args.area),
            description=f"{args.area}_reference_draft", folder=args.drive_folder, fileFormat="GeoJSON",
        ))

    for t in tasks:
        t.start()
        print("已提交导出任务：", t.id)
    print("到 https://code.earthengine.google.com/tasks 查看进度。")


if __name__ == "__main__":
    main()
