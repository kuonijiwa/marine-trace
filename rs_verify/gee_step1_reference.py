"""第 1 步：建参考边界（"真实答案"）。

两种来源，各分两个阶段运行（先把重计算存成云端资产，再在资产上勾边界）：

【推荐】光学 Sentinel-2（海带浮筏在光学图上清晰可见）：
  --stage s2mask     旺季（默认 2025 年 2–4 月）去云合成 → 计算"浮筏纹理"图 → 存成资产
  --stage s2vectors  纹理 ≥ 阈值 → 闭运算连成整块 → 勾成多边形（GeoJSON 导出到 Drive）
                     阈值可用 --threshold 调，不用重算纹理图

  参考边界来自光学、核验用雷达：两个独立的卫星，不会"自己考自己"。

【备用】雷达 Sentinel-1（适合网箱这类回波很强的大型结构）：
  --stage persistence  2025 年各景"亮像元"的出现频率 → 存成资产
  --stage vectors      持续度 ≥ 阈值的区域勾成多边形

人工核对（在 QGIS 里打开 GeoJSON + 真彩色图）：
  - 删掉明显不是养殖设施的多边形（港口、码头、风机）；
  - 核对后把文件放到 data/reference/<area>_reference.geojson。

用法（光学）：
  python -m rs_verify.gee_step1_reference --area ailun --project <项目ID> --stage s2mask
  （等 <area>_s2_texture_asset 完成）
  python -m rs_verify.gee_step1_reference --area ailun --project <项目ID> --stage s2vectors
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G

# S1 参考边界只需要一年的数据；全时段（2019-2025）留给第 2 步的活动曲线
REF_START = "2025-01-01"
REF_END = "2025-12-31"


def asset_id(project: str, area: str) -> str:
    """S1 持续度资产。"""
    return f"projects/{project}/assets/{area}_s1_persistence"


def s2_asset_id(project: str, area: str) -> str:
    """S2 纹理资产。"""
    return f"projects/{project}/assets/{area}_s2_texture"


def persistence_image(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = G.s1_collection(region, start, end).map(G.make_bright_fn(region))
    return col.select("bright").mean().rename("persistence").toFloat()


def _vectorize(mask: ee.Image, region: ee.Geometry, area_key: str, min_area: float, value_img: ee.Image,
               value_name: str, source: str) -> ee.FeatureCollection:
    vectors = mask.reduceToVectors(
        geometry=region,
        scale=10,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="label",
        maxPixels=1e10,
        tileScale=4,
    )
    vectors = vectors.map(lambda f: f.set({"area_m2": f.geometry().area(maxError=1), "area_key": area_key,
                                           "source": source}))
    vectors = vectors.filter(ee.Filter.gte("area_m2", min_area))
    return value_img.reduceRegions(collection=vectors, reducer=ee.Reducer.mean().setOutputs([value_name]),
                                   scale=10, tileScale=4)


def structure_polygons(pers: ee.Image, region: ee.Geometry, area_key: str) -> ee.FeatureCollection:
    mask = pers.gte(C.PERSISTENCE_THRESHOLD).selfMask()
    return _vectorize(mask, region, area_key, C.MIN_STRUCTURE_AREA_M2, pers, "mean_persistence", "s1")


def raft_polygons(tex: ee.Image, region: ee.Geometry, area_key: str, threshold: float) -> ee.FeatureCollection:
    mask = G.raft_mask_from_texture(tex, threshold)
    return _vectorize(mask, region, area_key, C.RAFT_MIN_AREA_M2, tex, "mean_texture", "s2")


def s2_truecolor(region: ee.Geometry, start: str, end: str) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    return col.median().select(["B4", "B3", "B2"]).divide(10000).clamp(0, 0.3).multiply(255 / 0.3).uint8()


def _load_asset(aid: str, hint: str) -> ee.Image:
    try:
        img = ee.Image(aid)
        img.bandNames().getInfo()
        return img
    except ee.EEException:
        raise SystemExit(f"找不到资产 {aid}：请先运行 {hint}，并等对应的 *_asset 任务完成。")


def main() -> None:
    ap = argparse.ArgumentParser(description="第1步：参考边界")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--project", required=True, help="GEE Cloud 项目 ID")
    ap.add_argument("--stage", required=True, choices=["s2mask", "s2vectors", "persistence", "vectors"])
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--threshold", type=float, default=C.S2_TEXTURE_THRESHOLD, help="s2vectors：纹理阈值")
    ap.add_argument("--start", default=None, help="合成时段起（默认：S2 用旺季，S1 用 2025 全年）")
    ap.add_argument("--end", default=None)
    ap.add_argument("--with-s2-truecolor", action="store_true", help="persistence 阶段顺便导出真彩色图")
    args = ap.parse_args()

    G.init(args.project)
    info = C.STUDY_AREAS[args.area]
    region = G.region_from_bbox(info["bbox"])
    tasks = []

    if args.stage == "s2mask":
        start, end = args.start or C.S2_REF_START, args.end or C.S2_REF_END
        comp = G.s2_composite(region, start, end)
        print(f"[{info['name']}] {start} ~ {end} 可用 S2 影像数：{comp.get('n_images').getInfo()}")
        tasks.append(ee.batch.Export.image.toAsset(
            image=G.s2_texture(comp), description=f"{args.area}_s2_texture_asset",
            assetId=s2_asset_id(args.project, args.area), region=region, scale=10, maxPixels=1e10,
        ))
        # 研究区框已放大，真彩色图重新导出一份（文件名带 _v2）
        tasks.append(ee.batch.Export.image.toDrive(
            image=comp.select(["B4", "B3", "B2"]).divide(10000).clamp(0, 0.3).multiply(255 / 0.3).uint8(),
            description=f"{args.area}_s2_truecolor_v2", folder=args.drive_folder,
            region=region, scale=10, maxPixels=1e10,
        ))
    elif args.stage == "s2vectors":
        tex = _load_asset(s2_asset_id(args.project, args.area), "--stage s2mask")
        tag = f"t{int(round(args.threshold * 1000)):03d}"
        tasks.append(ee.batch.Export.table.toDrive(
            collection=raft_polygons(tex, region, args.area, args.threshold),
            description=f"{args.area}_s2_reference_draft_{tag}", folder=args.drive_folder, fileFormat="GeoJSON",
        ))
        print(f"纹理阈值 = {args.threshold}（文件名后缀 {tag}）")
    elif args.stage == "persistence":
        start, end = args.start or REF_START, args.end or REF_END
        n = G.s1_collection(region, start, end).size().getInfo()
        print(f"[{info['name']}] {start} ~ {end} S1 影像数：{n}")
        tasks.append(ee.batch.Export.image.toAsset(
            image=persistence_image(region, start, end),
            description=f"{args.area}_persistence_asset", assetId=asset_id(args.project, args.area),
            region=region, scale=10, maxPixels=1e10,
        ))
        if args.with_s2_truecolor:
            tasks.append(ee.batch.Export.image.toDrive(
                image=s2_truecolor(region, C.S2_REF_START, C.S2_REF_END), description=f"{args.area}_s2_truecolor",
                folder=args.drive_folder, region=region, scale=10, maxPixels=1e10,
            ))
    else:  # vectors
        pers = _load_asset(asset_id(args.project, args.area), "--stage persistence")
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
