"""Google Earth Engine 公共函数：Sentinel-1 预处理和"设施像元"判定。

核心思想（通俗说法）：
  浮筏、网箱是金属/塑料浮体，雷达回波比周围平静海面亮很多。
  但大风天整片海面都会变亮，所以不能用固定阈值，
  而是拿"这个像元"和"它周围 1.5 km 的海面背景"比：亮出 6 dB 以上才算设施。

需要在你的电脑上运行：pip install earthengine-api，并先 `earthengine authenticate`。
"""
from __future__ import annotations

import ee

from . import config as C


def init(project: str | None = None) -> None:
    """初始化 GEE。新版 GEE 需要传一个 Cloud 项目 ID。"""
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def region_from_bbox(bbox) -> ee.Geometry:
    w, s, e, n = bbox
    return ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)


def s1_collection(region: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """IW 模式、含 VV 极化的 Sentinel-1 GRD（GEE 里已是 dB 值）。"""
    return (
        ee.ImageCollection(C.S1_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", C.S1_POLARIZATION))
        .select(C.S1_POLARIZATION)
    )


def water_mask() -> ee.Image:
    """ESA WorldCover 2021：类别 80 = 永久水体（含海面）。用来去掉陆地。"""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    return wc.eq(80).rename("water")


def add_bright_band(img: ee.Image) -> ee.Image:
    """给一景 S1 影像加上 'bright' 波段（1=设施像元，0=海面），陆地被掩掉。"""
    band = C.S1_POLARIZATION
    vv = img.select(band)
    # 1) 斑点噪声：小窗口中值滤波
    vv_f = vv.focalMedian(radius=C.SPECKLE_RADIUS_M, kernelType="square", units="meters")
    # 2) 周围海面背景：先降到 60 m 再取邻域 25% 分位数（暗的那部分 = 真正的海面）
    proj = vv.projection()
    coarse = vv_f.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(
        proj.atScale(C.BACKGROUND_SCALE_M)
    )
    bg = coarse.reduceNeighborhood(
        reducer=ee.Reducer.percentile([C.BACKGROUND_PERCENTILE]),
        kernel=ee.Kernel.circle(C.BACKGROUND_RADIUS_M, "meters"),
    ).rename("bg")
    # 3) 比背景亮 delta dB 以上 → 设施
    bright = vv_f.gt(bg.add(C.BRIGHT_DELTA_DB)).rename("bright")
    bright = bright.updateMask(water_mask())
    out = ee.Image.cat([bright, vv_f.rename("vv_f"), bg])
    # copyProperties 返回 ee.Element，要转回 ee.Image
    return ee.Image(
        out.copyProperties(img, ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start", "platform_number"])
    )
