"""Google Earth Engine 公共函数：Sentinel-1 预处理和"设施像元"判定。

核心思想（通俗说法）：
  浮筏、网箱是金属/塑料浮体，雷达回波比周围平静海面亮很多。
  但大风天整片海面都会变亮，所以不能用固定阈值，
  而是拿"这个像元"和"这一景影像里研究区海面的整体背景"比：亮出 6 dB 以上才算设施。

  背景 = 这一景里研究区海面像元亮度的 25% 分位数（每景只算一个数，计算量很小）。
  用 25% 分位而不是中位数：大片筏区会把中位数抬高，25% 分位更接近真正的空海面。

需要在你的电脑上运行：pip install earthengine-api，并先 `earthengine authenticate`。
"""
from __future__ import annotations

import ee

from . import config as C

NO_BACKGROUND_DB = 99.0  # 某景与研究区没有有效重叠时，背景设为极大值 → 该景不产生"亮"像元


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


def water_mask(shore_buffer_m: float = C.SHORE_BUFFER_M) -> ee.Image:
    """ESA WorldCover 2021：类别 80 = 永久水体（含海面）。

    再向海里收缩 shore_buffer_m 米：离岸太近的像元（礁石、海堤、码头边缘）一律不算，
    避免它们的强回波被误判成养殖设施。
    """
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    water = wc.eq(80)
    if shore_buffer_m and shore_buffer_m > 0:
        water = water.focalMin(radius=shore_buffer_m, kernelType="circle", units="meters")
    return water.rename("water")


def make_bright_fn(region: ee.Geometry):
    """返回一个函数：给一景 S1 影像加上 'bright' 波段（1=设施像元，0=海面），陆地被掩掉。

    region：用来估计海面背景的范围（研究区框，或全部多边形的外包框）。
    """
    band = C.S1_POLARIZATION
    water = water_mask()

    def add_bright_band(img):
        img = ee.Image(img)
        vv = img.select(band)
        # 1) 斑点噪声：小窗口中值滤波
        vv_f = vv.focalMedian(radius=C.SPECKLE_RADIUS_M, kernelType="square", units="meters")
        # 2) 这一景的海面背景：研究区内海面像元的 25% 分位数（一个数）
        stats = vv_f.updateMask(water).reduceRegion(
            reducer=ee.Reducer.percentile([C.BACKGROUND_PERCENTILE]),
            geometry=region,
            scale=C.BACKGROUND_SCALE_M,
            bestEffort=True,
            maxPixels=1e8,
        )
        raw = stats.values().get(0)
        bg = ee.Number(ee.Algorithms.If(raw, raw, NO_BACKGROUND_DB))
        # 3) 比背景亮 delta dB 以上 → 设施
        bright = vv_f.gt(bg.add(C.BRIGHT_DELTA_DB)).rename("bright").updateMask(water)
        # 4) 线性功率（第 2 步按多边形取平均，再除以背景，得到"亮度比"）
        vv_lin = ee.Image(10).pow(vv_f.divide(10)).rename("vv_lin").updateMask(water)
        bg_lin = ee.Number(10).pow(bg.divide(10))
        out = ee.Image.cat([bright, vv_lin, vv_f.rename("vv_f")]).set({"bg_db": bg, "bg_lin": bg_lin})
        # copyProperties 返回 ee.Element，要转回 ee.Image
        return ee.Image(
            out.copyProperties(img, ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start", "platform_number"])
        )

    return add_bright_band


# ---------------------------------------------------------------------------
# Sentinel-2 光学：浮筏纹理
# ---------------------------------------------------------------------------
def _s2_cloud_mask(img: ee.Image) -> ee.Image:
    """用 SCL 场景分类去掉云、云影、卷云。"""
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(ok)


def s2_composite(region: ee.Geometry, start: str, end: str, max_cloud: float = C.S2_MAX_CLOUD) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(_s2_cloud_mask)
    )
    return col.median().set("n_images", col.size())


def s2_texture(comp: ee.Image) -> ee.Image:
    """浮筏纹理：可见光亮度在 3x3 窗口内的变异系数，再在 50 m 内平滑。只在（收缩后的）海面上计算。

    通俗说：平静海面颜色均匀 → 纹理低；浮筏是一条条深色细线和海水相间 → 纹理高。
    除以均值是为了不受海水浑浊程度（整体明暗）的影响。
    """
    vis = comp.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean()).rename("vis")
    k = ee.Kernel.square(1)  # 3x3（10 m 像元 → 30 m 窗口）
    std = vis.reduceNeighborhood(ee.Reducer.stdDev(), k)
    mean = vis.reduceNeighborhood(ee.Reducer.mean(), k)
    cv = std.divide(mean.max(1)).focalMean(radius=50, kernelType="circle", units="meters")
    return cv.updateMask(water_mask()).rename("texture").toFloat()


def raft_mask_from_texture(tex: ee.Image, threshold: float, close_radius_m: float = C.S2_CLOSE_RADIUS_M) -> ee.Image:
    """纹理 ≥ 阈值 → 浮筏；再做闭运算（先膨胀后腐蚀），把同一片筏区的一条条线连成整块。"""
    raw = tex.gte(threshold).unmask(0)
    closed = raw.focalMax(radius=close_radius_m, kernelType="circle", units="meters").focalMin(
        radius=close_radius_m, kernelType="circle", units="meters"
    )
    return closed.updateMask(water_mask(0)).selfMask().rename("raft")
