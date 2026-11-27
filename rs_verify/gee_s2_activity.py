"""光学"第二证人"：每个多边形 × 每景 Sentinel-2 的筏格纹理、晴空比例、雾霾指标（导出到 Drive）。

为什么要光学：误判核查图显示，雷达判"不在养"的养殖地块里，绝大多数在同年光学图上筏格清清楚楚。
光学看得清但怕云雾，雷达全天候但会漏看——两个互补。后面只挑"好图"（晴、无雾）来用。

每景每个多边形输出：
  tex     筏格纹理：可见光亮度 3×3 窗口变异系数，只在晴空、海面像元上取平均（浮筏一条条 → 纹理高）
  clear   晴空比例：海面像元里不是云/云影/卷云/饱和的比例（SCL 分类）
  b2      蓝波段平均反射率（雾霾越重越高，用来挑"好图"）
  n_pix   有效（晴空海面）像元数
输出：Drive/marine_trace/<area>_s2_<year>.csv，列 pid, date, tex, clear, b2, n_pix, area_m2, tile

--inner-buffer M：统计前把每个地块向内收缩 M 米（建议 50 m ≈ 5 个像元），只看地块内部。
  原因：空地块四周往往是别人的筏架，3×3 纹理窗口在地块边缘会"看到"邻居的筏格，空地块被误判"在养"。
  收缩后 area_m2 换成收缩后的面积；输出文件名为 <area>_s2in<M>_<year>.csv，不覆盖原文件。

用法：
  python -m rs_verify.gee_s2_activity --area ailun --geojson data/claims/ailun_claim_geoms.geojson \
      --project <PROJECT> --by-year [--inner-buffer 50]
"""
from __future__ import annotations

import argparse

import ee

from . import config as C
from . import gee_common as G
from . import geo
from .gee_step2_activity import stats_region, to_ee_fc

SELECTORS = ["pid", "date", "tex", "clear", "b2", "n_pix", "area_m2", "tile"]
MAX_SCENE_CLOUD = 40  # 整景云量上限（%），只是粗筛，真正的"好图"在本地按多边形晴空比例和雾霾挑


def shrink(fc: ee.FeatureCollection, inner_m: float) -> ee.FeatureCollection:
    """每个地块向内收缩 inner_m 米，并把 area_m2 换成收缩后的面积（覆盖率按新面积算）。"""
    if not inner_m:
        return fc

    def one(f):
        g = ee.Feature(f).geometry().buffer(-float(inner_m), 1)
        return ee.Feature(f).setGeometry(g).set("area_m2", g.area(1))

    return fc.map(one).filter(ee.Filter.gt("area_m2", 0))


def month_filter(months):
    """只保留这些日历月的影像（如海带季 12,1,2,3,4,5），少算不用的月份、省 GEE 配额。"""
    return ee.Filter.Or(*[ee.Filter.calendarRange(int(m), int(m), "month") for m in months])


def s2_collection(region: ee.Geometry, start: str, end: str, months=None) -> ee.ImageCollection:
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD)))
    return col.filter(month_filter(months)) if months else col


def per_scene_table(fc: ee.FeatureCollection, start: str, end: str, region: ee.Geometry,
                    months=None) -> ee.FeatureCollection:
    water = G.water_mask()
    k = ee.Kernel.square(1)
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def per_image(img):
        img = ee.Image(img)
        scl = img.select("SCL")
        clear = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        vis = img.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean())
        cv = vis.reduceNeighborhood(ee.Reducer.stdDev(), k).divide(
            vis.reduceNeighborhood(ee.Reducer.mean(), k).max(1))
        bands = ee.Image.cat([
            cv.rename("tex").updateMask(clear),
            clear.rename("clear"),
            img.select("B2").divide(10000).rename("b2").updateMask(clear),
        ]).updateMask(water)
        stats = bands.reduceRegions(collection=fc, reducer=reducer, scale=10)
        stats = stats.filter(ee.Filter.gt("tex_count", 0))
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return stats.map(lambda f: f.set({
            "date": date, "tex": f.get("tex_mean"), "clear": f.get("clear_mean"), "b2": f.get("b2_mean"),
            "n_pix": f.get("tex_count"), "tile": img.get("MGRS_TILE"),
        }))

    return s2_collection(region, start, end, months).map(per_image).flatten()


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 光学逐景指标导出")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--by-year", action="store_true")
    ap.add_argument("--inner-buffer", type=float, default=0.0, help="地块向内收缩米数（建议 50）")
    ap.add_argument("--months", default=None, help="只导这些月份，如 12,1,2,3,4,5（海带季）；默认全年")
    args = ap.parse_args()
    months = [int(m) for m in args.months.split(",")] if args.months else None

    G.init(args.project)
    gj = geo.load_geojson(args.geojson)
    fc = shrink(to_ee_fc(gj), args.inner_buffer)
    region = stats_region(gj, pad_m=0)
    tag = f"_s2in{int(args.inner_buffer)}" if args.inner_buffer else "_s2"
    print("多边形数：", fc.size().getInfo())
    if args.by_year:
        y0, y1 = int(args.start[:4]), int(args.end[:4])
        periods = [(f"{y}-01-01", f"{y + 1}-01-01", f"_{y}") for y in range(y0, y1 + 1)]
    else:
        periods = [(args.start, args.end, "")]
    for start, end, suffix in periods:
        n = s2_collection(region, start, end, months).size().getInfo()
        task = ee.batch.Export.table.toDrive(
            collection=per_scene_table(fc, start, end, region, months),
            description=f"{args.area}{tag}{suffix}", folder=args.drive_folder, fileFormat="CSV", selectors=SELECTORS)
        task.start()
        print(f"已提交 {args.area}{tag}{suffix}（{start} ~ {end}，{n} 景）：", task.id)
    print("完成后把 CSV 放到 data/activity_s2/ 下。")


if __name__ == "__main__":
    main()
