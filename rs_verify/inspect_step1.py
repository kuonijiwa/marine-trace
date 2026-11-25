"""检查第 1 步结果：参考边界初稿的统计 + 两张快速预览图。

输入：从 Drive 下载的 <area>_reference_draft.geojson（放到 data/reference/ 下）。
输出（outputs/<area>/）：
  step1_summary.txt   统计结果（同时打印在终端）
  step1_polygons.png  多边形分布图（按持续度着色，红框 = 研究区范围）
  step1_overlay.png   卫星真彩色图 + 检测到的设施（红色），需要 GEE；加 --no-gee 跳过

用法：
  python -m rs_verify.inspect_step1 --area ailun --project <项目ID>
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402

EDGE_TOL_DEG = 0.002  # 约 200 m：多边形离研究区框边缘这么近，就算"碰边"


def polygons_of(geom: dict) -> list[list[list[float]]]:
    """返回外环列表（MultiPolygon 拆开）。"""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


def summarize(fc: dict, bbox: list[float]) -> tuple[str, list[dict]]:
    w, s, e, n = bbox
    rows = []
    for f in fc["features"]:
        g = f["geometry"]
        if g is None or g["type"] not in ("Polygon", "MultiPolygon"):
            continue
        p = f.get("properties", {}) or {}
        area = geo.geometry_area_m2(g)
        bw, bs, be, bn = geo.geometry_bounds(g)
        touch = bw - w < EDGE_TOL_DEG or e - be < EDGE_TOL_DEG or bs - s < EDGE_TOL_DEG or n - bn < EDGE_TOL_DEG
        cx, cy = (bw + be) / 2, (bs + bn) / 2
        rows.append({"area_m2": area, "pers": p.get("mean_persistence"), "touch": touch,
                     "cx": cx, "cy": cy, "geom": g})
    if not rows:
        return "没有找到任何多边形（可能阈值太高，或研究区框不在养殖区上）。", rows

    areas = np.array([r["area_m2"] for r in rows])
    pers = np.array([r["pers"] for r in rows if r["pers"] is not None], dtype=float)
    touch = [r for r in rows if r["touch"]]
    big = sorted(rows, key=lambda r: -r["area_m2"])[:10]
    lines = [
        f"多边形数量：{len(rows)}",
        f"总面积：{areas.sum() / 1e6:.2f} km²（{areas.sum() / 1e4:.0f} 公顷）",
        "单块面积（公顷）：最小 {:.2f} / 中位 {:.2f} / 90% 分位 {:.2f} / 最大 {:.1f}".format(
            *(np.percentile(areas, [0, 50, 90, 100]) / 1e4)),
        "面积分档：<1 ha: {}，1–10 ha: {}，10–100 ha: {}，>100 ha: {}".format(
            int((areas < 1e4).sum()), int(((areas >= 1e4) & (areas < 1e5)).sum()),
            int(((areas >= 1e5) & (areas < 1e6)).sum()), int((areas >= 1e6).sum())),
    ]
    if len(pers):
        lines.append("平均持续度：最小 {:.2f} / 中位 {:.2f} / 最大 {:.2f}".format(*np.percentile(pers, [0, 50, 100])))
    lines.append(f"碰到研究区框边缘的多边形：{len(touch)} 块，合计 {sum(r['area_m2'] for r in touch) / 1e4:.0f} 公顷"
                 "（多的话说明养殖区延伸到框外，框可能需要放大）")
    lines.append("最大的 10 块（面积 ha，中心经度，中心纬度，平均持续度）：")
    for r in big:
        pv = f"{r['pers']:.2f}" if r["pers"] is not None else "NA"
        lines.append(f"  {r['area_m2'] / 1e4:8.1f}  {r['cx']:.4f}  {r['cy']:.4f}  {pv}")
    return "\n".join(lines), rows


def plot_polygons(rows: list[dict], bbox: list[float], out: Path, title: str) -> None:
    w, s, e, n = bbox
    verts, vals = [], []
    for r in rows:
        for ring in polygons_of(r["geom"]):
            verts.append(np.array(ring)[:, :2])
            vals.append(r["pers"] if r["pers"] is not None else 0.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    if verts:
        pc = PolyCollection(verts, array=np.array(vals), cmap="viridis", edgecolor="none")
        pc.set_clim(C.PERSISTENCE_THRESHOLD, 1.0)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.7, label="mean persistence")
    ax.plot([w, e, e, w, w], [s, s, n, n, s], "r-", lw=1)
    pad = 0.01
    ax.set_xlim(w - pad, e + pad)
    ax.set_ylim(s - pad, n + pad)
    ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def gee_overlay(area: str, project: str, out: Path) -> None:
    import ee

    from . import gee_common as G
    from .gee_step1_reference import asset_id, s2_truecolor

    G.init(project)
    bbox = C.STUDY_AREAS[area]["bbox"]
    region = G.region_from_bbox(bbox)
    rgb = s2_truecolor(region, "2025-02-01", "2025-04-30").visualize(min=0, max=255)
    pers = ee.Image(asset_id(project, area))
    det = pers.gte(C.PERSISTENCE_THRESHOLD).selfMask().visualize(palette=["ff0000"], opacity=0.6)
    frame = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(region)]), 1, 2).visualize(palette=["ffff00"])
    img = ee.ImageCollection([rgb, det, frame]).mosaic()
    url = img.getThumbURL({"region": region, "dimensions": 1600, "format": "png"})
    urllib.request.urlretrieve(url, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="检查第1步结果")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--geojson", default=None, help="默认 data/reference/<area>_reference_draft.geojson")
    ap.add_argument("--project", default=None, help="GEE 项目 ID（画叠加图用）")
    ap.add_argument("--no-gee", action="store_true", help="不画卫星叠加图")
    args = ap.parse_args()

    gj = Path(args.geojson or Path(C.DATA_DIR) / "reference" / f"{args.area}_reference_draft.geojson")
    if not gj.exists():
        raise SystemExit(f"找不到 {gj}：请先从 Drive 的 marine_trace 文件夹下载 {gj.name} 放到 {gj.parent}/")
    od = Path(C.OUTPUT_DIR) / args.area
    od.mkdir(parents=True, exist_ok=True)

    bbox = C.STUDY_AREAS[args.area]["bbox"]
    text, rows = summarize(geo.load_geojson(gj), bbox)
    print(text)
    (od / "step1_summary.txt").write_text(text, encoding="utf-8")
    plot_polygons(rows, bbox, od / "step1_polygons.png", f"{args.area}: detected structures")
    print("已保存：", od / "step1_polygons.png")

    if not args.no_gee:
        if not args.project:
            raise SystemExit("画卫星叠加图需要 --project <项目ID>（或加 --no-gee 跳过）")
        gee_overlay(args.area, args.project, od / "step1_overlay.png")
        print("已保存：", od / "step1_overlay.png")


if __name__ == "__main__":
    main()
