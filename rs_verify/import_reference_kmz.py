"""把已发表的全国海水养殖区数据集（kmz）裁剪到研究区，转成参考边界 GeoJSON。

数据来源：Fu Y. et al. (2021) A new satellite-derived dataset for marine aquaculture areas in
China's coastal region. Earth Syst. Sci. Data 13, 1829–1842. https://doi.org/10.5194/essd-13-1829-2021
数据下载：https://zenodo.org/records/3881612 （CC BY 4.0，16 m，GF-1 2016–2019）

kmz 里可能是两种东西，脚本都能处理：
  1) 矢量多边形（Placemark/Polygon）：直接挑出落在研究区框里的多边形；
  2) 栅格图片（GroundOverlay，透明 PNG 等）：把落在框里的图片拼起来，
     有颜色（不透明）的像素当作养殖区，再勾成多边形。

输出：
  data/reference/<area>_fu2021_reference.geojson   参考边界（属性：pid, area_m2, source）
  outputs/<area>/fu2021_import_report.txt           kmz 结构和处理过程报告（排错用）
加 --copy-to <文件夹> 可把这两个文件再复制一份过去。

用法：
  python -m rs_verify.import_reference_kmz --area ailun --kmz "G:\\c_file\\Marine aquaculture areas in the China's coastal region.kmz"
"""
from __future__ import annotations

import argparse
import io
import posixpath
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from . import config as C
from . import geo

SOURCE_TAG = "Fu2021_ESSD"
MIN_AREA_M2 = 5000  # 数据集是 16 m，小于 0.5 公顷的斑块不要


def _local(tag: str) -> str:
    """去掉 XML 命名空间：'{http://...}Placemark' → 'Placemark'。"""
    return tag.rsplit("}", 1)[-1]


def _find(el, name):
    for c in el.iter():
        if _local(c.tag) == name:
            return c
    return None


def _parse_coords(text: str) -> list[list[float]]:
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _bbox_overlap(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


# ---------------------------------------------------------------------------
# 读 kmz
# ---------------------------------------------------------------------------
def scan_kmz(path: Path, bbox: list[float], log: list[str]):
    """返回 (矢量多边形列表, 与框相交的 GroundOverlay 列表, zip 对象)。"""
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    log.append(f"kmz 内文件数：{len(names)}")
    exts = Counter(Path(n).suffix.lower() for n in names)
    log.append("文件类型统计：" + "，".join(f"{k or '(无扩展名)'}: {v}" for k, v in exts.most_common()))

    polys, overlays = [], []
    tag_counts = Counter()
    for name in names:
        if not name.lower().endswith(".kml"):
            continue
        base = posixpath.dirname(name)
        with zf.open(name) as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                t = _local(el.tag)
                if t in ("Placemark", "GroundOverlay", "NetworkLink", "Polygon", "Point", "LineString"):
                    tag_counts[t] += 1
                if t == "Polygon":
                    ob_el = _find(el, "outerBoundaryIs")
                    outer = _find(ob_el if ob_el is not None else el, "coordinates")
                    if outer is None:
                        el.clear()
                        continue
                    ring = _parse_coords(outer.text)
                    holes = []
                    for ib in el.iter():
                        if _local(ib.tag) == "innerBoundaryIs":
                            c = _find(ib, "coordinates")
                            if c is not None:
                                holes.append(_parse_coords(c.text))
                    if len(ring) >= 4:
                        g = {"type": "Polygon", "coordinates": [ring] + [h for h in holes if len(h) >= 4]}
                        if _bbox_overlap(geo.geometry_bounds(g), bbox):
                            polys.append(g)
                    el.clear()
                elif t == "GroundOverlay":
                    box = _find(el, "LatLonBox")
                    href = _find(el, "href")
                    if box is not None and href is not None:
                        vals = {_local(c.tag): c.text for c in box}
                        try:
                            n, s = float(vals["north"]), float(vals["south"])
                            e, w = float(vals["east"]), float(vals["west"])
                        except (KeyError, TypeError, ValueError):
                            el.clear()
                            continue
                        rot = float(vals.get("rotation") or 0)
                        ob = [w, s, e, n]
                        if _bbox_overlap(ob, bbox):
                            img = posixpath.normpath(posixpath.join(base, href.text.strip())) if base else href.text.strip()
                            overlays.append({"href": img, "box": ob, "rotation": rot, "kml": name})
                    el.clear()
    log.append("KML 元素统计：" + "，".join(f"{k}: {v}" for k, v in tag_counts.most_common()))
    log.append(f"与研究区框相交：矢量多边形 {len(polys)} 个，栅格图片 {len(overlays)} 张")
    return polys, overlays, zf


# ---------------------------------------------------------------------------
# 栅格 → 多边形
# ---------------------------------------------------------------------------
def _load_image(zf: zipfile.ZipFile, href: str):
    from PIL import Image

    names = set(zf.namelist())
    key = href if href in names else next((n for n in names if n.endswith(href.lstrip("./"))), None)
    if key is None:
        return None
    return Image.open(io.BytesIO(zf.read(key))).convert("RGBA")


def _foreground(rgba: np.ndarray) -> np.ndarray:
    """有颜色的像素 = 养殖区。优先用透明度；整张不透明时，把最多的那种颜色当背景。"""
    alpha = rgba[..., 3]
    if (alpha < 255).any():
        return alpha > 0
    rgb = rgba[..., :3].reshape(-1, 3)
    vals, counts = np.unique(rgb, axis=0, return_counts=True)
    bg = vals[np.argmax(counts)]
    return ~np.all(rgba[..., :3] == bg, axis=-1)


def rasters_to_polygons(zf, overlays: list[dict], bbox: list[float], log: list[str]) -> list[dict]:
    from contourpy import FillType, contour_generator

    if any(abs(o["rotation"]) > 1e-6 for o in overlays):
        log.append("警告：有带旋转的图片，已跳过（这种数据集里很少见）")
        overlays = [o for o in overlays if abs(o["rotation"]) <= 1e-6]
    loaded = []
    colors = Counter()
    for o in overlays:
        im = _load_image(zf, o["href"])
        if im is None:
            log.append(f"找不到图片：{o['href']}")
            continue
        arr = np.asarray(im)
        res = (o["box"][2] - o["box"][0]) / arr.shape[1]
        loaded.append((o, arr, res))
        fg = _foreground(arr)
        if fg.any():
            for c, k in Counter(map(tuple, arr[fg][:, :3][:: max(1, fg.sum() // 2000)])).most_common(3):
                colors[c] += k
    if not loaded:
        return []
    # 多级金字塔时只用最细的一级
    finest = min(r for _, _, r in loaded)
    loaded = [x for x in loaded if x[2] <= finest * 1.1]
    log.append(f"使用最细一级的图片 {len(loaded)} 张，像元大小 ≈ {finest:.6f}°（约 {finest * 111000:.0f} m）")
    log.append("养殖区像素最常见颜色（RGB）：" + "，".join(str(tuple(int(v) for v in c)) for c, _ in colors.most_common(3)))

    w, s, e, n = bbox
    W = int(np.ceil((e - w) / finest))
    H = int(np.ceil((n - s) / finest))
    canvas = np.zeros((H + 2, W + 2), dtype=np.uint8)  # 四周留一圈 0，保证多边形闭合
    for o, arr, res in loaded:
        ow, os_, oe, on = o["box"]
        fg = _foreground(arr)
        dy = (on - os_) / arr.shape[0]
        # 目标画布每个像元中心 → 源图片行列（最近邻）
        cols = np.arange(W)
        rows = np.arange(H)
        lon = w + (cols + 0.5) * finest
        lat = n - (rows + 0.5) * finest
        ci = np.floor((lon - ow) / res).astype(int)
        ri = np.floor((on - lat) / dy).astype(int)
        cm = (ci >= 0) & (ci < arr.shape[1])
        rm = (ri >= 0) & (ri < arr.shape[0])
        if not cm.any() or not rm.any():
            continue
        sub = fg[np.ix_(ri[rm], ci[cm])]
        r0, c0 = np.where(rm)[0], np.where(cm)[0]
        canvas[1 + r0[0]: 1 + r0[-1] + 1, 1 + c0[0]: 1 + c0[-1] + 1] |= sub.astype(np.uint8)
    log.append(f"拼接画布 {W}×{H} 像元，养殖区像元 {int(canvas.sum())} 个")

    cg = contour_generator(z=canvas.astype(float), fill_type=FillType.OuterOffset)
    pts_list, offs_list = cg.filled(0.5, 1.5)
    polys = []
    for pts, offs in zip(pts_list, offs_list):
        rings = []
        for a, b in zip(offs[:-1], offs[1:]):
            xy = pts[a:b]
            # 画布索引 → 经纬度（画布多了一圈边，所以减 1）
            lon = w + (xy[:, 0] - 1 + 0.5) * finest
            lat = n - (xy[:, 1] - 1 + 0.5) * finest
            ring = np.column_stack([lon, lat]).round(6).tolist()
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            if len(ring) >= 4:
                rings.append(ring)
        if rings:
            polys.append({"type": "Polygon", "coordinates": rings})
    return polys


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="已发表数据集（kmz）→ 研究区参考边界")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--kmz", required=True)
    ap.add_argument("--min-area", type=float, default=MIN_AREA_M2)
    ap.add_argument("--copy-to", default=None, help="把结果再复制到这个文件夹（如 Google Drive 同步文件夹）")
    args = ap.parse_args()

    bbox = C.STUDY_AREAS[args.area]["bbox"]
    log = [f"研究区 {args.area} 框：{bbox}", f"输入：{args.kmz}"]
    polys, overlays, zf = scan_kmz(Path(args.kmz), bbox, log)
    if not polys and overlays:
        polys = rasters_to_polygons(zf, overlays, bbox, log)

    feats = []
    for g in polys:
        a = geo.geometry_area_m2(g)
        cx, cy = geo.centroid(g["coordinates"][0])
        if a < args.min_area or not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
            continue
        feats.append({"type": "Feature", "geometry": g,
                      "properties": {"pid": f"F{len(feats) + 1:04d}", "area_m2": round(a, 1), "source": SOURCE_TAG}})
    areas = np.array([f["properties"]["area_m2"] for f in feats]) if feats else np.array([0.0])
    log.append(f"保留多边形 {len(feats)} 个（≥ {args.min_area / 1e4:.1f} 公顷且中心在框内），"
               f"总面积 {areas.sum() / 1e6:.2f} km²，最大单块 {areas.max() / 1e4:.1f} 公顷")
    if not feats:
        log.append("没有得到任何多边形：请把这份报告传到 Drive，我来看 kmz 的结构再改。")

    out_gj = Path(C.DATA_DIR) / "reference" / f"{args.area}_fu2021_reference.geojson"
    geo.save_geojson(geo.feature_collection(feats), out_gj)
    od = Path(C.OUTPUT_DIR) / args.area
    od.mkdir(parents=True, exist_ok=True)
    rep = od / "fu2021_import_report.txt"
    rep.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print("已保存：", out_gj)
    print("已保存：", rep)
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in (out_gj, rep):
            shutil.copy2(p, dst / p.name)
        print("已复制到：", dst)


if __name__ == "__main__":
    main()
