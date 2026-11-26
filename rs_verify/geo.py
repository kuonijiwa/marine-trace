"""不依赖 GIS 库的几何小工具（纯 Python + numpy）。

用途：生成"位置冒用"的假声明时，需要把多边形平移若干米。
研究区范围只有几十公里，用局部等距近似就足够精确（误差远小于 10 m 像元）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

EARTH_RADIUS_M = 6_371_008.8


def _deg_per_m(lat_deg: float) -> tuple[float, float]:
    """在纬度 lat 处，1 米对应的经度、纬度度数。"""
    dlat = 1.0 / (EARTH_RADIUS_M * math.pi / 180.0)
    dlon = dlat / max(math.cos(math.radians(lat_deg)), 1e-12)
    return dlon, dlat


def centroid(ring: list[list[float]]) -> tuple[float, float]:
    """多边形外环的顶点平均值（研究区多边形很小，够用）。"""
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lon, lat


def shift_ring(ring: list[list[float]], distance_m: float, bearing_deg: float) -> list[list[float]]:
    """把一个多边形环整体平移 distance_m 米，方向 bearing（0=正北，90=正东）。"""
    _, lat0 = centroid(ring)
    dlon_per_m, dlat_per_m = _deg_per_m(lat0)
    b = math.radians(bearing_deg)
    dx = distance_m * math.sin(b)  # 向东
    dy = distance_m * math.cos(b)  # 向北
    return [[p[0] + dx * dlon_per_m, p[1] + dy * dlat_per_m] for p in ring]


def shift_geometry(geom: dict, distance_m: float, bearing_deg: float) -> dict:
    """平移 GeoJSON Polygon / MultiPolygon。"""
    t = geom["type"]
    if t == "Polygon":
        return {"type": t, "coordinates": [shift_ring(r, distance_m, bearing_deg) for r in geom["coordinates"]]}
    if t == "MultiPolygon":
        return {
            "type": t,
            "coordinates": [[shift_ring(r, distance_m, bearing_deg) for r in poly] for poly in geom["coordinates"]],
        }
    raise ValueError(f"不支持的几何类型: {t}")


def ring_area_m2(ring: list[list[float]]) -> float:
    """局部等距投影下的鞋带公式面积（平方米）。"""
    _, lat0 = centroid(ring)
    dlon_per_m, dlat_per_m = _deg_per_m(lat0)
    xs = [p[0] / dlon_per_m for p in ring]
    ys = [p[1] / dlat_per_m for p in ring]
    s = 0.0
    for i in range(len(ring) - 1):
        s += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(s) / 2.0


def geometry_area_m2(geom: dict) -> float:
    """Polygon / MultiPolygon 面积（外环减内环）。"""
    def poly_area(rings):
        return ring_area_m2(rings[0]) - sum(ring_area_m2(r) for r in rings[1:])

    if geom["type"] == "Polygon":
        return poly_area(geom["coordinates"])
    if geom["type"] == "MultiPolygon":
        return sum(poly_area(p) for p in geom["coordinates"])
    raise ValueError(f"不支持的几何类型: {geom['type']}")


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    """射线法判断点是否在环内。"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_in_geometry(x: float, y: float, geom: dict) -> bool:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for rings in polys:
        if _point_in_ring(x, y, rings[0]) and not any(_point_in_ring(x, y, r) for r in rings[1:]):
            return True
    return False


def geometry_bounds(geom: dict) -> tuple[float, float, float, float]:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    xs = [p[0] for rings in polys for p in rings[0]]
    ys = [p[1] for rings in polys for p in rings[0]]
    return min(xs), min(ys), max(xs), max(ys)


def overlap_fraction(geom: dict, others: list[dict], n_grid: int = 30) -> float:
    """geom 的面积里，有多大比例落在 others（任一）之内。用网格点抽样近似。

    用途：位置冒用的假围栏可能恰好挪到了"别人的养殖区"上，
    这种情况卫星本来就分不出来，实验里要单独统计。
    """
    w, s, e, n = geometry_bounds(geom)
    boxes = [(o, geometry_bounds(o)) for o in others]
    total = hit = 0
    for i in range(n_grid):
        for j in range(n_grid):
            x = w + (i + 0.5) * (e - w) / n_grid
            y = s + (j + 0.5) * (n - s) / n_grid
            if not point_in_geometry(x, y, geom):
                continue
            total += 1
            for o, (ow, os_, oe, on) in boxes:
                if ow <= x <= oe and os_ <= y <= on and point_in_geometry(x, y, o):
                    hit += 1
                    break
    return hit / total if total else 0.0


class RasterMask:
    """把一组多边形栅格化成规则网格上的 True/False（默认 20 m），用来快速算"某块地有多大比例是养殖区"。

    已发表数据集里有面积几十平方公里、几千个顶点的大多边形，逐点判断太慢；
    先栅格化一次，之后任何查询都只是数格子。
    """

    def __init__(self, fc: dict, bbox: Iterable[float], res_m: float = 20.0):
        from matplotlib.path import Path as MPath
        import numpy as np

        self.w, self.s, self.e, self.n = bbox
        lat0 = (self.s + self.n) / 2
        dlon, dlat = _deg_per_m(lat0)
        self.res_m = res_m
        self.dx, self.dy = res_m * dlon, res_m * dlat
        self.nx = int(math.ceil((self.e - self.w) / self.dx))
        self.ny = int(math.ceil((self.n - self.s) / self.dy))
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)  # 行 0 = 最南
        for f in fc["features"]:
            g = f["geometry"]
            if g is None:
                continue
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
            for rings in polys:
                pw, ps, pe, pn = geometry_bounds({"type": "Polygon", "coordinates": rings})
                i0, i1 = self._col(pw), self._col(pe) + 1
                j0, j1 = self._row(ps), self._row(pn) + 1
                i0, j0 = max(i0, 0), max(j0, 0)
                i1, j1 = min(i1, self.nx), min(j1, self.ny)
                if i0 >= i1 or j0 >= j1:
                    continue
                xs = self.w + (np.arange(i0, i1) + 0.5) * self.dx
                ys = self.s + (np.arange(j0, j1) + 0.5) * self.dy
                X, Y = np.meshgrid(xs, ys)
                pts = np.column_stack([X.ravel(), Y.ravel()])
                inside = MPath(np.asarray(rings[0])[:, :2]).contains_points(pts)
                for hole in rings[1:]:
                    inside &= ~MPath(np.asarray(hole)[:, :2]).contains_points(pts)
                self.mask[j0:j1, i0:i1] |= inside.reshape(X.shape)

    def _col(self, lon: float) -> int:
        return int(math.floor((lon - self.w) / self.dx))

    def _row(self, lat: float) -> int:
        return int(math.floor((lat - self.s) / self.dy))

    def fraction(self, geom: dict) -> float:
        """geom 里有多大比例的格子是养殖区（格子落在网格外的按"不是"算）。"""
        from matplotlib.path import Path as MPath
        import numpy as np

        gw, gs, ge, gn = geometry_bounds(geom)
        i0, i1 = self._col(gw), self._col(ge) + 1
        j0, j1 = self._row(gs), self._row(gn) + 1
        ii = np.arange(i0, i1)
        jj = np.arange(j0, j1)
        X, Y = np.meshgrid(self.w + (ii + 0.5) * self.dx, self.s + (jj + 0.5) * self.dy)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        inside = np.zeros(len(pts), dtype=bool)
        for rings in polys:
            m = MPath(np.asarray(rings[0])[:, :2]).contains_points(pts)
            for hole in rings[1:]:
                m &= ~MPath(np.asarray(hole)[:, :2]).contains_points(pts)
            inside |= m
        if not inside.any():
            return 0.0
        I, J = np.meshgrid(ii, jj)
        I, J = I.ravel()[inside], J.ravel()[inside]
        ok = (I >= 0) & (I < self.nx) & (J >= 0) & (J < self.ny)
        hit = np.zeros(len(I), dtype=bool)
        hit[ok] = self.mask[J[ok], I[ok]]
        return float(hit.mean())


def bbox_polygon(bbox: Iterable[float]) -> dict:
    w, s, e, n = bbox
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def load_geojson(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_geojson(fc: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=1)


def feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}
