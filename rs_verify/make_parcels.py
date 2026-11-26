"""把参考养殖区切成规则网格"地块"，作为产地声明的基本单位。

为什么要切：已发表数据集把整片海域的浮筏连成一个多边形（单块可达几千公顷）。
真实的养殖户声明的是自己那一块海（几公顷到几十公顷），而且把一块 4 公里宽的围栏挪 100 m，
几乎还压在自己身上，造假实验就没有意义。所以用 500 m × 500 m（25 公顷）的网格切开，
只保留"大部分面积是养殖区"的格子（默认覆盖率 ≥ 60%）。

输出：
  data/reference/<area>_parcels.geojson   地块（属性：pid, cover, area_m2, product, source）
  outputs/<area>/parcels_preview.png       预览图：灰色 = 参考养殖区，彩色方格 = 保留的地块（颜色 = 覆盖率）
加 --copy-to <文件夹> 可把这两个文件再复制一份过去。

用法：
  python -m rs_verify.make_parcels --area ailun --reference data/reference/ailun_fu2021_reference.geojson
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402

RES_M = 20.0


def make_parcels(ref_fc: dict, bbox: list[float], farm_type: str, cell_m: float = 500.0,
                 min_cover: float = 0.6, res_m: float = RES_M, source: str = "") -> tuple[dict, geo.RasterMask]:
    k = int(round(cell_m / res_m))
    if abs(k * res_m - cell_m) > 1e-6:
        raise ValueError(f"地块边长 {cell_m} m 必须是栅格分辨率 {res_m} m 的整数倍")
    rm = geo.RasterMask(ref_fc, bbox, res_m)
    ny, nx = (rm.ny // k) * k, (rm.nx // k) * k
    blocks = rm.mask[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))
    product = C.DEFAULT_PRODUCT[farm_type]
    feats = []
    for j, i in zip(*np.nonzero(blocks >= min_cover)):
        x0, y0 = rm.w + i * k * rm.dx, rm.s + j * k * rm.dy
        x1, y1 = x0 + k * rm.dx, y0 + k * rm.dy
        ring = [[round(x0, 7), round(y0, 7)], [round(x1, 7), round(y0, 7)], [round(x1, 7), round(y1, 7)],
                [round(x0, 7), round(y1, 7)], [round(x0, 7), round(y0, 7)]]
        g = {"type": "Polygon", "coordinates": [ring]}
        feats.append({"type": "Feature", "geometry": g,
                      "properties": {"pid": f"P{len(feats) + 1:04d}", "cover": round(float(blocks[j, i]), 3),
                                     "area_m2": round(geo.geometry_area_m2(g), 1), "product": product,
                                     "source": source}})
    return geo.feature_collection(feats), rm


def preview(rm: geo.RasterMask, parcels: dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(np.where(rm.mask, 1.0, np.nan), extent=[rm.w, rm.w + rm.nx * rm.dx, rm.s, rm.s + rm.ny * rm.dy],
              origin="lower", cmap="Greys", vmin=0, vmax=2, interpolation="nearest")
    verts = [np.array(f["geometry"]["coordinates"][0]) for f in parcels["features"]]
    if verts:
        pc = PolyCollection(verts, array=np.array([f["properties"]["cover"] for f in parcels["features"]]),
                            cmap="viridis", edgecolor="k", linewidth=0.2, alpha=0.8)
        pc.set_clim(0.6, 1.0)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.6, label="raft cover in parcel")
    ax.set_xlim(rm.w, rm.e)
    ax.set_ylim(rm.s, rm.n)
    ax.set_aspect(1 / np.cos(np.radians((rm.s + rm.n) / 2)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="参考养殖区 → 网格地块")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--reference", required=True)
    ap.add_argument("--cell-m", type=float, default=500.0, help="地块边长（米），须为 20 的整数倍")
    ap.add_argument("--min-cover", type=float, default=0.6, help="格子里养殖区面积占比至少多少才保留")
    ap.add_argument("--copy-to", default=None)
    args = ap.parse_args()

    info = C.STUDY_AREAS[args.area]
    ref = geo.load_geojson(args.reference)
    src = (ref["features"][0]["properties"] or {}).get("source", "") if ref["features"] else ""
    parcels, rm = make_parcels(ref, info["bbox"], info["farm_type"], args.cell_m, args.min_cover, source=src)

    out_gj = Path(C.DATA_DIR) / "reference" / f"{args.area}_parcels.geojson"
    geo.save_geojson(parcels, out_gj)
    od = Path(C.OUTPUT_DIR) / args.area
    od.mkdir(parents=True, exist_ok=True)
    png = od / "parcels_preview.png"
    preview(rm, parcels, png, f"{args.area}: {len(parcels['features'])} parcels ({args.cell_m:.0f} m)")

    covers = np.array([f["properties"]["cover"] for f in parcels["features"]]) if parcels["features"] else np.array([0])
    print(f"参考养殖区栅格化：{rm.nx}×{rm.ny} 格（{RES_M:.0f} m），养殖区面积 {rm.mask.sum() * RES_M ** 2 / 1e6:.1f} km²")
    print(f"地块：{len(parcels['features'])} 个（{args.cell_m:.0f} m 见方，覆盖率 ≥ {args.min_cover}），"
          f"覆盖率中位数 {np.median(covers):.2f}")
    print("已保存：", out_gj)
    print("已保存：", png)
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in (out_gj, png):
            shutil.copy2(p, dst / p.name)
        print("已复制到：", dst)


if __name__ == "__main__":
    main()
