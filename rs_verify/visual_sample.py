"""目视判读样本：分层抽取"多边形 × 年"，每个配一张当年最清晰的 Sentinel-2 单景真彩色小图，供人工判读真值。

为什么：参考数据（2016–2019）已过时，"空海面"里混有后来新建的筏区，不能直接当真值。
目视判读光学图是遥感验证的标准做法：浮筏一格一格非常清楚，人眼判断可靠。

盲判：小图标题只有编号，不显示雷达/光学的判定和地块类型，避免判读时被结论带偏。
分层：{养殖地块, 空海面, 挪到邻近养殖区的围栏} × {雷达说有, 雷达说没有}，每层最多 n 个。
选图：该多边形该年所有晴空（≥95%）且覆盖完整的光学景里，蓝波段最低（雾霾最轻）的一张。

输入：fusion.py 的 fusion_composite.csv、S2 导出表、声明多边形 GeoJSON。
输出（outputs/<area>_visual/）：
  visual_page_01.png …     每页 36 张（6×6），黄框 = 多边形，周围留 600 m
  visual_index.csv          编号 → pid, year, date, 分层（判读时不看），visual 列留空待填 yes/no/unclear

用法（需要 GEE）：
  python -m rs_verify.visual_sample --area ailun --project <PROJECT> \
      --claim-geoms data/claims/ailun_claim_geoms.geojson \
      --fusion outputs/ailun_fusion/fusion_composite.csv --s2 data/activity_s2/ailun_s2_*.csv --n 34
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402

PAD_M = 600
PER_PAGE = 36


def best_dates(s2_raw: pd.DataFrame, min_clear: float = C.S2_MIN_CLEAR) -> pd.DataFrame:
    a = s2_raw.copy()
    a["pid"] = a["pid"].astype(str)
    a["date"] = pd.to_datetime(a["date"])
    a["year"] = a["date"].dt.year
    a["cov"] = pd.to_numeric(a["n_pix"]) * 100.0 / pd.to_numeric(a["area_m2"])
    a = a[(a["clear"] >= min_clear) & (a["cov"] >= 0.8)].dropna(subset=["b2"])
    a = a.sort_values("b2").drop_duplicates(["pid", "year"])
    return a[["pid", "year", "date", "b2", "tile"]]


def stratified(f: pd.DataFrame, dates: pd.DataFrame, n: int, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    f = f.copy()
    f["group"] = np.select([f["kind"] == "farm", f["open_sea"].astype(bool),
                            (f["kind"] == "shifted") & (f["overlap_ref"] >= 0.5)],
                           ["farm", "open_sea", "near_farm"], default="other")
    th_sar = f["T_sar"].iloc[0] if "T_sar" in f else np.nanmedian(f["m_sar"])
    f["sar_says"] = np.where(f["m_sar"] >= th_sar, "sar_yes", "sar_no")
    f = f[f["group"] != "other"].merge(dates, on=["pid", "year"], how="inner")
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in f.groupby(["group", "sar_says"])]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)  # 打乱顺序，盲判
    out.insert(0, "no", np.arange(1, len(out) + 1))
    return out


def make_fetcher(project: str):
    import ee

    from . import gee_common as G

    G.init(project)

    def fetch(geom: dict, date: pd.Timestamp) -> np.ndarray | None:
        w, s, e, n = geo.geometry_bounds(geom)
        dlon, dlat = geo._deg_per_m((s + n) / 2)
        box = ee.Geometry.Rectangle([w - PAD_M * dlon, s - PAD_M * dlat, e + PAD_M * dlon, n + PAD_M * dlat],
                                    proj="EPSG:4326", geodesic=False)
        d0 = date.strftime("%Y-%m-%d")
        d1 = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box).filterDate(d0, d1).mosaic()
        rgb = img.select(["B4", "B3", "B2"]).visualize(min=0, max=2500)
        outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(geom))]), 1, 2) \
            .visualize(palette=["ffff00"])
        url = ee.ImageCollection([rgb, outline]).mosaic().getThumbURL(
            {"region": box, "dimensions": 256, "format": "png"})
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return plt.imread(io.BytesIO(r.read()), format="png")
        except Exception as ex:  # noqa: BLE001
            print("  取图失败：", ex)
            return None

    return fetch


def pages(sample: pd.DataFrame, geoms: dict[str, dict], fetch, od: Path) -> list[Path]:
    outs = []
    for p0 in range(0, len(sample), PER_PAGE):
        chunk = sample.iloc[p0:p0 + PER_PAGE]
        fig, axes = plt.subplots(6, 6, figsize=(14.4, 15.6))
        axes = axes.ravel()
        for ax, r in zip(axes, chunk.itertuples(index=False)):
            img = fetch(geoms[r.pid], pd.Timestamp(r.date))
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "no image", ha="center", va="center")
            ax.set_title(f"#{r.no}", fontsize=9)
            ax.axis("off")
            print(f"  #{r.no}")
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.suptitle("Visual interpretation sample — is there raft aquaculture inside the yellow box? "
                     "(S2 RGB, clearest scene of the year)", fontsize=10)
        fig.tight_layout()
        out = od / f"visual_page_{p0 // PER_PAGE + 1:02d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        outs.append(out)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="目视判读样本图集（盲判）")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--project", required=True)
    ap.add_argument("--claim-geoms", required=True)
    ap.add_argument("--fusion", required=True, help="fusion.py 输出的 fusion_composite.csv")
    ap.add_argument("--fusion-metrics", default=None, help="fusion_metrics.json（取雷达门限；不给则用中位数分层）")
    ap.add_argument("--s2", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=34, help="每层最多几个")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    f = pd.read_csv(args.fusion)
    f["pid"] = f["pid"].astype(str)
    mpath = args.fusion_metrics or str(Path(args.fusion).with_name("fusion_metrics.json"))
    if Path(mpath).exists():
        f["T_sar"] = json.load(open(mpath, encoding="utf-8"))["thresholds"]["T_sar"]
    files = sorted({x for p in args.s2 for x in (glob.glob(p) or [p])})
    dates = best_dates(pd.concat([pd.read_csv(p) for p in files], ignore_index=True))
    sample = stratified(f, dates, args.n)
    geoms = {str(x["properties"]["pid"]): x["geometry"] for x in geo.load_geojson(args.claim_geoms)["features"]}

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_visual")
    od.mkdir(parents=True, exist_ok=True)
    idx = sample[["no", "pid", "year", "date", "group", "sar_says"]].copy()
    idx["date"] = pd.to_datetime(idx["date"]).dt.strftime("%Y-%m-%d")
    idx["visual"] = ""
    idx.to_csv(od / "visual_index.csv", index=False, encoding="utf-8-sig")
    print(f"样本 {len(sample)} 个：", sample.groupby(["group", "sar_says"]).size().to_dict())
    pages(sample, geoms, make_fetcher(args.project), od)
    print("输出目录：", od)


if __name__ == "__main__":
    main()
