"""误判核查图集：雷达年度检查和参考标签"对不上"的地块，逐个拿同一年的 Sentinel-2 光学图来看。

为什么要做：参考边界（Fu et al. 2021）是 2016–2019 年的，检查年份是 2019–2025。
  - "养殖地块却被判为不在养"：可能是雷达误判，也可能这块地真的弃养/拆了；
  - "空海面却被判为在养"：可能是雷达误判，也可能后来新建了筏子。
光学图上浮筏是一条条清楚的细线，肉眼就能判断。核查结果可以把"标签错了"和"雷达错了"分开。

输入：annual.py 的 outputs/<area>_annual/annual_composite.csv + 声明多边形 GeoJSON。
输出（outputs/<area>_gallery/）：
  gallery_farm_absent.png    养殖地块被判"不在养"（误报候选）
  gallery_sea_present.png    空海面被判"在养"（漏检候选）
  gallery_controls.png       判对的对照样本（养殖+空海面各若干）
  gallery_index.csv          每个小图的编号、地块、年份、指标值、判定，留一列给人工填写
每个小图：该年检查窗口内 S2 去云合成真彩色，黄框 = 声明多边形，周围留 600 m。

用法（需要 GEE）：
  python -m rs_verify.disagreement_gallery --area ailun --project <PROJECT> \
      --claim-geoms data/claims/ailun_claim_geoms.geojson \
      --composite outputs/ailun_annual/annual_composite.csv --metrics outputs/ailun_annual/annual_metrics.json
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from . import geo  # noqa: E402

PAD_M = 600


def pick_cases(comp: pd.DataFrame, n: int = 24, seed: int = C.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    te = comp[(comp["split"] == "test") & (comp["status"] == "ok")]
    farm = te[te["kind"] == "farm"]
    sea = te[te["open_sea"].astype(bool)]
    take = lambda d, k: d.sample(min(k, len(d)), random_state=seed) if len(d) else d  # noqa: E731
    return {
        "farm_absent": take(farm[farm["present"] == 0], n),
        "sea_present": take(sea[sea["present"] == 1], n),
        "controls": pd.concat([take(farm[farm["present"] == 1], n // 2), take(sea[sea["present"] == 0], n // 2)]),
    }


def window_dates(year: int, window: list[int]) -> tuple[str, str]:
    wraps = any(window[i] > window[i + 1] for i in range(len(window) - 1))
    y0 = year - 1 if wraps else year
    start = pd.Timestamp(y0, window[0], 1)
    end = pd.Timestamp(year, window[-1], 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def make_gee_fetcher(project: str, window: list[int]) -> Callable[[dict, int], np.ndarray | None]:
    import ee

    from . import gee_common as G

    G.init(project)

    def fetch(geom: dict, year: int) -> np.ndarray | None:
        w, s, e, n = geo.geometry_bounds(geom)
        dlon, dlat = geo._deg_per_m((s + n) / 2)
        box = ee.Geometry.Rectangle([w - PAD_M * dlon, s - PAD_M * dlat, e + PAD_M * dlon, n + PAD_M * dlat],
                                    proj="EPSG:4326", geodesic=False)
        start, end = window_dates(year, window)
        comp = G.s2_composite(box, start, end, max_cloud=60)
        if int(comp.get("n_images").getInfo() or 0) == 0:  # 窗口内没有可用光学图 → 放宽到同年 1–5 月
            comp = G.s2_composite(box, f"{year}-01-01", f"{year}-06-01", max_cloud=60)
        rgb = comp.select(["B4", "B3", "B2"]).visualize(min=0, max=2500)
        outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(geom))]), 1, 2) \
            .visualize(palette=["ffff00"])
        img = ee.ImageCollection([rgb, outline]).mosaic()
        url = img.getThumbURL({"region": box, "dimensions": 256, "format": "png"})
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return plt.imread(io.BytesIO(r.read()), format="png")
        except Exception as ex:  # noqa: BLE001
            print("  取图失败：", ex)
            return None

    return fetch


def gallery(cases: pd.DataFrame, geoms: dict[str, dict], fetch, title: str, out: Path, start_no: int) -> list[dict]:
    n = len(cases)
    if n == 0:
        return []
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()
    index = []
    for i, (ax, r) in enumerate(zip(axes, cases.itertuples(index=False))):
        no = start_no + i
        img = fetch(geoms[r.pid], int(r.year))
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center")
        verdict = "present" if r.present == 1 else "absent"
        ax.set_title(f"#{no} {r.pid[:14]}\n{int(r.year)} m={r.m:.1f} {verdict}", fontsize=7)
        ax.axis("off")
        index.append({"no": no, "pid": r.pid, "year": int(r.year), "kind": r.kind, "m": round(float(r.m), 3),
                      "sar_verdict": verdict, "s2_visual_raft(yes/no/unclear)": ""})
        print(f"  #{no} {r.pid} {int(r.year)}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="误判核查图集（S2 真彩色）")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--project", required=True)
    ap.add_argument("--claim-geoms", required=True)
    ap.add_argument("--composite", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--n", type=int, default=24, help="每类最多几个样本")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    comp = pd.read_csv(args.composite)
    window = json.load(open(args.metrics, encoding="utf-8"))["window_months"]
    geoms = {str(f["properties"]["pid"]): f["geometry"] for f in geo.load_geojson(args.claim_geoms)["features"]}
    picks = pick_cases(comp, args.n)
    fetch = make_gee_fetcher(args.project, window)

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_gallery")
    od.mkdir(parents=True, exist_ok=True)
    index, no = [], 1
    titles = {"farm_absent": "Farm parcels judged ABSENT by SAR (false-alarm candidates)",
              "sea_present": "Open-sea polygons judged PRESENT by SAR (new rafts?)",
              "controls": "Controls (SAR agrees with reference)"}
    for key, d in picks.items():
        print(f"{key}: {len(d)} 个")
        idx = gallery(d, geoms, fetch, f"{titles[key]} — S2 RGB, inspection window {window}",
                      od / f"gallery_{key}.png", no)
        for x in idx:
            x["group"] = key
        index += idx
        no += len(idx)
    pd.DataFrame(index).to_csv(od / "gallery_index.csv", index=False, encoding="utf-8-sig")
    print("输出目录：", od)


if __name__ == "__main__":
    main()
