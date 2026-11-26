"""诊断：养殖地块和空海面的雷达亮度，随时间怎么变？

用来回答"时间核验为什么不灵"：如果养殖地块全年都比空海面亮、没有明显的淡季，
那么"错报季节"在卫星上就看不出来。

输出（outputs/<area>/）：
  activity_diagnosis.png   (a) 2019–2025 逐景曲线：养殖地块 vs 空海面（中位数）
                           (b) 按月份汇总（中位数 + 四分位范围）
                           (c) 每景的海面背景亮度（dB），看大风天的影响
                           (d) 每景"养殖地块/空海面"的中位数 vs 背景亮度：看太平静或大风时指标是否失真
  (a)(b) 已去掉背景低于 config.MIN_BACKGROUND_DB 的景；(c)(d) 显示全部景，虚线是这个门限。
  activity_diagnosis.txt   按月份的数字表
加 --copy-to <文件夹> 可把这两个文件再复制一份过去。

用法：
  python -m rs_verify.diagnose_activity --area ailun --claims data/claims/ailun_claims.csv \
      --activity data/activity/ailun_activity_*.csv --copy-to G:/c_file
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .verify import prepare_activity  # noqa: E402


def groups(claims: pd.DataFrame) -> dict[str, set[str]]:
    cl = claims.copy()
    cl["pid"] = cl["pid"].astype(str)
    ref = set(cl.loc[cl["label"] == "true", "pid"])
    sea = cl[(cl["label"] == "fake_location") & (cl["shift_m"] >= 2000) & (cl["overlap_ref"] < 0.05)]
    near = cl[(cl["label"] == "fake_location") & (cl["overlap_ref"] >= 0.5)]
    return {"Farm parcels": ref, "Neighbouring farms (shifted, overlap≥0.5)": set(near["pid"]),
            "Open sea (shifted ≥2 km, no farm)": set(sea["pid"])}


def run(area: str, claims_path: str, activity_paths: list[str], outdir: str | None = None) -> tuple[Path, Path]:
    files = sorted({f for p in activity_paths for f in (glob.glob(p) or [p])})
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    act_all = prepare_activity(raw, min_bg_db=None)
    act = prepare_activity(raw)
    g = groups(pd.read_csv(claims_path))
    colors = {k: c for k, c in zip(g, ["tab:green", "tab:orange", "tab:blue"])}

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), gridspec_kw={"height_ratios": [1.2, 1, 0.7, 1]})
    lines = []
    for name, pids in g.items():
        a = act[act["pid"].isin(pids)]
        if a.empty:
            continue
        ts = a.groupby("date")["metric"].median()
        axes[0].plot(ts.index, ts.values, ".-", ms=2, lw=0.7, color=colors[name], label=f"{name} (n={len(pids)})")
        mo = a.groupby("month")["metric"].quantile([0.25, 0.5, 0.75]).unstack()
        mo = mo.reindex(range(1, 13))
        axes[1].plot(mo.index, mo[0.5], "o-", color=colors[name], label=name)
        axes[1].fill_between(mo.index, mo[0.25], mo[0.75], color=colors[name], alpha=0.15)
        lines.append(mo[0.5].rename(name))
    for ax in axes[:2]:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Excess backscatter\n(lin_ratio − 1)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{area}: per-scene median")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xlabel("Month")
    axes[1].set_title("Monthly median and IQR (all years)")
    bg = act_all.groupby("date")["bg_db"].first()
    axes[2].plot(bg.index, bg.values, ".", ms=3, color="gray")
    axes[2].axhline(C.MIN_BACKGROUND_DB, color="r", ls="--", lw=0.8)
    axes[2].set_ylabel("Sea background (dB)")
    axes[2].set_title("Per-scene sea-surface background (windy scenes are brighter)")
    for name, pids in g.items():
        a = act_all[act_all["pid"].isin(pids)]
        if a.empty or name.startswith("Neighbouring"):
            continue
        per = a.groupby("date").agg(m=("metric", "median"), bg=("bg_db", "first"))
        axes[3].scatter(per["bg"], per["m"], s=6, color=colors[name], label=name)
    axes[3].axvline(C.MIN_BACKGROUND_DB, color="r", ls="--", lw=0.8)
    axes[3].axhline(0, color="k", lw=0.6)
    axes[3].set_yscale("symlog", linthresh=1)
    axes[3].set_xlabel("Sea background (dB)")
    axes[3].set_ylabel("Per-scene median excess")
    axes[3].legend(fontsize=8)
    axes[3].set_title("Metric vs sea state (left of red line: calm sea near noise floor, dropped)")
    fig.tight_layout()

    od = Path(outdir or Path(C.OUTPUT_DIR) / area)
    od.mkdir(parents=True, exist_ok=True)
    png, txt = od / "activity_diagnosis.png", od / "activity_diagnosis.txt"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    tab = pd.concat(lines, axis=1).round(3) if lines else pd.DataFrame()
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"文件：{', '.join(files)}\n")
        f.write(f"有效记录：{len(act)}，日期 {act['date'].min().date()} ~ {act['date'].max().date()}，"
                f"景数 {act['date'].nunique()}（另有 {act_all['date'].nunique() - act['date'].nunique()} 景因海面背景"
                f" < {C.MIN_BACKGROUND_DB} dB 被去掉）\n")
        for name, pids in g.items():
            f.write(f"{name}: {len(pids)} 个多边形\n")
        f.write("\n按月份的中位数（lin_ratio − 1）：\n")
        f.write(tab.to_string())
        f.write("\n")
    print(txt.read_text(encoding="utf-8"))
    return png, txt


def main() -> None:
    ap = argparse.ArgumentParser(description="活动曲线诊断图")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--claims", required=True)
    ap.add_argument("--activity", required=True, nargs="+")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--copy-to", default=None)
    args = ap.parse_args()
    png, txt = run(args.area, args.claims, args.activity, args.outdir)
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in (png, txt):
            shutil.copy2(p, dst / p.name)
        print("已复制到：", dst)


if __name__ == "__main__":
    main()
