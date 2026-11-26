"""论文主图（英文标注）。

Fig. A  取舍图（tradeoff_table.csv → fig_tradeoff.png/.pdf）
  (a) 筛查名单 vs 目视真值：横轴 = 图上有筏却进名单（真冤枉），纵轴 = 图上没筏进了名单（真查出）；
      一季 / 连续两季两条线，点 = 旋钮 A（判没筏分位 q），星号 = 均衡档。
  (b) 模拟"有证不养"检出率 vs 诚实户被判可疑（产量旋钮固定在均衡档 p），两种荒芜规则。
  (c) 模拟洗货检出率 vs 诚实户被判可疑（q 固定在均衡档，一季规则），按放大倍数分线，点 = 旋钮 B（p）。
Fig. B  两级名单与新《渔业法》第六十五条第二款的对应流程图（fig_workflow.png/.pdf），不需要数据。

用法：
  python -m rs_verify.paper_figs --tradeoff-table outputs/v3_tradeoff_per/tradeoff_table.csv --outdir outputs/paper_figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# tradeoff_table.csv 是中文列名（tradeoff.to_cn），这里映射回英文键
CN2KEY = {
    "旋钮C_荒芜认定季数": "seasons", "旋钮A_判没筏分位q": "q", "旋钮B_产量预设误报p": "p",
    "诚实户_被判可疑(冤枉率)": "honest", "有证不养_检出率": "idle",
    "洗货1.5倍_检出率": "l15", "洗货2倍_检出率": "l2", "洗货3倍_检出率": "l3",
    "目视有筏_进名单(真冤枉)": "v_false", "目视无筏_进名单(真检出)": "v_found", "名单中目视确为空": "v_purity",
}
EN2KEY = {"idle_seasons": "seasons", "q_absent": "q", "p_quantity": "p", "honest_flagged": "honest",
          "S2_idle_flagged": "idle", "S4_launder_x1.5_flagged": "l15", "S4_launder_x2_flagged": "l2",
          "S4_launder_x3_flagged": "l3", "visual_farmed_listed": "v_false", "visual_empty_listed": "v_found",
          "visual_list_truly_empty": "v_purity"}
BAL_Q, BAL_P = 0.05, 0.02      # 均衡档（tradeoff.PRESETS）
STYLE = {1: dict(marker="o", color="#1f5fa8", label="1-season rule (rectification notice)"),
         2: dict(marker="s", color="#c0392b", label="2-season rule (revocation review)")}


def load_table(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(columns={**CN2KEY, **EN2KEY})
    d["seasons"] = d["seasons"].astype(int)
    return d


def fig_tradeoff(d: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) 目视真值
    ax = axes[0]
    if {"v_false", "v_found"} <= set(d.columns):
        for sn, g in d.groupby("seasons"):
            g = g.drop_duplicates("q").sort_values("q")
            st = STYLE.get(sn, {})
            ax.plot(g["v_false"] * 100, g["v_found"] * 100, "-", marker=st.get("marker", "^"),
                    color=st.get("color"), label=st.get("label", f"{sn}-season"))
            for r in g.itertuples():
                ax.annotate(f"q={r.q:g}", (r.v_false * 100, r.v_found * 100), textcoords="offset points",
                            xytext=(5, -9 if sn == 2 else 4), fontsize=7, color=st.get("color"))
            b = g[g["q"] == BAL_Q]
            ax.plot(b["v_false"] * 100, b["v_found"] * 100, "*", ms=16, color=st.get("color"), mec="k", mew=0.6)
        ax.set_xlabel("Truly farmed parcels put on the list (%)\n(false accusation, visual truth)")
        ax.set_ylabel("Truly empty parcels put on the list (%)\n(detection, visual truth)")
        ax.set_title("(a) Screening list vs. visual truth", loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(left=-0.3)
        ax.set_ylim(0, 100)
    # (b) 模拟有证不养
    ax = axes[1]
    for sn, g in d[d["p"] == BAL_P].groupby("seasons"):
        g = g.sort_values("q")
        st = STYLE.get(sn, {})
        ax.plot(g["honest"] * 100, g["idle"] * 100, "-", marker=st.get("marker", "^"), color=st.get("color"),
                label=st.get("label", f"{sn}-season"))
        b = g[g["q"] == BAL_Q]
        ax.plot(b["honest"] * 100, b["idle"] * 100, "*", ms=16, color=st.get("color"), mec="k", mew=0.6)
    ax.set_xlabel("Honest farmers flagged as suspicious (%)")
    ax.set_ylabel("Idle permits flagged (%)")
    ax.set_title(f"(b) Idle permits (simulated), p = {BAL_P:g}", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 100)
    # (c) 模拟洗货
    ax = axes[2]
    g = d[(d["q"] == BAL_Q) & (d["seasons"] == 1)].sort_values("p")
    for col, lab, c in [("l15", "Declared ×1.5", "#7fb3d5"), ("l2", "Declared ×2", "#2e86c1"),
                        ("l3", "Declared ×3", "#1b4f72")]:
        if col in g:
            ax.plot(g["honest"] * 100, g[col] * 100, "-o", color=c, label=lab)
    for r in g.itertuples():
        ax.annotate(f"p={r.p:g}", (r.honest * 100, getattr(r, "l3", 0) * 100), textcoords="offset points",
                    xytext=(3, 5), fontsize=7, color="#1b4f72")
    ax.set_xlabel("Honest farmers flagged as suspicious (%)")
    ax.set_ylabel("Laundering batches flagged (%)")
    ax.set_title(f"(c) Quantity laundering (simulated), q = {BAL_Q:g}", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="center right")
    ax.set_ylim(0, 105)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, fs=9, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=fc, ec="#333333", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def _arrow(ax, a, b, text=None, rad=0.0, dy=0.012):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.005, (a[1] + b[1]) / 2 + dy, text, ha="left", va="center", fontsize=7.5,
                style="italic", color="#444444")


def fig_workflow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    grey, blue, green, amber, red = "#eeeeee", "#d6e6f5", "#d5f0dd", "#fdebc8", "#f8d0cb"
    _box(ax, (0.02, 0.60), 0.14, 0.24, "Aquaculture permit\n(registered parcel\nboundary + holder,\ngovernment survey)", grey)
    _box(ax, (0.20, 0.60), 0.17, 0.24, "Annual satellite check\nSentinel-1 SAR (all year)\n+ Sentinel-2 best scenes\n"
         "(kelp season, Dec–May)", blue)
    _box(ax, (0.42, 0.78), 0.14, 0.10, "Present → pass", green)
    _box(ax, (0.42, 0.645), 0.14, 0.10, "Weak / conflicting\n→ review", amber)
    _box(ax, (0.42, 0.51), 0.14, 0.10, "Absent in both\nsensors this season", red)
    _box(ax, (0.61, 0.51), 0.15, 0.10, "Level-1 list:\nrectification notice", "#f5b7b1", bold=True)
    _box(ax, (0.81, 0.51), 0.17, 0.10, "Holder explains or\nresumes farming", grey)
    _box(ax, (0.61, 0.24), 0.15, 0.12, "Absent again next season\n→ Level-2 list:\nrevocation review",
         "#ec7063", bold=True)
    _box(ax, (0.81, 0.24), 0.17, 0.12, "On-site inspection\nby the fishery authority\n(decides revocation)", grey)
    ax.text(0.685, 0.66, "Fisheries Law Art. 65(2):\n\"order to develop within a time limit\"",
            ha="center", va="bottom", fontsize=7.5, color="#7b241c")
    ax.text(0.79, 0.175, "\"... if not developed when the time limit expires,\nrevoke the aquaculture permit\"",
            ha="center", va="center", fontsize=7.5, color="#7b241c")
    _box(ax, (0.02, 0.17), 0.35, 0.25, "Traceability layer\n\nEach batch ID is bound to a permit.\n"
         "Registry checks stop outside-zone and borrowed-parcel claims.\nCumulative declared quantity is compared with "
         "satellite capacity.\nBatches from Level-2 parcels are flagged on the QR page.", "#f4f6f7", fs=8.5)
    _arrow(ax, (0.16, 0.72), (0.20, 0.72))
    _arrow(ax, (0.37, 0.72), (0.42, 0.83))
    _arrow(ax, (0.37, 0.72), (0.42, 0.695))
    _arrow(ax, (0.37, 0.72), (0.42, 0.56))
    _arrow(ax, (0.56, 0.56), (0.61, 0.56))
    _arrow(ax, (0.76, 0.56), (0.81, 0.56))
    _arrow(ax, (0.685, 0.51), (0.685, 0.36), "next annual check", dy=0.0)
    _arrow(ax, (0.76, 0.30), (0.81, 0.30))
    _arrow(ax, (0.09, 0.60), (0.09, 0.42), "batch IDs", dy=0.0)
    _arrow(ax, (0.61, 0.30), (0.37, 0.30), "flag batches", dy=0.025)
    ax.text(0.02, 0.95, "Satellites screen registered parcels once a year; enforcement decisions stay with inspectors.",
            fontsize=10.5, weight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="论文主图")
    ap.add_argument("--tradeoff-table", required=True, help="tradeoff.py 输出的 tradeoff_table.csv（建议 persist 版）")
    ap.add_argument("--outdir", default="outputs/paper_figs")
    args = ap.parse_args()
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    fig_tradeoff(load_table(args.tradeoff_table), od / "fig_tradeoff")
    fig_workflow(od / "fig_workflow")
    print("输出目录：", od)


if __name__ == "__main__":
    main()
