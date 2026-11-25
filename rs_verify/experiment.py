"""造假实验：评估核验算法能识别多少虚假声明。

输出（outputs/<area>/）：
  claims_scored.csv     每份声明的得分
  metrics.json          各类造假的 AUC、检出率、误报率
  roc.png               ROC 曲线（三类造假）
  sensitivity_shift.png 检出率随围栏偏移距离的变化
  quantity.png          产量冒用检出率随放大倍数的变化

用法：
  python -m rs_verify.experiment --area ailun \
      --claims data/claims/ailun_claims.csv --activity data/activity/ailun_activity.csv --calibrate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from . import config as C  # noqa: E402
from .verify import calibrate_active_threshold, prepare_activity, verify_claims  # noqa: E402

CLEAN_OVERLAP = 0.1  # 假围栏与参考养殖区重叠 < 10% 才算"挪到了别处"


def _score(df: pd.DataFrame) -> np.ndarray:
    """数据不足（无法核验）按 0 分处理 = 不予通过。"""
    return df["confidence"].fillna(0.0).to_numpy()


def _auc(pos: pd.DataFrame, neg: pd.DataFrame, score_fn=_score) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[score_fn(pos), score_fn(neg)]
    return float(roc_auc_score(y, s))


def youden_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    return float(min(thr[i], 1.0))


def evaluate(res: pd.DataFrame) -> dict:
    true = res[res["label"] == "true"]
    loc = res[res["label"] == "fake_location"]
    loc_clean = loc[loc["overlap_ref"] < CLEAN_OVERLAP]
    tim = res[res["label"] == "fake_time"]
    qty = res[res["label"] == "fake_quantity"]

    # 统一的"通过"阈值：在真实 vs (位置冒用[干净] + 时段错报) 上取 Youden 最优
    thr = youden_threshold(_score(true), np.r_[_score(loc_clean), _score(tim)])
    passed = lambda d: (_score(d) >= thr)  # noqa: E731

    m = {
        "n_claims": {k: int(v) for k, v in res["label"].value_counts().items()},
        "insufficient_rate": {k: float(v) for k, v in res.groupby("label")["status"].apply(lambda s: (s != "ok").mean()).items()},
        "confidence_threshold": thr,
        "false_alarm_rate_true": float(1 - passed(true).mean()) if len(true) else float("nan"),
        "location": {
            "auc_all": _auc(true, loc),
            "auc_clean": _auc(true, loc_clean),
            "share_landing_on_other_farms": float((loc["overlap_ref"] >= CLEAN_OVERLAP).mean()) if len(loc) else float("nan"),
            "detection_by_shift_clean": {
                int(d): float(1 - passed(g).mean()) for d, g in loc_clean.groupby("shift_m")
            },
            "detection_by_shift_all": {int(d): float(1 - passed(g).mean()) for d, g in loc.groupby("shift_m")},
        },
        "time": {
            "auc": _auc(true, tim),
            "detection_by_months": {int(k): float(1 - passed(g).mean()) for k, g in tim.groupby("time_shift_months")},
        },
        "quantity": {
            # 规则：产量比 > 1 判为可疑
            "false_alarm_rate_true": float((true["qty_ratio"].fillna(np.inf) > 1).mean()) if len(true) else float("nan"),
            "auc": _auc(true, qty, score_fn=lambda d: -d["qty_ratio"].replace(np.inf, 1e9).fillna(1e9).to_numpy()),
            "detection_by_inflation": {
                float(r): float((g["qty_ratio"].fillna(np.inf) > 1).mean()) for r, g in qty.groupby("inflation")
            },
        },
    }
    return m


def plot_all(res: pd.DataFrame, m: dict, outdir: Path) -> None:
    true = res[res["label"] == "true"]
    groups = {
        "Location (clean)": res[(res["label"] == "fake_location") & (res["overlap_ref"] < CLEAN_OVERLAP)],
        "Time": res[res["label"] == "fake_time"],
    }
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, neg in groups.items():
        if len(neg) and len(true):
            y = np.r_[np.ones(len(true)), np.zeros(len(neg))]
            s = np.r_[_score(true), _score(neg)]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, s):.2f})")
    qty = res[res["label"] == "fake_quantity"]
    if len(qty) and len(true):
        y = np.r_[np.ones(len(true)), np.zeros(len(qty))]
        s = -np.r_[true["qty_ratio"].replace(np.inf, 1e9).fillna(1e9), qty["qty_ratio"].replace(np.inf, 1e9).fillna(1e9)]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"Quantity (AUC={roc_auc_score(y, s):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for key, lab in [("detection_by_shift_clean", "Moved to open sea / elsewhere"), ("detection_by_shift_all", "All shifted claims")]:
        d = m["location"][key]
        if d:
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], "o-", label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Geofence shift distance (m)")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sensitivity_shift.png", dpi=200)
    plt.close(fig)

    d = m["quantity"]["detection_by_inflation"]
    if d:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], "o-")
        ax.axhline(m["quantity"]["false_alarm_rate_true"], color="gray", ls="--", label="False alarm (true claims)")
        ax.set_xlabel("Declared quantity inflation factor")
        ax.set_ylabel("Detection rate")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "quantity.png", dpi=200)
        plt.close(fig)


def run(area: str, claims_path: str, activity_paths: list[str], calibrate: bool, outdir: str | None = None) -> dict:
    claims = pd.read_csv(claims_path)
    act_raw = pd.concat([pd.read_csv(p) for p in activity_paths], ignore_index=True)
    thr = calibrate_active_threshold(act_raw, claims) if calibrate else C.ACTIVE_FRAC_THRESHOLD
    act = prepare_activity(act_raw, active_threshold=thr)
    res = verify_claims(act, claims)
    m = evaluate(res)
    m["active_frac_threshold"] = thr

    od = Path(outdir or Path(C.OUTPUT_DIR) / area)
    od.mkdir(parents=True, exist_ok=True)
    res.to_csv(od / "claims_scored.csv", index=False)
    with open(od / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    plot_all(res, m, od)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="造假实验评估")
    ap.add_argument("--area", required=True, choices=list(C.STUDY_AREAS))
    ap.add_argument("--claims", required=True)
    ap.add_argument("--activity", required=True, nargs="+", help="一个或多个活动 CSV（GEE 导出可能分多个文件）")
    ap.add_argument("--calibrate", action="store_true", help="用数据自动标定'有设施'阈值")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    m = run(args.area, args.claims, args.activity, args.calibrate, args.outdir)
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
