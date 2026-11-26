import numpy as np
import pandas as pd

from rs_verify.fusion import calibrate, decide, s2_annual


def test_s2_best_scenes_and_decision():
    rows = []
    for pid, tex in [("F", 0.12), ("S", 0.03)]:
        for i, b2 in enumerate([0.03, 0.04, 0.09, 0.05, 0.06, 0.035, 0.08]):
            rows.append({"pid": pid, "date": f"2021-0{i % 9 + 1}-1{i}", "tex": tex if b2 < 0.085 else 0.01,
                         "clear": 1.0 if i != 3 else 0.5, "b2": b2, "n_pix": 2500, "area_m2": 250000})
    s2 = s2_annual(pd.DataFrame(rows), [2021], k_best=5)
    f_ = s2.set_index("pid")
    assert f_.loc["F", "n_s2"] == 5 and abs(f_.loc["F", "tex_s2"] - 0.12) < 1e-9  # 雾霾最重的那张没用上


def test_precision_first_decision():
    rng = np.random.default_rng(0)
    n = 200
    farm = pd.DataFrame({"kind": "farm", "open_sea": False, "split": "calib",
                         "m_sar": rng.normal(4, 1, n), "tex_s2": rng.normal(0.12, 0.02, n)})
    sea = pd.DataFrame({"kind": "shifted", "open_sea": True, "split": "calib",
                        "m_sar": rng.normal(1, 0.7, n), "tex_s2": rng.normal(0.04, 0.01, n)})
    f = pd.concat([farm, sea], ignore_index=True)
    th = calibrate(f)
    assert th["L_sar"] <= th["T_sar"] and th["L_s2"] <= th["T_s2"]
    d = decide(f, th)
    # 真实养殖地块几乎不会被判"不在养"
    assert (d.loc[d["kind"] == "farm", "decision"] == "absent").mean() <= 0.01
    assert (d.loc[d["open_sea"], "decision"] == "absent").mean() >= 0.8
    # 雷达漏看但光学看到 → 在养；只有雷达、且雷达说没有 → 待复核
    one = pd.DataFrame({"m_sar": [0.0, 0.0], "tex_s2": [0.2, np.nan]})
    r = decide(one, th)["decision"].tolist()
    assert r == ["present", "review"]
