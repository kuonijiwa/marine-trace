import numpy as np
import pandas as pd

from rs_verify.fusion import calibrate, decide, s2_annual


def test_s2_best_scenes_and_decision():
    rows = []
    for pid, tex in [("F", 0.12), ("S", 0.03)]:
        for i, b2 in enumerate([0.03, 0.04, 0.09, 0.05, 0.06, 0.035, 0.08]):
            day = ["2021-01-10", "2021-02-11", "2021-03-12", "2021-04-13", "2021-05-14", "2020-12-15", "2021-01-20"][i]
            rows.append({"pid": pid, "date": day, "tex": tex if b2 < 0.085 else 0.01,
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


def test_s2_only_kelp_season():
    rows = [{"pid": "F", "date": d, "tex": t, "clear": 1.0, "b2": 0.03, "n_pix": 2500, "area_m2": 250000}
            for d, t in [("2021-08-10", 0.01), ("2021-09-10", 0.01), ("2020-12-20", 0.12), ("2021-03-01", 0.12)]]
    s2 = s2_annual(pd.DataFrame(rows), [2021], k_best=5).set_index("pid")
    assert s2.loc["F", "n_s2"] == 2 and abs(s2.loc["F", "tex_s2"] - 0.12) < 1e-9  # 夏秋图不用；12 月算到 2021 季


def test_load_visual_merges_rounds(tmp_path):
    import pandas as pd
    from rs_verify.fusion import load_visual
    a = tmp_path / "r1.csv"; b = tmp_path / "r2.csv"
    pd.DataFrame({"pid": ["P1", "P2"], "year": [2020, 2020], "visual": ["yes", "no"]}).to_csv(a, index=False)
    pd.DataFrame({"pid": ["P2", "P3"], "year": [2020, 2021], "visual": ["yes", "no"]}).to_csv(b, index=False)
    v = load_visual([a, b]).set_index("pid")
    assert len(v) == 3 and v.loc["P2", "visual"] == "yes"


def test_persist_aggregation_suppresses_one_off_texture():
    from rs_verify import config as C
    # 5 张好图：真筏架每张都有纹理；海面只有 2 张偶然出现纹理（海浪）
    rows = []
    for pid, texs in [("F", [0.10, 0.11, 0.12, 0.10, 0.11]), ("W", [0.12, 0.11, 0.02, 0.02, 0.02])]:
        for i, t in enumerate(texs):
            rows.append({"pid": pid, "date": f"2021-0{i + 1}-10", "tex": t, "clear": 1.0, "b2": 0.03 + i * 0.001,
                         "n_pix": 2500, "area_m2": 250000})
    med = s2_annual(pd.DataFrame(rows), [2021], agg="median").set_index("pid")["tex_s2"]
    per = s2_annual(pd.DataFrame(rows), [2021], agg="persist").set_index("pid")["tex_s2"]
    assert per["F"] >= 0.10 and per["W"] <= 0.03        # 持续性：海浪被压下去，筏架不受影响
    assert med["W"] < med["F"]
    assert C.S2_PERSIST_Q == 0.25


def test_idle_rule_two_seasons():
    from rs_verify.fusion import apply_idle_rule
    f = pd.DataFrame({"pid": ["A", "A", "A", "B", "B"], "year": [2020, 2021, 2022, 2021, 2022],
                      "decision": ["present", "absent", "absent", "absent", "present"],
                      "present": [1.0, 0.0, 0.0, 0.0, 1.0], "status": "ok", "evidence": ""})
    one = apply_idle_rule(f, 1)
    assert (one["decision"] == "absent").sum() == 3
    two = apply_idle_rule(f, 2).set_index(["pid", "year"])["decision"]
    assert two[("A", 2021)] == "review"      # 第一季没看到 → 待复核
    assert two[("A", 2022)] == "absent"      # 连续第二季没看到 → 进名单
    assert two[("B", 2021)] == "review"      # 没有上一季数据 → 待复核
