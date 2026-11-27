"""端到端冒烟测试：合成一个小世界，跑通 fusion.build（含 2 季规则、持续性合成）和 tradeoff.run。"""
import numpy as np
import pandas as pd

from rs_verify import config as C
from rs_verify import fusion, tradeoff


def _world(seed=0):
    rng = np.random.default_rng(seed)
    claims, feats, act, s2 = [], [], [], []
    cover = {}
    n_farm = 24
    sar_dates = pd.date_range("2019-01-05", "2025-12-25", freq="24D")
    s2_dates = [d for d in pd.date_range("2019-01-03", "2025-12-28", freq="10D") if d.month in C.S2_MONTHS]

    def square(lon, lat, pid, area=250000.0):
        s = 0.0045
        ring = [[lon, lat], [lon + s, lat], [lon + s, lat + s], [lon, lat + s], [lon, lat]]
        return {"type": "Feature", "properties": {"pid": pid, "area_m2": area},
                "geometry": {"type": "Polygon", "coordinates": [ring]}}

    for i in range(n_farm):
        ref = f"R{i:04d}"
        lon, lat = 122.45 + 0.01 * (i % 6), 37.15 + 0.03 * (i // 6)     # 都在示范区内
        cover[ref] = float(rng.uniform(0.5, 0.9))
        polys = [(ref, "true", 0.0, 1.0, True), (f"{ref}_S500_B090", "fake_location", 500.0, 0.0, False),
                 (f"{ref}_S5000_B090", "fake_location", 5000.0, 0.0, False)]
        for pid, label, shift, ov, farm in polys:
            claims.append({"pid": pid, "ref_pid": ref, "label": label, "shift_m": shift, "overlap_ref": ov})
            feats.append(square(lon + shift / 90000.0, lat, pid))
            for d in sar_dates:
                bg_db = float(rng.uniform(-25, -20))
                ex = (8.0 if farm else 0.05) * rng.uniform(0.6, 1.4)
                act.append({"pid": pid, "date": d.strftime("%Y-%m-%d"), "lin_ratio": 1 + ex, "bg_db": bg_db,
                            "bright_frac": 0.5 if farm else 0.01, "n_pix": 2500, "area_m2": 250000.0})
            for d in s2_dates:
                tex = (0.10 if farm else 0.02) + rng.normal(0, 0.01)
                if not farm and rng.random() < 0.2:          # 海面偶发纹理
                    tex = 0.11
                s2.append({"pid": pid, "date": d.strftime("%Y-%m-%d"), "tex": tex, "clear": 1.0,
                           "b2": float(rng.uniform(0.02, 0.06)), "n_pix": 2500, "area_m2": 250000.0, "tile": "T"})
    return (pd.DataFrame(claims), pd.DataFrame(act), cover, pd.DataFrame(s2),
            {"type": "FeatureCollection", "features": feats})


def test_fusion_and_tradeoff_end_to_end(tmp_path):
    claims, act, cover, s2, gj = _world()
    old = C.ACTIVITY_METRIC
    C.ACTIVITY_METRIC = "abs_excess"
    try:
        visual = pd.DataFrame({"pid": ["R0000", "R0000_S500_B090"], "year": [2021, 2021],
                               "date": ["2021-01-10", "2021-01-10"], "visual": ["yes", "no"]})
        for agg in ("median", "persist"):
            f, m, am = fusion.build(claims, act, cover, s2, visual, s2_agg=agg, idle_seasons=2)
            assert m["s2_agg"] == agg and m["idle_seasons"] == 2
            assert m["test_gap"]["n"] > 0
            assert m["test_farm"]["absent"] <= 0.05
            assert "list" in m["visual"] and "screening_list" in m and "visual_two_level" in m
            fusion.plot(f, m["thresholds"], m, tmp_path / f"fusion_{agg}.png")
        df, presets = tradeoff.run(claims, act, cover, s2, gj, "ailun", q_grid=[0.02], p_grid=[0.01],
                                   seasons_grid=(1, 2), visual=visual, s2_agg="persist")
        assert set(df["idle_seasons"]) == {1, 2} and len(presets) == 6
        # 2 季规则冤枉不会多于 1 季规则
        h = df.groupby("idle_seasons")["farm_years_absent"].max()
        assert h[2] <= h[1]
        tradeoff.plot(df, presets, tmp_path / "tradeoff.png")
        out = tradeoff.to_cn(df)
        assert "旋钮C_荒芜认定季数" in out.columns and "区内空地块年_判不在养" in out.columns
    finally:
        C.ACTIVITY_METRIC = old


def test_paper_figs_from_tradeoff_table(tmp_path):
    from rs_verify import paper_figs
    rows = []
    for sn in (1, 2):
        for q in (0.01, 0.05, 0.2):
            for p in (0.005, 0.02, 0.1):
                rows.append({"idle_seasons": sn, "q_absent": q, "p_quantity": p, "honest_flagged": q / 10 + p / 5,
                             "S2_idle_flagged": q * 3 / sn, "S4_launder_x1.5_flagged": p * 3,
                             "S4_launder_x2_flagged": p * 6, "S4_launder_x3_flagged": min(1, p * 10),
                             "visual_farmed_listed": q / 5 / sn, "visual_empty_listed": q * 3 / sn,
                             "visual_list_truly_empty": 0.95})
    path = tmp_path / "t.csv"
    tradeoff.to_cn(pd.DataFrame(rows)).to_csv(path, index=False, encoding="utf-8-sig")
    paper_figs.fig_tradeoff(paper_figs.load_table(path), tmp_path / "fig_tradeoff")
    paper_figs.fig_workflow(tmp_path / "fig_workflow")
    assert (tmp_path / "fig_tradeoff.png").exists() and (tmp_path / "fig_workflow.pdf").exists()


def test_uncertainty_and_case_study(tmp_path):
    from rs_verify import case_study, trace_sim, uncertainty
    claims, act, cover, s2, gj = _world(seed=1)
    old = C.ACTIVITY_METRIC
    C.ACTIVITY_METRIC = "abs_excess"
    try:
        visual = pd.DataFrame({"pid": ["R0000", "R0001", "R0000_S500_B090", "R0001_S5000_B090"],
                               "year": [2021, 2022, 2021, 2022], "date": ["2021-01-10"] * 2 + ["2022-01-10"] * 2,
                               "visual": ["yes", "yes", "no", "no"]})
        f, m, am = fusion.build(claims, act, cover, s2, visual, s2_agg="persist", idle_seasons=1)
        path = tmp_path / "comp.csv"
        f.to_csv(path, index=False)
        comp = pd.read_csv(path)
        _, _, _, batches = trace_sim.run_with_comp(f, am, claims, cover, gj, "ailun")
        t = uncertainty.run(comp, visual, m["thresholds"], batches, n=200)
        row = t.set_index("metric")
        assert row.loc["farmed_listed_1season", "estimate"] == 0.0
        assert row.loc["list_purity_2season", "denominator"] >= 0
        assert row.loc["honest_flagged", "denominator"] > 0
        est, lo, hi = row.loc["open_sea_years_listed", ["estimate", "ci_low", "ci_high"]]
        assert lo <= est <= hi
        assert "| 指标 |" in uncertainty.to_markdown(t)
        # 0% / 100% 时自助法区间退化，改报精确二项区间（与 scipy beta 分位一致）
        assert abs(uncertainty.clopper_pearson(0, 114)[1] - 0.0318) < 5e-4
        assert abs(uncertainty.clopper_pearson(16, 16)[0] - 0.7941) < 5e-4
        assert abs(uncertainty.clopper_pearson(44, 112)[0] - 0.3019) < 5e-4
        pid = "R0002_S5000_B090"
        case = case_study.annual_table(comp, pid)
        assert len(case) == len(am["years"]) and "level" in case
        sar, s2s = case_study.scene_series(act, s2, pid)
        case_study.figure(case, m["thresholds"], tmp_path / "fig_case", sar, s2s,
                          case_study.annual_table(comp, "R0002"), *case_study.scene_series(act, s2, "R0002"),
                          pid=pid, cmp_pid="R0002")
        assert (tmp_path / "fig_case.png").exists()
    finally:
        C.ACTIVITY_METRIC = old
