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
            assert "list" in m["visual"]
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
