import numpy as np
import pandas as pd

from rs_verify import trace_sim
from rs_verify.annual import full_years, season_year, select_window


def test_window_helpers():
    sep = pd.DataFrame({"month": range(1, 13), "auc": [0.9, 0.95, 0.97, 0.96, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.85]})
    assert select_window(sep) == [2, 3, 4]
    d = pd.Series(pd.to_datetime(["2020-12-15", "2021-01-10", "2021-02-01"]))
    assert list(season_year(d, [12, 1, 2])) == [2021, 2021, 2021]
    assert full_years([12, 1, 2]) == list(range(2020, 2026))


def _world():
    # 4 个区内养殖地块、2 个区外养殖地块、2 个区内空海面
    polys = pd.DataFrame({
        "pid": ["A", "B", "C", "D", "E", "F", "S1", "S2"],
        "ref_pid": ["A", "B", "C", "D", "E", "F", "A", "B"],
        "kind": ["farm"] * 6 + ["shifted"] * 2,
        "open_sea": [False] * 6 + [True] * 2,
        "in_zone": [True, True, True, True, False, False, True, True],
        "area_ha": [25.0] * 8, "lon": 0.0, "lat": 0.0, "split": "test",
    })
    rows = []
    for pid in polys["pid"]:
        farm = not pid.startswith("S")
        for y in [2021, 2022]:
            rows.append({"pid": pid, "year": y, "status": "ok", "present": 1.0 if farm else 0.0,
                         "cover_sat": 0.9 if farm else 0.0, "cover_ref": 0.9 if farm else np.nan})
    return polys, pd.DataFrame(rows)


def test_layers_stop_the_right_scenarios():
    polys, comp = _world()
    rng = np.random.default_rng(0)
    reg = trace_sim.build_registry(polys, rng, idle_share=0.5)
    b = trace_sim.make_cases(reg, comp, [2021, 2022], rng, 160.0)
    v = trace_sim.verify_batches(b, reg, comp, tau=1.05, y_t_ha=160.0)
    first = v[v["layer"] != "passed"].sort_values("seq").groupby("case")["layer"].first()
    cases = v.groupby("case")["scenario"].first().to_frame().assign(caught=lambda d: d.index.map(first))
    assert (cases[cases["scenario"] == "S1_outside"]["caught"] == "registry").all()
    assert (cases[cases["scenario"] == "S3_neighbour"]["caught"] == "registry").all()
    assert (cases[cases["scenario"] == "S2_idle"]["caught"] == "presence").all()
    assert cases[cases["scenario"] == "honest"]["caught"].isna().all()
    launder3 = v[(v["scenario"] == "S4_launder") & (v["factor"] == 3.0)]
    assert (launder3.groupby("case")["layer"].apply(lambda s: (s == "quantity").any())).all()


def test_tradeoff_table_columns():
    import pandas as pd
    from rs_verify.tradeoff import to_cn
    df = pd.DataFrame([{"q_absent": 0.01, "p_quantity": 0.005, "honest_flagged": 0.01, "honest_review": 0.02,
                        "S2_idle_flagged": 0.9, "S4_launder_x2_flagged": 0.5}])
    cols = list(to_cn(df).columns)
    assert "诚实户_被判可疑(冤枉率)" in cols and "有证不养_检出率" in cols and "洗货2倍_检出率" in cols
