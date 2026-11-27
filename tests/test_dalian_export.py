"""大连导出准备：只挑卫星可见且自检 ok 的地块、补 area_m2；时间没到不提交。"""
import datetime as dt

from rs_verify import dalian_export as D
from rs_verify import geo


def _sq(pid, vis, qc, ha):
    ring = [[122.0, 39.0], [122.001, 39.0], [122.001, 39.001], [122.0, 39.001], [122.0, 39.0]]
    return {"type": "Feature", "properties": {"pid": pid, "raft_visible": vis, "qc": qc, "area_calc_ha": ha},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_prepare_screen_filters_and_adds_area(tmp_path):
    reg = tmp_path / "reg.geojson"
    geo.save_geojson(geo.feature_collection([_sq("DL00001", True, "ok", 1.5), _sq("DL00002", False, "ok", 2.0),
                                             _sq("DL00003", True, "area_mismatch", 3.0),
                                             _sq("DL00004", "True", "ok", 0.25)]), reg)
    out = D.prepare_screen(reg, tmp_path / "screen.geojson")
    pids = [f["properties"]["pid"] for f in out["features"]]
    assert pids == ["DL00001", "DL00004"]
    assert out["features"][0]["properties"]["area_m2"] == 15000.0
    assert (tmp_path / "screen.geojson").exists()


def test_not_before_guard_and_years():
    tz = dt.timezone(dt.timedelta(hours=9))
    assert D.too_early(D.NOT_BEFORE, dt.datetime(2026, 12, 1, 16, 59, tzinfo=tz))
    assert not D.too_early(D.NOT_BEFORE, dt.datetime(2026, 12, 1, 17, 10, tzinfo=tz))
    ys = D.years("2019-01-01", "2025-12-31")
    assert len(ys) == 7 and ys[0] == ("2019-01-01", "2020-01-01", "_2019")
