import math

import numpy as np
import pandas as pd

from rs_verify import geo
from rs_verify.claims import season_windows, shift_window
from rs_verify.verify import prepare_activity, verify_claims


def haversine_m(lon1, lat1, lon2, lat2):
    R = geo.EARTH_RADIUS_M
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def square(lon0, lat0, size_m):
    dlon, dlat = geo._deg_per_m(lat0)
    x1, y1 = lon0 + size_m * dlon, lat0 + size_m * dlat
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [x1, lat0], [x1, y1], [lon0, y1], [lon0, lat0]]]}


def test_shift_distance_and_area():
    g = square(122.6, 37.2, 500)
    assert abs(geo.geometry_area_m2(g) - 250_000) / 250_000 < 0.005
    for bearing in (0, 45, 90, 200):
        g2 = geo.shift_geometry(g, 1000, bearing)
        c1, c2 = geo.centroid(g["coordinates"][0]), geo.centroid(g2["coordinates"][0])
        assert abs(haversine_m(*c1, *c2) - 1000) < 5


def test_overlap_fraction():
    g = square(122.6, 37.2, 500)
    assert geo.overlap_fraction(g, [g]) > 0.99
    half = geo.shift_geometry(g, 250, 90)
    assert 0.4 < geo.overlap_fraction(half, [g]) < 0.6
    far = geo.shift_geometry(g, 5000, 0)
    assert geo.overlap_fraction(far, [g]) == 0.0


def test_season_windows_cross_year():
    w = season_windows("kelp", 2020, 2021)
    assert w[0] == ("2019-11-01", "2020-05-31")
    assert shift_window("2019-11-01", "2020-05-31", 6) == ("2020-05-01", "2020-11-30")


def _toy_activity():
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="12D")
    season = (dates.month >= 11) | (dates.month <= 5)
    rows = []
    for d, s in zip(dates, season):
        rows.append({"pid": "R", "date": d, "bright_frac": 0.5 if s else 0.03, "n_pix": 100, "area_m2": 10000})
        rows.append({"pid": "SEA", "date": d, "bright_frac": 0.01, "n_pix": 100, "area_m2": 10000})
    return prepare_activity(pd.DataFrame(rows), active_threshold=0.15)


def _claim(cid, pid, start, end, label, q=0.8):
    return {"claim_id": cid, "pid": pid, "start": start, "end": end, "label": label, "quantity_rel": q}


def test_verify_separates_true_and_fake():
    act = _toy_activity()
    claims = pd.DataFrame([
        _claim("t", "R", "2019-11-01", "2020-05-31", "true"),
        _claim("loc", "SEA", "2019-11-01", "2020-05-31", "fake_location"),
        _claim("time", "R", "2020-06-01", "2020-10-31", "fake_time"),
        _claim("qty", "R", "2019-11-01", "2020-05-31", "fake_quantity", q=2.0),
        _claim("short", "R", "2020-01-01", "2020-01-05", "true"),
    ])
    r = verify_claims(act, claims).set_index("claim_id")
    assert r.loc["t", "confidence"] > 0.9
    assert r.loc["loc", "confidence"] < 0.1
    assert r.loc["time", "confidence"] < 0.1
    assert r.loc["t", "qty_ratio"] <= 1.0 < r.loc["qty", "qty_ratio"]
    assert r.loc["short", "status"] == "insufficient"
    assert np.isnan(r.loc["short", "confidence"])


def test_excess_metric_and_fallback():
    from rs_verify.verify import activity_metric

    df = pd.DataFrame({"lin_ratio": [1.0, 2.0], "bright_frac": [0.1, 0.9]})
    assert list(activity_metric(df, "excess")) == [0.0, 1.0]
    assert list(activity_metric(df, "bright_frac")) == [0.1, 0.9]
    # 旧版导出没有 lin_ratio 列时退回 bright_frac
    assert list(activity_metric(df.drop(columns="lin_ratio"), "excess")) == [0.1, 0.9]


def test_verify_with_excess_metric():
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="12D")
    season = (dates.month >= 11) | (dates.month <= 5)
    rows = []
    for d, s in zip(dates, season):
        rows.append({"pid": "R", "date": d, "lin_ratio": 2.0 if s else 1.05, "n_pix": 100, "area_m2": 10000})
        rows.append({"pid": "SEA", "date": d, "lin_ratio": 1.0, "n_pix": 100, "area_m2": 10000})
    act = prepare_activity(pd.DataFrame(rows), active_threshold=0.3)
    claims = pd.DataFrame([
        _claim("t", "R", "2019-11-01", "2020-05-31", "true"),
        _claim("loc", "SEA", "2019-11-01", "2020-05-31", "fake_location"),
        _claim("time", "R", "2020-06-01", "2020-10-31", "fake_time"),
    ])
    r = verify_claims(act, claims).set_index("claim_id")
    assert r.loc["t", "confidence"] > 0.9
    assert r.loc["loc", "confidence"] < 0.1
    assert r.loc["time", "confidence"] < 0.1
