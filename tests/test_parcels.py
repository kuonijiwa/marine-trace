from rs_verify import config as C
from rs_verify import geo
from rs_verify.claims import build_claims
from rs_verify.make_parcels import make_parcels

BBOX = C.STUDY_AREAS["ailun"]["bbox"]


def rect(lon0, lat0, w_m, h_m):
    dlon, dlat = geo._deg_per_m(lat0)
    x1, y1 = lon0 + w_m * dlon, lat0 + h_m * dlat
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [x1, lat0], [x1, y1], [lon0, y1], [lon0, lat0]]]}


def fc_of(*geoms):
    return geo.feature_collection([{"type": "Feature", "geometry": g, "properties": {}} for g in geoms])


def test_raster_mask_fraction():
    big = rect(122.55, 37.10, 3000, 2000)
    rm = geo.RasterMask(fc_of(big), BBOX, 20)
    assert abs(rm.mask.sum() * 400 - 6e6) / 6e6 < 0.03
    inside = rect(122.56, 37.11, 500, 500)
    assert rm.fraction(inside) > 0.99
    half = geo.shift_geometry(rect(122.55, 37.11, 500, 500), 250, 270)  # 一半在西边界外
    assert 0.4 < rm.fraction(half) < 0.6
    assert rm.fraction(geo.shift_geometry(inside, 5000, 0)) == 0.0
    # 研究区框外按"不是养殖区"处理
    assert rm.fraction(rect(121.0, 30.0, 500, 500)) == 0.0


def test_make_parcels_cover_filter():
    big = rect(122.55, 37.10, 3000, 2000)   # 3 km × 2 km → 约 6×4 = 24 个 500 m 格子（对不齐时边上的覆盖率低）
    parcels, _ = make_parcels(fc_of(big), BBOX, "raft", 500, 0.6)
    n = len(parcels["features"])
    assert 15 <= n <= 30
    for f in parcels["features"]:
        assert f["properties"]["cover"] >= 0.6
        assert abs(f["properties"]["area_m2"] - 250_000) / 250_000 < 0.03
        assert f["properties"]["product"] == "kelp"


def test_claims_with_raft_mask():
    big = rect(122.55, 37.10, 3000, 2000)
    parcels, rm = make_parcels(fc_of(big), BBOX, "raft", 500, 0.6)
    claims, geoms = build_claims(parcels, "ailun", max_refs=3, n_bearings=4, raft_mask=rm)
    loc = claims[claims["label"] == "fake_location"]
    # 挪 100 m 的假围栏大多还压在大筏区上；挪 5 km 的完全在外面
    assert loc[loc["shift_m"] == 100]["overlap_ref"].mean() > 0.5
    assert loc[loc["shift_m"] == 5000]["overlap_ref"].max() == 0.0
    # 所有假围栏的中心都在研究区框内
    for f in geoms["features"]:
        cx, cy = geo.centroid(f["geometry"]["coordinates"][0])
        assert BBOX[0] <= cx <= BBOX[2] and BBOX[1] <= cy <= BBOX[3]


def test_quantity_check_flags_inflation():
    import numpy as np
    import pandas as pd
    from rs_verify.quantity import run as qrun

    rows, claims = [], []
    dates = pd.date_range("2020-11-01", "2021-05-31", freq="12D")
    rng = np.random.default_rng(0)
    cover = {}
    for i in range(40):
        pid = f"R{i:04d}"
        cover[pid] = rng.uniform(0.6, 1.0)
        for d in dates:
            rows.append({"pid": pid, "date": d, "lin_ratio": 1 + 0.8 * cover[pid] * rng.uniform(0.9, 1.1),
                         "bg_db": -20.0, "n_pix": 2500, "area_m2": 250000})
        q = rng.uniform(0.5, 0.95)
        base = dict(pid=pid, ref_pid=pid, start="2020-11-01", end="2021-05-31", overlap_ref=1.0)
        claims.append(dict(base, claim_id=f"T{i}", label="true", quantity_rel=q, inflation=1.0))
        claims.append(dict(base, claim_id=f"Q{i}", label="fake_quantity", quantity_rel=q * 2, inflation=2.0))
    act = __import__("rs_verify.verify", fromlist=["prepare_activity"]).prepare_activity(pd.DataFrame(rows))
    cl, m = qrun(pd.DataFrame(claims), act, cover)
    assert m["test_false_alarm_true"] <= 0.2
    assert m["test_detection_by_inflation"][2.0] >= 0.9
    assert m["test_cover_sat_vs_ref_pearson_r"] > 0.8
