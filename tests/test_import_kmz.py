import io
import zipfile

import numpy as np

from rs_verify import geo
from rs_verify.import_reference_kmz import rasters_to_polygons, scan_kmz

BBOX = [122.42, 36.98, 122.72, 37.34]
NS = 'xmlns="http://www.opengis.net/kml/2.2"'


def _square_coords(lon0, lat0, size_deg):
    pts = [(lon0, lat0), (lon0 + size_deg, lat0), (lon0 + size_deg, lat0 + size_deg), (lon0, lat0 + size_deg), (lon0, lat0)]
    return " ".join(f"{x},{y},0" for x, y in pts)


def test_vector_kmz(tmp_path):
    kml = f"""<?xml version="1.0"?><kml {NS}><Document>
    <Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{_square_coords(122.5, 37.1, 0.01)}</coordinates>
    </LinearRing></outerBoundaryIs></Polygon></Placemark>
    <Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{_square_coords(120.0, 30.0, 0.01)}</coordinates>
    </LinearRing></outerBoundaryIs></Polygon></Placemark>
    </Document></kml>"""
    p = tmp_path / "v.kmz"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("doc.kml", kml)
    log = []
    polys, overlays, _ = scan_kmz(p, BBOX, log)
    assert len(polys) == 1 and not overlays  # 框外的那个被排除
    assert abs(geo.geometry_area_m2(polys[0]) - 1.11e6 * 0.887) / 1e6 < 0.05


def _png(mask: np.ndarray) -> bytes:
    from PIL import Image

    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = [255, 0, 0, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def test_raster_kmz_tiles_and_pyramid(tmp_path):
    # 两张相邻的细图片（各 0.1°×0.1°，200×200 像元），一块"养殖区"横跨两张图片的接缝
    res = 0.1 / 200
    m1 = np.zeros((200, 200), bool)
    m2 = np.zeros((200, 200), bool)
    m1[80:120, 150:200] = True  # 左图右侧
    m2[80:120, 0:30] = True     # 右图左侧（同一块）
    m2[10:30, 100:140] = True   # 右图里另一块
    coarse = np.zeros((10, 10), bool)
    coarse[:] = True            # 粗一级（应被忽略）
    tiles = [("files/t1.png", m1, (122.50, 37.10, 122.60, 37.20)),
             ("files/t2.png", m2, (122.60, 37.10, 122.70, 37.20)),
             ("files/c.png", coarse, (122.40, 37.00, 122.80, 37.40))]
    overlays_xml = "".join(
        f"<GroundOverlay><Icon><href>{h}</href></Icon><LatLonBox><north>{n}</north><south>{s}</south>"
        f"<east>{e}</east><west>{w}</west></LatLonBox></GroundOverlay>"
        for h, _, (w, s, e, n) in tiles)
    p = tmp_path / "r.kmz"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("doc.kml", f'<?xml version="1.0"?><kml {NS}><Document>{overlays_xml}</Document></kml>')
        for h, m, _ in tiles:
            z.writestr(h, _png(m))
    log = []
    polys, overlays, zf = scan_kmz(p, BBOX, log)
    assert not polys and len(overlays) == 3
    out = rasters_to_polygons(zf, overlays, BBOX, log)
    areas = sorted(geo.geometry_area_m2(g) for g in out)
    assert len(out) == 2  # 跨接缝的那块被拼成一整块
    px = (res * 111_320) * (res * 111_320 * np.cos(np.radians(37.15)))
    assert abs(areas[1] - 80 * 40 * px) / (80 * 40 * px) < 0.1
    assert abs(areas[0] - 40 * 20 * px) / (40 * 20 * px) < 0.15
    cx, cy = geo.centroid(max(out, key=geo.geometry_area_m2)["coordinates"][0])
    assert abs(cx - 122.60) < 0.01 and abs(cy - (37.20 - 100 * res)) < 0.01
