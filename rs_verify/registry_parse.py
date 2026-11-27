"""把政府公开的养殖证 / 海域使用权申请公示（PDF 提取的文字、Excel 导出的 CSV）解析成登记地块多边形。

数据来源：大连金普新区政府网站"拟受理用海申请及水域滩涂养殖证发放的公示""发放水域滩涂养殖证的公示"等
（Codex 下载、pdfplumber 提取文字，见 data/registry/dalian_jinpu/raw/index.csv）。

为什么不直接按表格列解析：PDF 版式提取出来的文字里，一宗地的项目名、面积、养殖方式等单元格
会和坐标行交错（一个单元格被拆到好几行，夹在第 3、4 个界址点之间）。这里按"界址点"来切分：
  1. 每一行找出所有界址点"序号, 纬度度分秒, 经度度分秒"（序号可带 A，如 A1）；
  2. 序号回到 1（或变小）就是新的一宗地；
  3. 夹在这宗地第一个和最后一个界址点之间的其余文字归这宗地，两宗地之间的文字归下一宗；
  4. 从这些文字里取：申请面积（不带"公顷"后缀的小数，取最后一个）、海域管理号、养殖方式、
     用海方式（海面/海底）、申请类型（设立/续期/变更/转让）、所属街道。
不提取申请人姓名（隐私；论文里也只用匿名编号）。

自检：用坐标算出的面积和公示面积比，相对误差 ≤ 5% 记为 ok。一宗地的界址点编号若中途重新从 1 开始
（多块组成一宗），后面那块没有自己的面积，就并进前一块，合并后面积对得上才算 ok。

坐标按 CGCS2000 给出，与 WGS84 相差在厘米级，直接当 WGS84 用。

用法：
  python -m rs_verify.registry_parse --raw data/registry/dalian_jinpu/raw \
      --out data/registry/dalian_jinpu/registry
输出：registry_parcels.geojson（去重后的全部地块）、registry_parcels.csv（属性表）、
      registry_items.csv（去重前逐条记录，含来源文件）、registry_qc.md（自检报告）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from . import geo

LAT_RANGE = (30.0, 45.0)
LON_RANGE = (115.0, 130.0)
AREA_TOL = 0.05

_DMS = r"(\d{2,3})\s*°\s*(\d{1,2})\s*′\s*(\d{1,2}(?:\.\d+)?)\s*″?"
POINT_RE = re.compile(r"(?:(?<![\d.])A?(\d{1,3})\s*[,、]?\s*)?" + _DMS + r"\s*,?\s*" + _DMS)
AREA_RE = re.compile(r"(?<![\d.A-Za-z])(\d{1,4}\.\d{1,6})(?![\d.]|\s*公顷|\s*hm)")
MGMT_RE = re.compile(r"\d{4}D\d{11}|国海证号?\s*\d{6,}|\b0\d{8}\b")
STREET_TOKEN_RE = re.compile(r"(?:^|[\s,])([一-龥]{2,4})街道(?=$|[\s,])")
STREET_IN_RE = re.compile(r"在([一-龥]{2,4}?)街道")
CULTURE_KEYS = ["筏吊式", "筏式", "浮筏", "吊笼", "网箱", "延绳", "滩涂底播", "底播", "池塘", "围海"]
APP_KEYS = ["设立", "续期", "变更", "转让", "更正"]
SKIP_LINE = re.compile(r"^\s*(序号|编号|第\s*\d+\s*页|附件\s*$)")


def normalize(text: str) -> str:
    """统一引号、撇号、逗号写法；Excel 导出的 "" 还原成 "。"""
    t = text.replace('""', '"')
    for a in ("”", "“", '"', "''", "＂"):
        t = t.replace(a, "″")
    for a in ("’", "‘", "'", "´"):
        t = t.replace(a, "′")
    t = t.replace("，", ",").replace("：", ":").replace("º", "°").replace("˚", "°")
    t = t.replace("″″", "″")
    return t


def dms(d, m, s) -> float:
    return float(d) + float(m) / 60.0 + float(s) / 3600.0


def scan_points(text: str):
    """逐行找界址点；返回 points=[(line, idx, lon, lat)] 和 residual=[每行去掉坐标后的文字]。"""
    points, residual = [], []
    for ln, line in enumerate(normalize(text).splitlines()):
        rest, last = [], 0
        for m in POINT_RE.finditer(line):
            lat = dms(m.group(2), m.group(3), m.group(4))
            lon = dms(m.group(5), m.group(6), m.group(7))
            if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1]):
                continue
            idx = int(m.group(1)) if m.group(1) else None
            points.append((ln, idx, lon, lat))
            rest.append(line[last:m.start()])
            last = m.end()
        rest.append(line[last:])
        residual.append("" if SKIP_LINE.match(line) else " ".join(rest))
    return points, residual


def segment(points) -> list[list[tuple]]:
    """序号回到 1 或变小 → 新的一宗。"""
    polys, cur, prev = [], [], None
    for p in points:
        idx = p[1]
        if cur and idx is not None and (idx == 1 or (prev is not None and idx <= prev)):
            polys.append(cur)
            cur = []
        cur.append(p)
        prev = idx if idx is not None else prev
    if cur:
        polys.append(cur)
    return [c for c in polys if len(c) >= 3]


def assign_text(polys, residual) -> list[str]:
    """每行文字归属：行里有界址点 → 该行第一个点所属的那宗；在某宗首末点行之间 → 该宗；
    两宗之间的空隙 → 下一宗（续表的抬头、跨行的项目名通常在下一宗的坐标之前）。"""
    owner = {}
    for k, pl in enumerate(polys):
        for p in pl:
            owner.setdefault(p[0], k)
    spans = [(pl[0][0], pl[-1][0]) for pl in polys]
    texts = [[] for _ in polys]
    k = 0
    for ln, txt in enumerate(residual):
        if not txt.strip():
            continue
        if ln in owner:
            texts[owner[ln]].append(txt)
            continue
        while k < len(spans) and spans[k][1] < ln:
            k += 1
        if k < len(spans):
            texts[k].append(txt)       # 在第 k 宗之内或之前的空隙
        elif texts:
            texts[-1].append(txt)      # 文件末尾的备注
    return [" ".join(t) for t in texts]


def parse_meta(text: str) -> dict:
    areas = AREA_RE.findall(text)
    culture = [k for k in CULTURE_KEYS if k in text]
    if "滩涂底播" in culture and "底播" in culture:
        culture.remove("底播")
    if "筏吊式" in culture and "筏式" in culture:
        culture.remove("筏式")
    if "海面海底" in text:
        mode = "海面海底"
    elif re.search(r"海面", text):
        mode = "海面"
    elif re.search(r"海底", text):
        mode = "海底"
    else:
        mode = ""
    streets = Counter(t.split("在")[-1] for t in STREET_TOKEN_RE.findall(text) if t != "所属")
    if not streets:
        streets = Counter(STREET_IN_RE.findall(text))
    return {
        "area_decl_ha": float(areas[-1]) if areas else None,
        "mgmt_no": ";".join(dict.fromkeys(re.sub(r"\s+", "", m) for m in MGMT_RE.findall(text))),
        "culture": "、".join(culture),
        "sea_mode": mode,
        "app_type": "、".join(k for k in APP_KEYS if k in text),
        "street": (streets.most_common(1)[0][0] + "街道") if streets else "",
    }


def ring_of(pl) -> list[list[float]]:
    ring = [[p[2], p[3]] for p in pl]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def self_intersects(ring) -> bool:
    n = len(ring) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[i + 1], ring[j], ring[j + 1]):
                return True
    return False


def rel_err(calc, decl):
    return abs(calc - decl) / decl if decl else None


def visible_from_space(culture: str, mode: str) -> bool:
    """卫星看得到的水面设施：筏、吊笼、网箱；纯海底（底播）和池塘、围海不算。"""
    if mode == "海底":
        return False
    return any(k in culture for k in ("筏", "吊笼", "网箱", "延绳"))


def parse_text(text: str, source: str = "") -> list[dict]:
    points, residual = scan_points(text)
    polys = segment(points)
    texts = assign_text(polys, residual)
    items = []
    for k, (pl, txt) in enumerate(zip(polys, texts)):
        meta = parse_meta(txt)
        ring = ring_of(pl)
        items.append({"source": source, "order": k + 1, "rings": [ring], "n_pts": len(pl), **meta,
                      "area_calc_ha": geo.ring_area_m2(ring) / 1e4, "self_intersect": self_intersects(ring)})
    return merge_multipart(items)


def try_split(it: dict) -> None:
    """一宗地由两块组成但界址点连续编号（1–4 一块、5–8 另一块）：在相邻点距离最大的地方断开，
    两块面积之和对得上公示面积才采用。"""
    if it["area_decl_ha"] is None or len(it["rings"]) != 1:
        return
    e = rel_err(it["area_calc_ha"], it["area_decl_ha"])
    if e is None or (e <= AREA_TOL and not it["self_intersect"]):
        return
    pts = it["rings"][0][:-1]
    n = len(pts)
    if n < 6:
        return
    gaps = sorted(range(2, n - 3), key=lambda i: -((pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2))
    for i in gaps[:3]:
        a, b = pts[:i + 1], pts[i + 1:]
        ra, rb = a + [a[0]], b + [b[0]]
        tot = (geo.ring_area_m2(ra) + geo.ring_area_m2(rb)) / 1e4
        if rel_err(tot, it["area_decl_ha"]) <= AREA_TOL and not (self_intersects(ra) or self_intersects(rb)):
            it["rings"], it["area_calc_ha"], it["self_intersect"] = [ra, rb], tot, False
            return


def merge_multipart(items: list[dict]) -> list[dict]:
    """没有自己面积的一块，若与前一宗合并后面积对得上，就并进去（多块组成一宗）。"""
    out = []
    for it in items:
        prev = out[-1] if out else None
        if it["area_decl_ha"] is None and prev is not None and prev["area_decl_ha"]:
            e_alone = rel_err(prev["area_calc_ha"], prev["area_decl_ha"])
            merged = prev["area_calc_ha"] + it["area_calc_ha"]
            if e_alone is not None and e_alone > AREA_TOL and rel_err(merged, prev["area_decl_ha"]) <= AREA_TOL:
                prev["rings"] += it["rings"]
                prev["n_pts"] += it["n_pts"]
                prev["area_calc_ha"] = merged
                prev["self_intersect"] = prev["self_intersect"] or it["self_intersect"]
                for f in ("culture", "sea_mode", "app_type", "street"):
                    prev[f] = prev[f] or it[f]
                continue
        out.append(it)
    for it in out:
        try_split(it)
        e = rel_err(it["area_calc_ha"], it["area_decl_ha"]) if it["area_decl_ha"] else None
        it["area_rel_err"] = e
        it["qc"] = ("no_area" if e is None else "ok" if e <= AREA_TOL else "area_mismatch") + \
                   (";self_intersect" if it["self_intersect"] else "")
        it["raft_visible"] = visible_from_space(it["culture"], it["sea_mode"])
    return out


def geometry_of(it: dict) -> dict:
    if len(it["rings"]) == 1:
        return {"type": "Polygon", "coordinates": [it["rings"][0]]}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in it["rings"]]}


def geom_key(it: dict) -> str:
    pts = sorted({(round(x, 6), round(y, 6)) for r in it["rings"] for x, y in r})
    return hashlib.sha1(repr(pts).encode()).hexdigest()[:16]


def read_index(raw: Path) -> dict:
    p = raw / "index.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["text_file"]: r for r in csv.DictReader(f) if r.get("text_file")}


def run(raw: Path) -> tuple[list[dict], list[dict]]:
    index = read_index(raw)
    items = []
    for path in sorted(list(raw.glob("*.txt")) + list(raw.glob("*.csv"))):
        if path.name == "index.csv":
            continue
        meta = index.get(path.name, {})
        date = meta.get("publish_date") or path.name[:10]
        for it in parse_text(path.read_text(encoding="utf-8-sig", errors="replace"), path.name):
            it["notice_date"] = date
            it["notice_id"] = meta.get("notice_id", "")
            it["key"] = geom_key(it)
            items.append(it)
    # 去重：同一组界址点（续期、更正、重复公示）只保留最新一次公示，记下出现次数和最早日期
    by_key: dict[str, list[dict]] = {}
    for it in items:
        by_key.setdefault(it["key"], []).append(it)
    parcels = []
    for key, grp in by_key.items():
        grp = sorted(grp, key=lambda r: r["notice_date"])
        last = dict(grp[-1])
        last["first_notice"] = grp[0]["notice_date"]
        last["n_notices"] = len(grp)
        parcels.append(last)
    parcels.sort(key=lambda r: (r["first_notice"], r["source"], r["order"]))
    for i, p in enumerate(parcels):
        p["pid"] = f"DL{i + 1:05d}"
    return items, parcels


FIELDS = ["pid", "notice_date", "first_notice", "n_notices", "source", "order", "area_decl_ha", "area_calc_ha",
          "area_rel_err", "qc", "n_pts", "mgmt_no", "culture", "sea_mode", "app_type", "street", "raft_visible"]


def qc_report(items: list[dict], parcels: list[dict]) -> str:
    def cnt(rows, f):
        return Counter(r[f] for r in rows)
    lines = ["# 登记地块解析自检", "",
             f"- 逐条记录 {len(items)} 条（去重前），来自 {len({r['source'] for r in items})} 个文件",
             f"- 去重后地块 {len(parcels)} 块；其中卫星可见（筏/吊笼/网箱，非纯海底）"
             f" {sum(p['raft_visible'] for p in parcels)} 块",
             "", "## 面积自检（坐标算出 vs 公示面积，±5% 为 ok）", ""]
    for k, v in cnt(parcels, "qc").most_common():
        lines.append(f"- {k}: {v}")
    ok_vis = [p for p in parcels if p["raft_visible"] and p["qc"] == "ok"]
    lines += ["", f"可直接用于卫星筛查（卫星可见 且 面积自检 ok）：**{len(ok_vis)} 块**，"
              f"合计 {sum(p['area_calc_ha'] for p in ok_vis):.0f} 公顷", "", "## 按文件", "",
              "| 文件 | 记录数 | ok | 面积不符 | 无面积 |", "|---|---|---|---|---|"]
    for src in sorted({r["source"] for r in items}):
        rs = [r for r in items if r["source"] == src]
        c = cnt(rs, "qc")
        lines.append(f"| {src} | {len(rs)} | {sum(v for k, v in c.items() if k.startswith('ok'))} | "
                     f"{sum(v for k, v in c.items() if k.startswith('area_mismatch'))} | "
                     f"{sum(v for k, v in c.items() if k.startswith('no_area'))} |")
    lines += ["", "## 养殖方式 / 用海方式 / 街道（去重后）", ""]
    for f in ("culture", "sea_mode", "street", "app_type"):
        lines.append(f"- {f}: " + "；".join(f"{k or '（空）'} {v}" for k, v in cnt(parcels, f).most_common(12)))
    bad = sorted((p for p in parcels if p["qc"] != "ok"), key=lambda p: -(p["area_rel_err"] or 9))[:30]
    if bad:
        lines += ["", "## 未通过自检的（前 30 条，人工核对用）", "",
                  "| pid | 文件 | 第几宗 | 公示面积 | 算出面积 | 点数 | qc |", "|---|---|---|---|---|---|---|"]
        for p in bad:
            lines.append(f"| {p['pid']} | {p['source']} | {p['order']} | {p['area_decl_ha']} | "
                         f"{p['area_calc_ha']:.4f} | {p['n_pts']} | {p['qc']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="解析养殖证 / 海域使用权公示 → 登记地块多边形")
    ap.add_argument("--raw", default="data/registry/dalian_jinpu/raw")
    ap.add_argument("--out", default="data/registry/dalian_jinpu/registry")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    items, parcels = run(raw)
    out.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "properties": {k: p.get(k) for k in FIELDS}, "geometry": geometry_of(p)}
             for p in parcels]
    geo.save_geojson(geo.feature_collection(feats), out / "registry_parcels.geojson")
    for name, rows in (("registry_parcels.csv", parcels), ("registry_items.csv", items)):
        with open(out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS + (["key"] if name == "registry_items.csv" else []),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    rep = qc_report(items, parcels)
    (out / "registry_qc.md").write_text(rep, encoding="utf-8")
    print(rep)
    print("输出目录：", out)


if __name__ == "__main__":
    main()
