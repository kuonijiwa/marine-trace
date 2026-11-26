"""完整流程模拟：海域登记库 + 批次溯源 ID + 三层核验 + 五类造假。

数据：地块、卫星年度检查结果都是真实的（来自 annual.py）；海域登记库、养殖户、批次是模拟的
（真实登记数据不公开），论文里要说明。

登记库（模拟）：
  - 示范区内的真实养殖地块，随机分给养殖户 H001…（每户 1–3 块），每块一个海域使用权证号；
  - 示范区外的真实养殖地块，分给"区外养殖户" O001…（他们的证在示范区外）；
  - 一部分示范区内的空地块也登记给养殖户 = 有证不养（荒芜）的养殖证。位置由 C.IDLE_PLACEMENT 定：
      gap（默认）养殖区内或旁边、四周有别人筏架的空地块——现实里荒废的证就在养殖区里；
      open_sea    远处开阔海（原做法，作对照）。

每批货：批次号（溯源 ID）、养殖户、地块、年份、数量（吨）。养殖户一年的产量拆成 2–6 批卖。

三层核验（每批依次过，第一道不通过就停）：
  1. 登记层：地块在示范区内？证上的人是本人？
  2. 卫星位置层：该地块该年检查窗口内是否在养（无法核验也算不通过）？
  3. 卫星产量层：该地块该年累计声明量（含本批）是否超过 τ × 卫星产能？
     卫星产能 = 卫星覆盖率 × 面积 × 单产；τ 用标定组诚实养殖户定（约 5% 误报）。

场景：
  honest       诚实养殖户
  S1_outside   区外养殖户把自己区外的货说成示范区产品（声明自己的区外地块）
  S2_idle      有证不养：声明自己名下的空置海域
  S3_neighbour 冒用别人的地块（声明示范区内他人名下的地块）
  S4_launder   有地也在养，但多报（把外面的货洗进来），放大 r 倍
  S5_collusion 借别人账号出货：账号主人的正常批次 + 外人的货（额外 e 倍）

结果只在检验组地块上统计（标定组只用来定门限）。

输出（outputs/<area>_trace/）：
  registry.csv           模拟登记库（权证号、持证人、地块、是否在示范区）
  batches.csv            全部批次与核验结论（扫码原型用）
  trace_metrics.json     各场景检出率、拦截层、诚实误报率
  scenario_table.csv     场景 × 拦截层 表
  trace_layers.png       各场景被哪一层拦住（堆叠柱状图）+ 洗货检出率 vs 倍数

用法（参数同 annual.py）：
  python -m rs_verify.trace_sim --area ailun --claims data/claims/ailun_claims.csv \
      --claim-geoms data/claims/ailun_claim_geoms.geojson \
      --raft-reference data/reference/ailun_fu2021_reference.geojson \
      --activity data/activity/ailun_activity_*.csv --metric abs_excess
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import annual  # noqa: E402
from . import config as C  # noqa: E402
from . import geo  # noqa: E402

COLLUSION_EXTRA = [0.2, 0.5, 1.0, 2.0]
LAYERS = ["registry", "presence", "quantity", "review", "insufficient", "passed"]
LAYER_CN = {"registry": "登记层", "presence": "卫星位置层", "quantity": "卫星产量层",
            "review": "待复核", "insufficient": "无法核验", "passed": "通过"}
FLAG_LAYERS = {"registry", "presence", "quantity"}   # 判为"可疑"（会直接告诉买家/渔民有问题）


# ---------------------------------------------------------------------------
# 登记库
# ---------------------------------------------------------------------------
def polygon_info(claim_geoms: dict, zone_bbox: list[float]) -> pd.DataFrame:
    rows = []
    for f in claim_geoms["features"]:
        p = f["properties"]
        cx, cy = geo.centroid(f["geometry"]["coordinates"][0])
        rows.append({"pid": str(p["pid"]), "area_ha": float(p["area_m2"]) / 1e4, "lon": cx, "lat": cy,
                     "in_zone": zone_bbox[0] <= cx <= zone_bbox[2] and zone_bbox[1] <= cy <= zone_bbox[3]})
    return pd.DataFrame(rows)


def idle_candidates(polys: pd.DataFrame, placement: str | None = None) -> list[str]:
    placement = placement or C.IDLE_PLACEMENT
    col = {"gap": "gap", "open_sea": "open_sea"}.get(placement)
    if col is None:
        raise ValueError(f"未知的空挂证位置：{placement}")
    if col not in polys:
        polys = polys.assign(gap=annual.is_gap(polys))
    return polys[polys[col] & polys["in_zone"]]["pid"].tolist()


def build_registry(polys: pd.DataFrame, rng: np.random.Generator, idle_share: float = 0.2,
                   placement: str | None = None) -> pd.DataFrame:
    farm = polys[polys["kind"] == "farm"]
    inz = farm[farm["in_zone"]]["pid"].tolist()
    outz = farm[~farm["in_zone"]]["pid"].tolist()
    rng.shuffle(inz)
    rng.shuffle(outz)
    rows = []

    def assign(pids, prefix):
        i, h = 0, 0
        while i < len(pids):
            h += 1
            k = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            for pid in pids[i:i + k]:
                rows.append({"holder": f"{prefix}{h:03d}", "pid": pid, "use": "farm"})
            i += k
        return h

    n_h = assign(inz, "H")
    assign(outz, "O")
    # 有证不养：示范区内的空地块，登记给部分区内养殖户（或单独的"只有空证"的户）
    sea = idle_candidates(polys, placement)
    n_idle = min(len(sea), max(1, int(round(idle_share * len(inz)))))
    for j, pid in enumerate(rng.choice(sea, size=n_idle, replace=False) if n_idle else []):
        holder = f"H{int(rng.integers(1, n_h + 1)):03d}" if rng.random() < 0.7 else f"H{n_h + 1 + j:03d}"
        rows.append({"holder": holder, "pid": pid, "use": "idle"})
    reg = pd.DataFrame(rows)
    reg = reg.merge(polys[["pid", "in_zone", "area_ha", "lon", "lat", "split"]], on="pid", how="left")
    reg["permit_id"] = [f"荣成海证-{2019}-{i + 1:04d}" for i in range(len(reg))]
    reg["valid_from"], reg["valid_to"] = 2019, 2026
    return reg[["permit_id", "holder", "pid", "use", "in_zone", "area_ha", "lon", "lat", "split", "valid_from", "valid_to"]]


# ---------------------------------------------------------------------------
# 批次生成
# ---------------------------------------------------------------------------
def split_batches(total: float, rng: np.random.Generator) -> list[float]:
    k = int(rng.integers(2, 7))
    return list(total * rng.dirichlet(np.ones(k)))


def make_cases(reg: pd.DataFrame, comp: pd.DataFrame, years: list[int], rng: np.random.Generator,
               y_t_ha: float) -> pd.DataFrame:
    """每个 case = 一个"养殖户 × 声明地块 × 年份 × 场景"，返回其全部批次。"""
    cover = comp.drop_duplicates("pid").set_index("pid")["cover_ref"].to_dict()
    farm_reg = reg[reg["use"] == "farm"]
    typical_cover = float(np.nanmedian([cover.get(p, np.nan) for p in farm_reg["pid"]]))
    rows = []
    cid = 0

    def actual(pid, area):
        c = cover.get(pid, np.nan)
        c = typical_cover if not np.isfinite(c) else c
        return float(c * area * y_t_ha * rng.uniform(0.6, 1.0))

    def add_case(scenario, holder, pid, year, chunks, factor=1.0, split=None):
        nonlocal cid
        cid += 1
        for i, (q, src) in enumerate(chunks):
            rows.append({"case": f"K{cid:06d}", "scenario": scenario, "factor": factor, "holder": holder,
                         "pid": pid, "year": year, "seq": i + 1, "qty_t": round(q, 2), "source": src,
                         "split": split})

    for year in years:
        for r in farm_reg.itertuples(index=False):
            base = actual(r.pid, r.area_ha)
            if r.in_zone:
                add_case("honest", r.holder, r.pid, year, [(q, "own") for q in split_batches(base, rng)], 1.0, r.split)
                for f in C.QUANTITY_INFLATION:
                    add_case("S4_launder", r.holder, r.pid, year,
                             [(q, "mixed") for q in split_batches(base * f, rng)], f, r.split)
                for e in COLLUSION_EXTRA:
                    own = [(q, "own") for q in split_batches(base, rng)]
                    ext = [(q, "outsider") for q in split_batches(base * e, rng)]
                    chunks = own + ext
                    order = rng.permutation(len(chunks))
                    add_case("S5_collusion", r.holder, r.pid, year, [chunks[i] for i in order], 1 + e, r.split)
            else:
                add_case("S1_outside", r.holder, r.pid, year, [(q, "own") for q in split_batches(base, rng)],
                         1.0, r.split)
        for r in reg[reg["use"] == "idle"].itertuples(index=False):
            fake = typical_cover * r.area_ha * y_t_ha * rng.uniform(0.6, 1.0)
            add_case("S2_idle", r.holder, r.pid, year, [(q, "outside") for q in split_batches(fake, rng)],
                     1.0, r.split)
        # S3：区外养殖户冒用示范区内他人地块（每个区外户每年冒用一块）
        inz = farm_reg[farm_reg["in_zone"]]
        for holder in sorted(reg[reg["holder"].str.startswith("O")]["holder"].unique()):
            v = inz.iloc[int(rng.integers(0, len(inz)))]
            add_case("S3_neighbour", holder, v.pid, year,
                     [(q, "outside") for q in split_batches(actual(v.pid, v.area_ha), rng)], 1.0, v.split)
    b = pd.DataFrame(rows)
    b["batch_id"] = [f"RC-{y}-{h}-{c[1:]}-{s:02d}" for y, h, c, s in zip(b["year"], b["holder"], b["case"], b["seq"])]
    return b


# ---------------------------------------------------------------------------
# 三层核验
# ---------------------------------------------------------------------------
def verify_batches(b: pd.DataFrame, reg: pd.DataFrame, comp: pd.DataFrame, tau: float, y_t_ha: float) -> pd.DataFrame:
    owner = reg.set_index("pid")[["holder", "in_zone", "valid_from", "valid_to", "area_ha"]]
    c = comp.set_index(["pid", "year"])
    out = b.copy()
    out = out.sort_values(["case", "seq"])
    out["cum_t"] = out.groupby("case")["qty_t"].cumsum()
    layer, reason, cap = [], [], []
    for r in out.itertuples(index=False):
        if r.pid not in owner.index:
            layer.append("registry"); reason.append("无海域登记"); cap.append(np.nan)  # noqa: E702
            continue
        o = owner.loc[r.pid]
        if not bool(o["in_zone"]):
            layer.append("registry"); reason.append("地块不在示范区内"); cap.append(np.nan)  # noqa: E702
            continue
        if o["holder"] != r.holder:
            layer.append("registry"); reason.append("非本人海域"); cap.append(np.nan)  # noqa: E702
            continue
        if not (o["valid_from"] <= r.year <= o["valid_to"]):
            layer.append("registry"); reason.append("权证不在有效期"); cap.append(np.nan)  # noqa: E702
            continue
        st = c.loc[(r.pid, r.year), "status"] if (r.pid, r.year) in c.index else "insufficient"
        if st == "review":
            layer.append("review"); reason.append("卫星证据不一致，待复核（不判可疑）"); cap.append(np.nan)  # noqa: E702
            continue
        if st != "ok":
            layer.append("insufficient"); reason.append("卫星影像不足，无法核验"); cap.append(np.nan)  # noqa: E702
            continue
        s = c.loc[(r.pid, r.year)]
        capacity = float(s["cover_sat"]) * float(o["area_ha"]) * y_t_ha
        cap.append(capacity)
        if not bool(s["present"]):
            layer.append("presence"); reason.append("卫星未见养殖设施")  # noqa: E702
            continue
        if r.cum_t > tau * capacity:
            layer.append("quantity"); reason.append("累计声明量超出卫星产能")  # noqa: E702
            continue
        layer.append("passed"); reason.append("通过")  # noqa: E702
    out["layer"], out["reason"], out["capacity_t"] = layer, reason, cap
    out["verdict"] = np.select([out["layer"] == "passed", out["layer"] == "review", out["layer"] == "insufficient"],
                               ["通过", "待复核", "无法核验"], default="可疑")
    return out


def calibrate_tau(reg: pd.DataFrame, comp: pd.DataFrame, b: pd.DataFrame, y_t_ha: float) -> float:
    h = b[(b["scenario"] == "honest") & (b["split"] == "calib")]
    tot = h.groupby(["case", "pid", "year"])["qty_t"].sum().reset_index()
    tot = tot.merge(comp[["pid", "year", "cover_sat", "status", "present"]], on=["pid", "year"], how="left")
    tot = tot[(tot["status"] == "ok") & (tot["present"] == 1.0)]
    area = reg.set_index("pid")["area_ha"]
    ratio = tot["qty_t"] / (tot["cover_sat"] * tot["pid"].map(area) * y_t_ha).clip(lower=1e-9)
    return float(np.quantile(ratio, 1 - C.QTY_FALSE_ALARM_TARGET))


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def summarise(v: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    te = v[v["split"] == "test"]
    order = {k: i for i, k in enumerate(LAYERS)}
    # 每个 case 的"拦截层" = 最早被拦的那一批的层（没被拦 = passed）
    flagged = te[te["layer"] != "passed"]
    first = flagged.sort_values("seq").groupby("case")["layer"].first()
    cases = te.groupby("case").agg(scenario=("scenario", "first"), factor=("factor", "first"),
                                   total_t=("qty_t", "sum")).reset_index()
    cases["caught_by"] = cases["case"].map(first).fillna("passed")
    vol = te.assign(f=te["layer"] != "passed").groupby("case").apply(
        lambda d: float(d.loc[d["f"], "qty_t"].sum() / d["qty_t"].sum()), include_groups=False)
    cases["vol_flagged"] = cases["case"].map(vol)

    tab = pd.crosstab(cases["scenario"], cases["caught_by"], normalize="index").reindex(columns=LAYERS, fill_value=0)
    tab["n_cases"] = cases.groupby("scenario").size()
    m = {"n_cases_test": int(len(cases))}
    for sc, g in cases.groupby("scenario"):
        m[sc] = {"n_cases": int(len(g)),
                 "flagged_suspicious": float(g["caught_by"].isin(FLAG_LAYERS).mean()),
                 "detected": float((g["caught_by"] != "passed").mean()),
                 "caught_by": {k: float((g["caught_by"] == k).mean()) for k in LAYERS},
                 "volume_flagged_share": float(g["vol_flagged"].mean())}
        if sc in ("S4_launder", "S5_collusion"):
            m[sc]["detected_by_factor"] = {float(f): float((x["caught_by"] != "passed").mean())
                                           for f, x in g.groupby("factor")}
            m[sc]["flagged_by_factor"] = {float(f): float(x["caught_by"].isin(FLAG_LAYERS).mean())
                                          for f, x in g.groupby("factor")}
    return m, tab


def plot(tab: pd.DataFrame, m: dict, out: Path) -> None:
    colors = {"registry": "tab:purple", "presence": "tab:blue", "insufficient": "tab:gray",
              "quantity": "tab:orange", "review": "khaki", "passed": "#dddddd"}
    order = [s for s in ["honest", "S1_outside", "S2_idle", "S3_neighbour", "S4_launder", "S5_collusion"] if s in tab.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    bottom = np.zeros(len(order))
    for L in LAYERS:
        vals = tab.loc[order, L].to_numpy()
        axes[0].bar(order, vals, bottom=bottom, color=colors[L], label=L)
        bottom += vals
    axes[0].set_ylabel("Share of cases (test parcels)")
    axes[0].set_title("Which layer stops each scenario")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    for sc, lab in [("S4_launder", "Laundering (declared × r)"), ("S5_collusion", "Borrowed account (1 + extra)")]:
        d = m.get(sc, {}).get("detected_by_factor", {})
        if d:
            xs = sorted(d)
            axes[1].plot(xs, [d[x] for x in xs], "o-", label=lab)
    if "honest" in m:
        axes[1].axhline(m["honest"]["flagged_suspicious"], color="gray", ls="--", label="Honest flagged (false alarm)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Declared / actual production")
    axes[1].set_ylabel("Detected")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run(claims: pd.DataFrame, act_raw: pd.DataFrame, cover: dict[str, float], claim_geoms: dict, area: str,
        window: list[int] | None = None, seed: int = C.RANDOM_SEED, y_t_ha: float = C.KELP_FRESH_T_PER_HA):
    am, sep, comp = annual.run(claims, act_raw, cover, window)
    m, tab, reg, v = run_with_comp(comp, am, claims, cover, claim_geoms, area, seed, y_t_ha)
    return m, tab, reg, v, sep


def run_with_comp(comp: pd.DataFrame, am: dict, claims: pd.DataFrame, cover: dict[str, float], claim_geoms: dict,
                  area: str, seed: int = C.RANDOM_SEED, y_t_ha: float = C.KELP_FRESH_T_PER_HA):
    """comp：每个多边形 × 年的卫星结论（列 pid, year, status[ok/review/insufficient], present, cover_sat, cover_ref）。"""
    polys = annual.polygon_table(claims)
    calib, _ = annual.split_parcels(list(cover))
    polys["split"] = np.where(polys["ref_pid"].isin(calib), "calib", "test")
    polys["open_sea"] = annual.is_open_sea(polys)
    polys["gap"] = annual.is_gap(polys)
    polys = polys.merge(polygon_info(claim_geoms, C.STUDY_AREAS[area]["zone_bbox"]), on="pid", how="left")

    rng = np.random.default_rng(seed + 7)
    reg = build_registry(polys, rng)
    comp_r = comp[comp["pid"].isin(set(reg["pid"]))].copy()
    years = am["years"]
    b = make_cases(reg, comp_r, years, rng, y_t_ha)
    tau = calibrate_tau(reg, comp_r, b, y_t_ha)
    v = verify_batches(b, reg, comp_r, tau, y_t_ha)
    m, tab = summarise(v)
    m = {"annual": am, "tau": tau, "kelp_fresh_t_per_ha": y_t_ha,
         "registry": {"holders_in_zone": int(reg[reg["holder"].str.startswith("H")]["holder"].nunique()),
                      "holders_outside": int(reg[reg["holder"].str.startswith("O")]["holder"].nunique()),
                      "farm_parcels_in_zone": int(((reg["use"] == "farm") & reg["in_zone"]).sum()),
                      "farm_parcels_outside": int(((reg["use"] == "farm") & ~reg["in_zone"]).sum()),
                      "idle_permits": int((reg["use"] == "idle").sum()),
                      "idle_placement": C.IDLE_PLACEMENT},
         **m}
    return m, tab, reg, v


def main() -> None:
    ap = argparse.ArgumentParser(description="完整流程模拟：登记 + 批次 + 三层核验")
    annual.add_common_args(ap)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--idle-place", default=None, choices=["gap", "open_sea"])
    args = ap.parse_args()
    if args.metric:
        C.ACTIVITY_METRIC = args.metric
    if args.idle_place:
        C.IDLE_PLACEMENT = args.idle_place
    window = [int(x) for x in args.window.split(",")] if args.window else None
    claims, act_raw, cover = annual.load_inputs(args)
    m, tab, reg, v, sep = run(claims, act_raw, cover, geo.load_geojson(args.claim_geoms), args.area, window)

    od = Path(args.outdir or Path(C.OUTPUT_DIR) / f"{args.area}_trace")
    od.mkdir(parents=True, exist_ok=True)
    reg.to_csv(od / "registry.csv", index=False, encoding="utf-8-sig")
    v.to_csv(od / "batches.csv", index=False, encoding="utf-8-sig")
    tab.rename(columns=LAYER_CN).to_csv(od / "scenario_table.csv", encoding="utf-8-sig")
    with open(od / "trace_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    plot(tab, m, od / "trace_layers.png")
    print(tab.round(3).to_string())
    print(json.dumps({k: m[k] for k in m if k not in ("annual",)}, ensure_ascii=False, indent=2)[:3000])
    print("输出目录：", od)


if __name__ == "__main__":
    main()
