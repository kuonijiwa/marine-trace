"""大连金普新区登记地块的卫星指标导出（定时一次跑完：先雷达，雷达全部完成后再光学）。

为什么这样排：GEE 社区档每月 150 EECU-小时，每月 1 日太平洋时间零点（东京 17:00）重置。
雷达是全年可用的证人，先导；雷达任务全部结束后才提交光学，万一额度不够至少先有一个证人的数据。
光学只导海带季 12–5 月（不导其他月份、不做内缩 50 m 那类试验版本），尽量省额度。

步骤：
  1. 从 registry_parcels.geojson 挑出"卫星可见 且 面积自检 ok"的地块（应为 896 块），
     补上 area_m2，存成 jinpu_screen.geojson（不需要 GEE）。
  2. 时间没到 --not-before 就直接退出（防止在受限模式下提前跑，白白消耗）。
  3. 提交雷达逐年导出 jinpu_activity_<年>.csv（单一相对轨道，自动选覆盖最多的）。
  4. 每 --poll-min 分钟查一次任务状态，雷达全部结束（或超时）后提交光学逐年导出 jinpu_s2_<年>.csv。
  5. 再等光学结束，写状态文件。CSV 由 GEE 直接导出到 Drive 的 marine_trace 文件夹。
日志：outputs/dalian_export/export_log.txt；任务状态：outputs/dalian_export/tasks.json

用法（只准备、不提交，可随时跑）：
  python -m rs_verify.dalian_export --project <PROJECT> --prepare-only
正式（定时任务里跑）：
  python -m rs_verify.dalian_export --project <PROJECT>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from . import config as C
from . import geo

AREA = "jinpu"
REG = Path("data/registry/dalian_jinpu/registry/registry_parcels.geojson")
SCREEN = Path("data/registry/dalian_jinpu/registry/jinpu_screen.geojson")
OUT = Path("outputs/dalian_export")
NOT_BEFORE = "2026-12-01T17:05:00+09:00"
DONE_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(OUT / "export_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def prepare_screen(reg_path: Path = REG, out_path: Path = SCREEN) -> dict:
    """卫星可见 且 qc == ok 的登记地块；补 area_m2（GEE 导出脚本需要）。"""
    gj = geo.load_geojson(reg_path)
    feats = []
    for f in gj["features"]:
        p = f["properties"]
        if str(p.get("raft_visible")).lower() in ("true", "1") and p.get("qc") == "ok":
            q = dict(p)
            q["area_m2"] = float(p["area_calc_ha"]) * 1e4
            feats.append({"type": "Feature", "properties": q, "geometry": f["geometry"]})
    out = geo.feature_collection(feats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geo.save_geojson(out, out_path)
    return out


def too_early(not_before: str, now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    return now < dt.datetime.fromisoformat(not_before)


def years(start: str, end: str) -> list[tuple[str, str, str]]:
    return [(f"{y}-01-01", f"{y + 1}-01-01", f"_{y}") for y in range(int(start[:4]), int(end[:4]) + 1)]


def wait_all(tasks: list[tuple[str, object]], poll_min: float, max_hours: float, label: str) -> dict:
    """tasks = [(名称, ee 任务)]；全部结束（成功/失败/取消）或超时就返回各任务状态。"""
    t0 = time.time()
    while True:
        states = {}
        for name, t in tasks:
            try:
                states[name] = t.status().get("state")
            except Exception as ex:  # noqa: BLE001  网络抖动：本轮记为 UNKNOWN，下轮再查
                states[name] = f"UNKNOWN({ex.__class__.__name__})"
        left = [k for k, s in states.items() if s not in DONE_STATES]
        log(f"{label}：{len(tasks) - len(left)}/{len(tasks)} 个任务已结束 {states}")
        if not left:
            return states
        if time.time() - t0 > max_hours * 3600:
            log(f"{label}：等待超过 {max_hours} 小时，不再等（未结束：{left}）")
            return states
        time.sleep(poll_min * 60)


def save_tasks(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tasks.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="大连登记地块卫星指标导出（先雷达后光学）")
    ap.add_argument("--project", required=False, default=None)
    ap.add_argument("--registry", default=str(REG))
    ap.add_argument("--drive-folder", default="marine_trace")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    ap.add_argument("--not-before", default=NOT_BEFORE, help="早于这个时间就退出（默认 GEE 配额重置后 5 分钟）")
    ap.add_argument("--poll-min", type=float, default=5.0)
    ap.add_argument("--max-hours", type=float, default=20.0, help="每个证人最多等多久")
    ap.add_argument("--prepare-only", action="store_true", help="只生成 jinpu_screen.geojson，不连 GEE")
    ap.add_argument("--skip-s1", action="store_true", help="雷达已导过，只导光学")
    args = ap.parse_args()

    gj = prepare_screen(Path(args.registry), SCREEN)
    n = len(gj["features"])
    log(f"筛查地块 {n} 块 → {SCREEN}")
    if args.prepare_only:
        return
    if too_early(args.not_before):
        log(f"现在早于 {args.not_before}（GEE 配额尚未重置），退出，不提交任何任务")
        return

    import ee

    from . import gee_common as G
    from .gee_s2_activity import per_scene_table, s2_collection
    from .gee_step2_activity import activity_table, add_water_fraction, pick_orbit, stats_region, to_ee_fc

    G.init(args.project)
    record = {"area": AREA, "n_parcels": n, "s1": {}, "s2": {}}
    periods = years(args.start, args.end)

    if not args.skip_s1:
        fc = add_water_fraction(to_ee_fc(gj))
        region = stats_region(gj)
        orbit, hist = pick_orbit(region, args.start, args.end)
        log(f"雷达：各相对轨道影像数 {hist} → 选用 {orbit}")
        record["rel_orbit"] = orbit
        tasks = []
        for start, end, suf in periods:
            desc = f"{AREA}_activity{suf}"
            t = ee.batch.Export.table.toDrive(collection=activity_table(fc, start, end, region, orbit),
                                              description=desc, folder=args.drive_folder, fileFormat="CSV",
                                              selectors=["pid", "date", "lin_ratio", "bright_frac", "bg_db", "n_pix",
                                                         "orbit_pass", "rel_orbit", "platform", "water_frac",
                                                         "area_m2"])
            t.start()
            tasks.append((desc, t))
            record["s1"][desc] = t.id
            log(f"已提交雷达 {desc}（{start} ~ {end}）：{t.id}")
        save_tasks(record)
        record["s1_states"] = wait_all(tasks, args.poll_min, args.max_hours, "雷达")
        save_tasks(record)

    fc2 = to_ee_fc(gj)
    region2 = stats_region(gj, pad_m=0)
    tasks = []
    for start, end, suf in periods:
        desc = f"{AREA}_s2{suf}"
        nimg = s2_collection(region2, start, end, C.S2_MONTHS).size().getInfo()
        t = ee.batch.Export.table.toDrive(collection=per_scene_table(fc2, start, end, region2, C.S2_MONTHS),
                                          description=desc, folder=args.drive_folder, fileFormat="CSV",
                                          selectors=["pid", "date", "tex", "clear", "b2", "n_pix", "area_m2", "tile"])
        t.start()
        tasks.append((desc, t))
        record["s2"][desc] = t.id
        log(f"已提交光学 {desc}（{start} ~ {end}，海带季 {nimg} 景）：{t.id}")
    save_tasks(record)
    record["s2_states"] = wait_all(tasks, args.poll_min, args.max_hours, "光学")
    save_tasks(record)
    log("全部结束。CSV 在 Drive 的 marine_trace 文件夹：jinpu_activity_<年>.csv、jinpu_s2_<年>.csv")


if __name__ == "__main__":
    main()
