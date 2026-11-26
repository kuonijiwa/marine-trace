# marine-trace

卫星核验产地的海洋牧场海产品溯源系统（开发中）。

Satellite-verified provenance for marine-ranching seafood (work in progress).

## 这个项目要回答什么

现在的溯源系统里，"产地经纬度"是养殖户自己填的，相当于自己给自己开证明。
本项目用免费的 Sentinel-1 雷达卫星影像，对每一批产品的**产地声明**做独立核验：

| 检查 | 问的问题 | 能识别的造假 |
|---|---|---|
| 空间分 S | 声明的那片海，声明时段里卫星看到养殖设施了吗？ | 位置冒用（围栏挪到空海面/别处） |
| 时间分 T | 声明的月份，是这片海平时有设施的月份吗？ | 时段错报 |
| 产量比 Q | 声明数量是否超过卫星看到的养殖规模？ | 产量冒用（外面的货洗成牧场货） |

消费者扫码看到的"产地可信度" = w·S + (1−w)·T，产量比作为第三道检查。

研究区：国家级海洋牧场示范区——荣成爱伦湾（筏式海带）、长岛南隍城（大型网箱平台）。

## 目录

```
rs_verify/                 阶段1：卫星核验
  config.py                所有参数（标 TODO 的待确认）
  gee_common.py            GEE：S1 预处理、"设施像元"判定
  gee_step1_reference.py   第1步：自动提取参考边界 + 真彩色底图（导出到 Drive）
  gee_step2_activity.py    第2步：每个多边形的活动曲线（导出到 Drive）
  claims.py                生成真实/虚假产地声明
  verify.py                核验算法（空间/时间/产量）
  experiment.py            造假实验：AUC、检出率、敏感性曲线
  simulate.py              合成数据（仅测试程序用，不是结果！）
  geo.py                   几何小工具（平移、面积、重叠）
tests/                     单元测试
data/                      数据（大文件不进 git）
outputs/                   结果
docs/                      方案与实验记录
```

## 运行步骤

### 0. 环境（在你自己的电脑上）

```bash
pip install -r requirements.txt
earthengine authenticate          # 第一次需要登录 Google 账号
```

需要一个已开通 Earth Engine 的 Google Cloud 项目 ID（下面记作 `<PROJECT>`）。

### 1. 参考边界

**推荐：直接用已发表的全国海水养殖区数据集**（Fu et al., 2021, ESSD, 16 m, GF-1 2016–2019；
https://zenodo.org/records/3881612 ，CC BY 4.0）。浮筏识别不是本项目的创新点，用独立团队发表的数据作参考，
避免"自己考自己"。荣成试跑：与 2025 年 S2 真彩色图吻合很好，研究区内约 208 km²。

```bash
# 1a. 从全国 kmz 裁出研究区（kmz 先改名为 fu2021.kmz）
python -m rs_verify.import_reference_kmz --area ailun --kmz <路径>/fu2021.kmz
# 1b. 叠加到 S2 真彩色图上检查
python -m rs_verify.inspect_step1 --area ailun --source file \
    --geojson data/reference/ailun_fu2021_reference.geojson --project <PROJECT>
# 1c. 切成 500 m 网格地块（覆盖率 ≥ 60%）——声明的基本单位
python -m rs_verify.make_parcels --area ailun --reference data/reference/ailun_fu2021_reference.geojson
```

**备选：自己用 Sentinel-2 光学**提取（海带浮筏在光学图上清晰可见；参考来自光学、核验用雷达，两个独立传感器）。
分两个阶段（重计算先存成 GEE 资产，避免超时）：

```bash
# 阶段 A：旺季去云合成 → 浮筏纹理图（存到 GEE 资产），并导出真彩色图 ailun_s2_truecolor_v2.tif 到 Drive
python -m rs_verify.gee_step1_reference --area ailun --project <PROJECT> --stage s2mask
# 等任务页面里 ailun_s2_texture_asset 完成后，阶段 B：按阈值勾边界（可多试几个阈值，不用重算纹理）
python -m rs_verify.gee_step1_reference --area ailun --project <PROJECT> --stage s2vectors --threshold 0.05
```

Drive 的 `marine_trace/` 里会出现 `ailun_s2_reference_draft_t050.geojson`。下载后运行检查脚本：

```bash
python -m rs_verify.inspect_step1 --area ailun --source s2 --geojson <GeoJSON路径> --project <PROJECT> --threshold 0.05
```

它会打印统计和"海面纹理分位数"（帮助选阈值），并生成分布图和卫星叠加图。
确认后在 QGIS 里删掉港口等误检，保存为 `data/reference/ailun_reference.geojson`。

（雷达 S1 的 `--stage persistence / vectors` 仍保留，适合网箱等强回波结构。）

### 2. 生成声明 + 活动曲线

```bash
# 抽 100 个地块出声明；假围栏是否压到别人的养殖区，用全部参考养殖区判断
python -m rs_verify.claims --area ailun --reference data/reference/ailun_parcels.geojson \
    --max-refs 100 --raft-reference data/reference/ailun_fu2021_reference.geojson
# 只用一条相对轨道（自动选），每年一个导出任务
python -m rs_verify.gee_step2_activity --area ailun \
    --geojson data/claims/ailun_claim_geoms.geojson --project <PROJECT> --by-year
```

把导出的 `ailun_activity_<年份>.csv` 都放到 `data/activity/`。

### 3. 造假实验

```bash
python -m rs_verify.experiment --area ailun \
    --claims data/claims/ailun_claims.csv \
    --activity data/activity/ailun_activity_*.csv --calibrate
```

结果在 `outputs/ailun/`。（`*` 通配符由脚本自己展开，Windows 下也能用。）

可选参数：`--metric abs_excess`（不受风浪影响的绝对指标）；`--spatial-weight 1`（只看空间分，
全年挂养的筏区没有淡季，季节规律不可用）。

### 4. 产量核验（拦"基地里有地、再把外面的货洗进来"）

```bash
python -m rs_verify.quantity --area ailun --claims data/claims/ailun_claims.csv \
    --claim-geoms data/claims/ailun_claim_geoms.geojson \
    --raft-reference data/reference/ailun_fu2021_reference.geojson \
    --activity data/activity/ailun_activity_*.csv --metric abs_excess
```

卫星估计每块地的养殖覆盖率（一半地块标定、另一半检验），声明量超过"覆盖率 × 产能"的容差就报警。
结果在 `outputs/ailun_quantity/`。

### 5. 年度检查（主方法）+ 完整流程模拟

筏架几个月不变，不需要逐景监测：每块地每年只在"检查窗口"里（由数据选出可分性最好的连续 3 个月），
用海况合适（背景 −28 ~ −18 dB）的影像取中位数判断一次。

```bash
# 年度检查：选窗口、标定门限、检验组上的检出率
python -m rs_verify.annual --area ailun --claims data/claims/ailun_claims.csv \
    --claim-geoms data/claims/ailun_claim_geoms.geojson \
    --raft-reference data/reference/ailun_fu2021_reference.geojson \
    --activity data/activity/ailun_activity_*.csv --metric abs_excess
# 完整流程：模拟海域登记库 + 批次溯源 ID + 三层核验（登记 / 卫星位置 / 卫星产量）+ 五类造假
python -m rs_verify.trace_sim （参数同上）
```

结果在 `outputs/ailun_annual/` 和 `outputs/ailun_trace/`。登记库和批次是模拟的（真实登记数据不公开），
地块和卫星结果是真实的；示范区边界是 `config.STUDY_AREAS[...]["zone_bbox"]` 里的近似范围。

### 6. 雷达 + 光学融合（精度优先）与目视判读验证

误判核查图显示：雷达判"不在养"的养殖地块，同年光学图上大多筏格清楚 → 加光学作第二证人。
判定三态：**通过**（任一证人明确见筏）/ **可疑**（两个证人都明确没见）/ **待复核**（其余）。
"没见"门限取真实养殖地块的 2% 分位，诚实渔民被判可疑的概率压到很低。

```bash
# a. 光学逐景指标导出（GEE，每年一个任务）→ CSV 放到 data/activity_s2/
python -m rs_verify.gee_s2_activity --area ailun --geojson data/claims/ailun_claim_geoms.geojson --project <PROJECT> --by-year
# b. 融合判定 + 完整流程
python -m rs_verify.fusion （annual 的参数） --s2 data/activity_s2/ailun_s2_*.csv --metric abs_excess --trace
# c. 目视判读样本（盲判图集）
python -m rs_verify.visual_sample --area ailun --project <PROJECT> --claim-geoms data/claims/ailun_claim_geoms.geojson \
    --fusion outputs/ailun_fusion/fusion_composite.csv --s2 data/activity_s2/ailun_s2_*.csv
# d. 判读完后再跑 b，加 --visual-labels <判读表>，按目视真值评估
```

### 诊断图

```bash
python -m rs_verify.diagnose_activity --area ailun --claims data/claims/ailun_claims.csv \
    --activity data/activity/ailun_activity_*.csv [--metric abs_excess]
```

### 不连 GEE 先试跑（合成数据）

```bash
python -m rs_verify.simulate --outdir data/sim
python -m rs_verify.experiment --area ailun --claims data/sim/ailun_claims.csv \
    --activity data/sim/ailun_activity.csv --calibrate --outdir outputs/sim
python -m pytest -q tests
```

## 已知待定事项

- 研究区范围框已按第一次结果放大（2026-11-25），仍需核对。
- 第 2 步的主指标是"多边形平均雷达亮度 / 海面背景"（lin_ratio）：海带浮筏回波只比海面略亮，
  单像元"亮 6 dB"的规则会漏掉大部分浮筏（第一次试跑已证实）。
- 已证实：爱伦湾/桑沟湾筏架全年在海上（海带 11 月→次年 4–8 月，牡蛎、海参全年），卫星无法核验收获季节；
  因此主方法改为"每年一个检查窗口"的年度检查（annual.py）。
- 单位面积产量目前是粗估（config.KELP_FRESH_T_PER_HA），只影响展示的吨数；论文前换成统计年鉴原始出处。
- 海域登记库、示范区边界是模拟的（真实数据不公开）。
- 人工鱼礁在水下，卫星看不到；本方法只适用于有水面设施（筏架、网箱）的牧场。
