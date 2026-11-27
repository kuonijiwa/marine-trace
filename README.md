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

### 7. 定位：已登记地块的年度"是否在养"筛查（不做测绘）

位置和归属以政府登记（养殖证/海域使用权证上的界址坐标）为准，卫星不做测绘，只对已登记地块每年核查一次：
今年还养不养、规模对不对得上。依据：新《渔业法》（2026-05-01 施行）第六十五条第二款，
"无正当理由使水域、滩涂荒芜满一年的，责令限期开发利用；逾期未开发利用的，吊销养殖证"。
卫星只出**筛查名单**（判"不在养"），交渔政现场核查，不直接处罚。

三个新选项（fusion 和 tradeoff 都有）：
- `--s2-agg median|persist`：光学多景合成。persist = 5 张好图的 25% 分位（多数图上都持续有的纹理），
  压掉海浪、浑浊锋面这类偶发纹理。
- `--idle-seasons 1|2`：荒芜认定。1 = 一季两证人都没看到筏即进名单；2 = 连续两季才进（冤枉更少，晚一年发现）。
  tradeoff 默认两种都算（旋钮 C）。
- `--idle-place gap|open_sea`：模拟空挂证放在哪。gap（默认）= 养殖区内或旁边的空地块（偏移 ≤ 1000 m、
  与参考筏区不重叠），四周是别人的筏架，更接近现实；open_sea = 远处开阔海（对照）。

```bash
# 光学导出时地块向内收缩 50 m，防邻居筏架污染边缘纹理（输出 ailun_s2in50_<year>.csv）
python -m rs_verify.gee_s2_activity --area ailun --geojson data/claims/ailun_claim_geoms.geojson \
    --project <PROJECT> --by-year --inner-buffer 50
# 取舍表：两种荒芜规则 × 门限网格，并按两轮目视真值统计真冤枉/真检出
python -m rs_verify.tradeoff （annual 的参数） --s2 data/activity_s2/ailun_s2_*.csv --metric abs_excess \
    --s2-agg persist --visual-labels data/visual/ailun_visual_round1.csv data/visual/ailun_visual_round2.csv
```

### 8. 默认配置、两级名单与论文图（v3 结果之后）

v3 实测（目视真值 226 个：有筏 114、无筏 112）：
- 光学多景持续性（`S2_AGG = "persist"`，现为默认）：真筏判"在养" 83% → 94%，冤枉率不变（0.9%）。
- 地块内缩 50 m：真筏判在养降到 90%、空挂证检出 22% → 18%，**不采用**（邻居筏架污染不是误判主因）。
- 一季规则 vs 连续两季：名单准确率 98% vs 100%，查出真空地块 52% vs 21%（均衡档）。

`fusion` 现在另外输出 `screening_list.csv`，两级名单对应新《渔业法》第六十五条第二款：
**整改通知**（本季两证人都没看到筏，"责令限期开发利用"）→ **吊销核查**（连续两季没看到，"逾期未开发利用的，吊销养殖证"的线索，交渔政现场核查）。

```bash
# 论文主图：取舍图（目视真值 + 模拟造假）和两级名单流程图
python -m rs_verify.paper_figs --tradeoff-table outputs/v3_tradeoff_per/tradeoff_table.csv --outdir outputs/paper_figs
```

第二判读者盲判：`data/visual/blind2_key.csv` 是 50 张盲判图（B01–B50）与第二轮样本的对照（**不要给判读人看**）。

### 9. 置信区间与典型案例

```bash
# 关键指标的 95% 置信区间（按参考地块整群自助抽样 2000 次；同一地块不同年份不当作独立样本）
python -m rs_verify.uncertainty --composite outputs/v4_fusion/fusion_composite.csv \
    --metrics outputs/v4_fusion/fusion_metrics.json --batches outputs/v4_fusion/batches.csv \
    --visual-labels data/visual/ailun_visual_round1.csv data/visual/ailun_visual_round2.csv --outdir outputs/v4_uncertainty
# 典型案例 R0035（登记在、海已不养 → 整改通知 → 吊销核查），对照地块一直在养；--thumbs 需要 GEE
python -m rs_verify.case_study --composite outputs/v4_fusion/fusion_composite.csv \
    --metrics outputs/v4_fusion/fusion_metrics.json --pid R0035 --compare R0014 \
    --activity "data/activity/ailun_activity_*.csv" --s2 "data/activity_s2/ailun_s2_*.csv" --metric abs_excess \
    --thumbs --claim-geoms data/claims/ailun_claim_geoms.geojson --project <PROJECT> --outdir outputs/case_study
```

### 10. 真实登记地块：大连金普新区养殖证 / 海域使用权公示

金普新区政府网站公开了养殖证和海域使用权申请公示（含逐点界址坐标、面积、养殖方式、街道、海域管理号、
设立/续期/变更/转让）。Codex 下载原文并提取文字到 `data/registry/dalian_jinpu/raw/`（含 index.csv），
再解析成登记多边形：

```bash
python -m rs_verify.registry_parse --raw data/registry/dalian_jinpu/raw --out data/registry/dalian_jinpu/registry
```

按界址点切分每一宗（PDF 版式里单元格和坐标行交错），用"坐标算出的面积 ≈ 公示面积（±5%）"自检；
同一组界址点多次公示只保留一块；滩涂底播 / 纯海底的标为卫星不可见（`raft_visible=False`）。
不提取申请人姓名，地块用匿名编号 DL00001…。扫描件 PDF（没有文字层）暂不处理。

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
