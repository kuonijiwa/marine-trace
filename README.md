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

```bash
python -m rs_verify.gee_step1_reference --area ailun --project <PROJECT>
```

GEE 会往 Drive 的 `marine_trace/` 文件夹导出三个文件。在 QGIS 里叠加
`ailun_reference_draft.geojson` 和 `ailun_s2_truecolor.tif`，删掉不是养殖设施的多边形，
保存为 `data/reference/ailun_reference.geojson`。
同时检查 `config.py` 里研究区范围框是否合适，必要时收紧并把 `verified` 改成 True。

### 2. 生成声明 + 活动曲线

```bash
python -m rs_verify.claims --area ailun --reference data/reference/ailun_reference.geojson
python -m rs_verify.gee_step2_activity --area ailun \
    --geojson data/claims/ailun_claim_geoms.geojson --project <PROJECT>
```

把导出的 `ailun_activity.csv` 放到 `data/activity/`。

### 3. 造假实验

```bash
python -m rs_verify.experiment --area ailun \
    --claims data/claims/ailun_claims.csv \
    --activity data/activity/ailun_activity.csv --calibrate
```

结果在 `outputs/ailun/`。

### 不连 GEE 先试跑（合成数据）

```bash
python -m rs_verify.simulate --outdir data/sim
python -m rs_verify.experiment --area ailun --claims data/sim/ailun_claims.csv \
    --activity data/sim/ailun_activity.csv --calibrate --outdir outputs/sim
python -m pytest -q tests
```

## 已知待定事项

- 研究区范围框是估计值，第 1 步要人工核对。
- 爱伦湾的浮筏可能全年在海上（海带 11→5 月，龙须菜 7→10 月），时间核验能否成立要看第 2 步的活动曲线。
- 产量检查需要单位面积产量（统计年鉴/文献），目前是占位，实验设计也待定。
- 人工鱼礁在水下，卫星看不到；本方法只适用于有水面设施（筏架、网箱）的牧场。
