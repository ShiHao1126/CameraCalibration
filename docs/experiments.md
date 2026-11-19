# 历史实验与排障入口

正常使用请从 [README](../README.md) 的默认 `hybrid` 流程开始。本页保留开发期间的对照和假设检验，部分入口依赖本地历史缓存，克隆仓库不会自动获得这些数据。实验不连接设备，不驱动机器人。

## 不依赖现场数据的检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/tests/synthetic_demo.py
python scripts/tests/observability_probe.py
```

`synthetic_demo.py` 在 `outputs/` 生成合成数据和求解结果。`observability_probe.py` 检查单块四角标记在重复姿态、纯平移、单轴旋转、多轴旋转情况下的局部秩，默认仅打印结果；需保存时指定 `--output outputs/design_experiments/observability-results.json`，并先建立父目录。局部满秩不等于全局唯一，也不代表真实标定精度。

## 历史四角点比较

```bash
python scripts/tests/compare_methods.py --run '<采集目录>'
python scripts/reprocess_calibration.py --run '<采集目录>'
```

前者比较本项目 `baseline/joint`；后者将早期使用 SUBPIX 的数据按原 r3kit 检测重新处理，在新的 `calibration/<时间>_r3kit_corners_joint/` 中保存角点和四角点联合结果。原始数据、原处理结果及计划分组保留。

## 完整 r3kit 原方法排障

```bash
python scripts/tests/diagnose_r3kit.py \
  --run '<采集目录>' --processed '<r3kit_corners_joint 结果目录>'
```

此历史入口依赖该采集 `diagnostics/r3kit_reference/` 的源码快照。它比较首帧完整流程和相同连拍角点的求解器，并做相机模型诊断。默认 `hybrid` 已包含 r3kit 对照，调用仓库内固定版本源码，不需要这个旧缓存入口。

原方法的每图姿态拟合 RMS、手眼闭合平移误差、固定外参测试重投影误差是不同指标；闭合误差也不是独立物理定位误差。

## 剩余像素误差假设检验

```bash
python scripts/tests/investigate_residuals.py \
  --run '<采集目录>' --processed '<r3kit_corners_joint 结果目录>' --stage initial
# 可选阶段：additional、complexity
```

结果写入 `diagnostics/residual_investigation/`。比较边缘定位、内部图案定位、相机参数复杂度、漂移和重力变形假设。`additional` 还依赖此前生成的 `background_optical_flow.json`，不是无缓存即可运行的一键入口。图像细化的可复用实现现位于 `camera_calibration/features.py`。

曾观察到放开更多相机参数时测试 RMS 降至约 0.858 px，但训练内分组验证恶化至约 2.314 px，因此没有将这一模型设为默认。80 mm 尺寸保持物理给定值，不通过拟合修改板尺寸来解释误差。

## 内部点方法消融对照

```bash
python scripts/calibrate_multipoint.py \
  --run '<采集目录>' --processed '<r3kit_corners_joint 结果目录>'
```

此入口依赖 `diagnostics/residual_investigation/internal_detections/` 和 `edge_observations.json`。比较边缘外角点＋径向畸变、仅内部点固定相机模型、内部点＋径向畸变、内外点各半权重混合四种方案。所有方案在共同外角点和内部点上分别评分，使用相同测试成员与训练内五个连续分组。

结果写入 `calibration/<时间>_direct_multipoint/`。该实验说明了选择混合方法的依据；正常标定请使用 `scripts/calibrate.py --run ...`，它从原始图像建立测量并默认采用混合方法，无需这些缓存。旧实验报告中的 `production_changed: false` 描述的是当时的实验状态，不代表当前默认方法。

详细前期设计保存在 `archive/project-plan-20261118.json`，仅供追溯；当前接口和结果语义以 README 和 `design.md` 为准。
