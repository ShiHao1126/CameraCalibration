# camera_calibration

使用一块 **80 mm、ArUco `DICT_6X6_1000 / ID 0`** 标记，为固定在工作台上的第三视角相机做眼在手外标定。标记刚性固定在机械臂末端，程序同时估计相机位置和标记的安装位置，不要求预先测量标记相对法兰的变换。

默认流程：**手动拖拽录制轨迹 → 选择采样点 → 沿示教路径自动回放 → 停稳采集原图和实际位姿 → 图像筛选 → 内部点＋外角点联合优化 → 分组验证和 r3kit 对照**。

目前已在 Flexiv Rizon4R 和 RealSense D415 上完成 50 个姿态、每姿态 10 帧的实际采集。当前默认混合方法已接入自动采集后的计算，支持完全离线重算。当前结果尚未通过 1 px 外角点验收，**没有证明达到 1 mm 物理定位精度**。

## 设备与安装

当前配置在 [`configs/d415_rizon4r.yaml`](configs/d415_rizon4r.yaml)：

- 机械臂：`Rizon4R-062115`；RDK `1.9.2`，现场机器人软件 `3.11.1`。
- 相机：D415 `104122061018`；原生 RGB `1920×1080 @ 8 fps`，现场 USB 2.1。按序列号选择，不会选择同服务器上的 D405。
- 标记：黑色外边界边长 **80 mm**，不是外侧安装框的尺寸；工作距离约 0.6–1 m。
- 只采 RGB，不采深度。导出的外参对应 RGB 光学坐标。

已配置的服务器可直接使用：

```bash
cd /home/kbe6000/shihao/camera_calibration
source .venv/bin/activate
```

新环境推荐 Python 3.10。安装已验证版本（包括硬件 SDK 和测试依赖）：

```bash
git clone git@github.com:ShiHao1126/CameraCalibration.git camera_calibration
cd camera_calibration
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

只做离线计算或开发时，可在新环境中改用 `python -m pip install -e '.[test]' -c requirements-lock.txt`，无需安装机器人 SDK。未安装 RealSense SDK 时，对照 SDK 投影的两项测试会跳过。硬件适配器针对 RDK 1.9.x，不能直接换成 2.x。

本服务器另有 SSH 别名 `github-shihao`，对应 `ShiHao1126` 账号；服务器上的 Git 远程地址使用 `git@github-shihao:ShiHao1126/CameraCalibration.git`。该别名是本机配置，不随仓库分发。

## 使用方法

以下命令在项目根目录、激活 `.venv` 后执行。`<会话目录>`、`<采集目录>`、`<结果目录>` 都替换为程序实际打印的完整路径，不需要猜测或手动拼时间戳。

### 1. 检查连接（可选）

```bash
python scripts/check_devices.py
```

读取机械臂状态、相机参数并保存一张 RGB 图，不使能、不切控制模式、不移动机械臂。检查数据保存在 `outputs/`。不需要每次离线计算前运行。

### 2. 从 home 手动拖拽录制轨迹

将相机固定、标记刚性固定在末端。用 Elements 将机械臂放到选定的起始 home，并切到手动拖拽模式：

```bash
python scripts/record_trajectory.py
```

按提示在 home 停稳后按 Enter，随后拖拽机械臂扫过允许的工作范围；覆盖不同位置、距离和绕不同轴的倾斜，避免只有平移。**Ctrl+C 正常结束并保存**。程序打印 `SESSION=...`。

这个阶段仅以 50 Hz 记录机器人轨迹，**不连接相机、不采图、不判断标记是否完整**。home 是首次记录的实际关节状态，脚本不会调用 `PLAN-Home`。结束后留在示教终点，或手动回到起始 home。`--duration 60` 可立即开始定时录制，60 秒后自动保存。

### 3. 选择 50 个点、自动采集并标定

在 Elements 切到 AUTO，确认没有其他运动程序控制机械臂。将 `SESSION` 换成上一步的路径：

```bash
SESSION='outputs/<实际会话目录名>'
python scripts/auto_calibrate.py --session "$SESSION" --samples 50 --execute
```

程序先根据轨迹中的位置和朝向覆盖选点，再执行自动采集；默认 `--method hybrid`，无需另外指定。只想生成计划时去掉 `--execute`，此时不连接设备、不运动。

也可以先选点再执行已保存计划：

```bash
python scripts/select_samples.py --session "$SESSION" --samples 50
# 将 PLAN 换成上一条命令打印的计划文件路径
PLAN='outputs/<会话>/plans/<计划>/plan.json'
python scripts/auto_calibrate.py --plan "$PLAN" --execute
```

实际执行顺序如下：

1. 从示教 home 正向、或示教终点反向回放。当前位置不接近任一端点时拒绝启动，不自动从任意位置返回 home。
2. 使用 `NRT_JOINT_POSITION`，50 Hz 发送关节位置和目标速度；保留示教路径中间路点，只在采样点和端点停车，不直接连接相隔很远的采样点。
3. 到点后连续停稳 1 秒，丢弃 3 帧，再保存 10 帧原生 RGB 和每帧对应的**实际法兰位姿**。无论标记是否可见，都先保存原始数据。
4. 全部采完后停止机械臂并关闭硬件连接，再统一提取图像测量、筛选、求解、验证和运行 r3kit 对照。
5. 打印 `CAPTURE=...` 和 `RESULT=...`。失败或不达标时保留数据和报告；标定不达标不会触发自动补采。

`--samples 50` 指访问 50 个点，不保证筛选后有 50 个有效点。只采集并做历史四角点筛选、暂不计算时加 `--capture-only`；之后可用离线命令运行默认混合流程。

**配置与速度：**会话和计划保留创建时的配置，不会自动采用磁盘上的新设置。当前配置 `replay_speed_scale: 3.0`，有效关节速度上限 0.36 rad/s、加速度上限 1.80 rad/s²、指令曲线 jerk 上限 27 rad/s³；初始端点对齐采用基准限制，停稳和采图时间不缩短。旧配置缺少倍率时按 1 倍。需要对旧轨迹采用当前配置时，重新选点：

```bash
python scripts/auto_calibrate.py --session "$SESSION" \
  --config configs/d415_rizon4r.yaml --samples 50 --execute
```

启动时会打印实际倍率。回放遵循示教路径及有限平滑偏差，但不是碰撞检测器。跟踪异常、失联或 Ctrl+C 会请求停止并保存已采数据；若停在轨迹中间，需要手动回到示教端点后再开始新的采集批次，不支持从中间自动续跑。

### 4. 用现有数据离线重算

```bash
RUN='outputs/<会话>/captures/<采集批次>'
python scripts/calibrate.py --run "$RUN"
```

默认从原始 PNG 重新提取混合测量，计算并保存新的 `calibration/<时间>_hybrid/`。**不连接相机或机器人、不重新采集、不覆盖旧结果**，也不需要之前的排障缓存。原计划中的训练/测试归属保持不变。

历史四角点方法仍可显式运行：

```bash
# 使用采集配置中的角点流程，固定 K/D 的两阶段基线或四角点联合优化
python scripts/calibrate.py --run "$RUN" --method baseline
python scripts/calibrate.py --run "$RUN" --method joint

# 对早期 SUBPIX 配置的数据，用恢复后的 r3kit 角点重新做四角点联合优化
python scripts/reprocess_calibration.py --run "$RUN"
```

`baseline` 是本项目的两阶段实现；默认混合流程保存的 `r3kit_*.json` 才是调用固定提交的上游代码所得对照，两者不要混称。

### 5. 检查角点和结果

```bash
RESULT="$RUN/calibration/<实际时间>_hybrid"
python scripts/visualize_corners.py --run "$RUN" --processed "$RESULT"
python scripts/evaluate.py --result "$RESULT/result.json"
```

打开生成的 `diagnostics/corner_review/index.html`，可逐点、逐帧查看原图及检测结果：绿色为所选外角点、青色为内部实测点、橙色叉号为固定模型的预测位置。下拉框默认选择当前混合模型，并保留 r3kit 和历史模型供对照。原始图像不被修改。

`evaluate.py` 使用结果对应的冻结测量重新验证，不重新拟合模型。可加 `--references <独立参考测点.json>` 检查毫米误差，参考文件格式见 [`docs/design.md`](docs/design.md)。

## 技术路线

### 坐标、已知量与未知量

约定 `T_A_B` 将 B 坐标变换到 A：`p_A = R_A_B p_B + t_A_B`，平移统一用米。

- W：Flexiv WORLD；E：实际法兰；C：原生 RGB 光学坐标；M：标记中心。
- 已知 `G_i = T_W_E,i`：实际机器人反馈，**不是目标位姿，也不是 TCP 位姿**。
- 已知厂家 K、80 mm 标记尺寸、编码方向和内部网格坐标。
- 求解 `X = T_W_C`、`Y = T_E_M`，以及共享的径向畸变 `k1/k2`。

几何关系为 `G_i Y = X T_C_M,i`；对标记上的三维点 `P_j`，图像预测为：

```text
u_pred(i,j) = project(K, D, inverse(X) @ G_i @ Y @ P_j)
残差 = u_pred(i,j) - 图像实测像素坐标(i,j)
```

所有姿态共享同一个 X/Y/D。优化时不为每张图单独增加板姿态变量。

### 图像测量与质量筛选

首先恢复 r3kit 的前端：灰度图 → `CORNER_REFINE_APRILTAG`（窗口 5）→ `cornerSubPix`（5×5，30 次，epsilon 0.001）。这避免了早期仅使用 SUBPIX 时在带安装框标记上选错轮廓的问题，三个真实失败样例已作为回归测试保留。

在正确编码四角基础上提取两类点：

- **外角点：**沿黑白边界的局部梯度定位、拟合边缘直线，再取交点。边缘细化不通过图像质量检查时，记录原因并回退原 r3kit 角点。
- **内部点：**按 ID 0 的编码网格找到图案交界处，局部亚像素细化。单应矩阵只用于初始定位和图像质量筛选；真正参与求解的是实测内部像素，不是单应矩阵生成的虚拟角点。

单帧标记需离图像边缘至少 8 px、最短边至少 35 px；每姿态至少 6 帧有效。每个内部点也需至少 6 帧支持，每姿态至少 12 个内部点；同一网格点跨帧取中位数。当前数据每姿态有 15–27 个内部点。质量筛选不依据机器人预测残差，不为了降低误差而事后删掉某个测试点。

### 联合优化与验证

先在训练组中使用四角点、多初值和 IPPE 候选建立 X/Y 初值，再进行混合优化：

- 每姿态内部点与四个外角点**各占 50% 总权重**，每个姿态总权重相同，避免内部点数量多的姿态支配求解。
- 使用逐坐标 1 px Huber 稳健损失；X/Y 各 6 个自由度，加 k1/k2，共 14 个参数。
- **厂家 K 固定，标记边长固定 80 mm**；不同时放开全部内参来追求更低的训练误差。
- 检查收敛、边界、局部雅可比秩，并保守继承初值求解的退化/歧义状态。局部满秩不等于全局唯一或毫米准确。
- 训练/测试成员由采样计划确定；另在有效训练姿态中做五个连续分组验证，每折初值也只用该折训练数据。
- 测试时冻结 X/Y/D，结合实际机器人位姿预测图像点；外角点和内部点分别评分，不能换评价点后直接比较数字。

像素 RMS 是**预测位置与图像实测位置之间的二维距离误差**，不是机械臂末端误差，也不能直接等同于毫米误差。

### 与 r3kit 原方法的区别

r3kit 原方法先用四角点、固定 K/零畸变的 `calibrateCamera` 估计每张图的板姿态，再用 Shah 求两个固定变换。默认混合方法在此基础上，直接用跨姿态的内部点和外角点约束共享外参，并估计共享 k1/k2。

每次默认标定都会保存两个上游对照：

- `r3kit_same_burst.json`：同一有效姿态集合、同一 r3kit 角点连拍聚合，调用上游原求解器。
- `r3kit_full_first_frame.json`：每个共同训练姿态固定取第 0 帧，调用上游图像检测到求解的完整流程；首帧失败如实记录，不替换为更好的帧。

两者都在相同的测试外角点和内部点上评价。r3kit 自带 `error[0]` 允许每张训练图独立拟合板姿态，不能直接与固定外参测试误差比较。上游源码固定为 [`95ce4caf`](https://github.com/dadadadawjb/r3kit/tree/95ce4caf1cb2a457b65264ba57981dc54ac6daf4)，放在 `camera_calibration/reference_r3kit/`；仅调整导入命名空间，保留 MIT 许可证和源文件哈希。

## 当前验证结果与使用边界

2026-11-19 的已有数据：50 个采样姿态、500 张图；筛选后 40 个有效姿态，其中 34 个训练、6 个测试。混合方法结果：

- 原测试集：外角点 **1.237 px**，内部点 **0.748 px**。
- 训练内五组验证：外角点 **1.944 px**，内部点 **1.504 px**。
- 求解成功；外角点仍超过原定 **1 px** 阈值；物理精度 `unverified`。

相对原边缘外角点＋径向畸变方案，混合方法的测试误差由外角点/内部点 1.278/0.897 px 改善至 1.237/0.748 px；五组验证分别由 1.940/1.581 px 变为 1.944/1.504 px。改善主要体现在内部点，外角点分组验证基本持平。这批图像已反复用于排障和方法比较，**不能视作全新独立验收数据**。

`status_solve`、`status_internal_validation` 和 `status_physical_accuracy` 分别表示求解状态、固定测试集像素验收和独立物理精度。自动脚本在未通过求解/像素验收时会提示错误，但已经写出的原始数据和结果仍可查看。

## 结果怎么使用

`result.json` 包含：

- `camera.K`：固定的厂家内参矩阵。
- `camera.model / coeffs`：联合拟合的 **forward Brown** 模型，系数顺序 `[k1,k2,p1,p2,k3]`，后三项固定为零。
- `camera_factory`：采集时原始模型和参数；本机为 inverse Brown、全零畸变。拟合后的 D 不能当作 SDK 的厂家 inverse Brown 参数使用。
- `T_W_C`：RGB 相机到 WORLD；`T_C_W` 为逆变换；`T_E_M` 为标记到法兰。
- 若配置 `robot.T_base_world`，额外导出 `T_B_C = T_base_world @ T_W_C` 和逆变换。

使用时将该外参与 `camera` 中的 K/D 配套使用。WORLD 是否等同于现场基座坐标需要确认；当前配置没有填写 `T_base_world`，结果明确标为 WORLD。RGB 外参不直接适用于 depth 坐标；深度测量还涉及 RGB/depth 流间外参和深度本身的误差，本项目当前没有校正深度精度。

## 文件组织

```text
camera_calibration/
├── README.md
├── pyproject.toml / requirements-lock.txt
├── configs/d415_rizon4r.yaml
├── camera_calibration/          # 可复用算法、数据、设备适配器
│   ├── features.py             # 外边缘和内部实测点
│   ├── hybrid.py               # 默认离线处理、分组验证和 r3kit 对照
│   ├── multipoint.py           # 混合像素优化
│   └── reference_r3kit/        # 固定版本上游代码与许可证
├── scripts/                    # 正常使用的入口
│   └── tests/                  # 测试、合成数据和排障实验；fixtures 是回归样例
├── docs/design.md              # 实现细节及独立测点格式
├── docs/experiments.md         # 历史实验入口及缓存要求
├── docs/archive/               # 前期方案快照，不作为当前使用说明
└── outputs/                    # 本地轨迹、计划、原图、结果；不提交 Git
```

输出数据按时间和相机序列号组织：

```text
outputs/<时间>_cam-<序列号>_session/
├── manifest.json
├── trajectory/states.jsonl                 # 示教实际反馈
├── plans/<时间_cam-序列号>/plan.json        # 采样目标、路径和冻结配置
└── captures/<时间_cam-序列号>/
    ├── manifest.json                      # 实际设备、流参数和采集配置
    ├── replay/commands.jsonl               # 回放指令
    ├── replay/states.jsonl                 # 全程实际反馈
    ├── samples/0000/
    │   ├── target.json / actual.json       # 目标位姿与实际反馈/误差
    │   ├── frames.jsonl                    # 逐帧实际反馈及时间
    │   └── images/000.png                  # 原始 RGB，包含不可见标记的帧
    ├── raw_samples.json
    ├── calibration/<时间>_hybrid/
    │   ├── result.json / validation.json / report.html
    │   ├── cross_validation.json / comparison.json
    │   ├── r3kit_same_burst.json / r3kit_full_first_frame.json
    │   ├── observations.json               # r3kit 原角点聚合
    │   ├── outer_observations.json / internal_observations.json
    │   ├── detections/ / edge_detections/ / internal_detections/
    │   ├── filter_report.json              # 全部计划点的接受/拒绝原因
    │   └── protocol.json / input_hashes.json
    └── diagnostics/corner_review/index.html
```

历史四角点入口可能另在采集目录写 `observations.json`、`filter_report.json` 以及样本目录下的 `detections.json`；默认混合方法的对应文件位于自己的结果目录。失败会保存诊断；原始日志损坏不会被悄悄跳过。

## 开发与复现

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/tests/synthetic_demo.py
python scripts/tests/observability_probe.py
```

本机最近完整测试为 **55 passed**，包括实际图像角点回归、合成几何恢复、r3kit 坐标转换、固定分组、原始数据保护以及关闭设备后才运行离线计算的流程测试。合成结果只验证软件，不证明实体设备精度。实验脚本说明见 [`docs/experiments.md`](docs/experiments.md)。

Git 提交源码、配置、文档、依赖版本和三个必要的角点回归图像；`outputs/`、虚拟环境、缓存和构建产物留在本地。克隆仓库后可运行测试，但不会自带完整现场轨迹、500 张采集图像或现场标定结果，需要另行取得原始采集目录或按上述流程采集。
