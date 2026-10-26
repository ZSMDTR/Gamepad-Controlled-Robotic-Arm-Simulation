# Gamepad-Controlled Robotic Arm Simulation

用普通手柄控制末端 **XYZ 位置**，通过 SDLS / DLS 逆运动学计算六个关节的动作。

第一版是本地运动学实验台：Python + NumPy 求解，浏览器显示三维机械臂并读取手柄。没有云端服务、前端 CDN 或机械臂驱动。

## 快速开始

需要 Python 3.10+ 和支持 Gamepad API 的现代浏览器，建议 Chrome / Edge。

```bash
git clone https://github.com/ZSMDTR/Gamepad-Controlled-Robotic-Arm-Simulation.git
cd Gamepad-Controlled-Robotic-Arm-Simulation
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m arm_teleop
```

打开 [本地仿真页面](http://127.0.0.1:8000)。端口被占用时：`python -m arm_teleop --port 8001`。

## 操作

| 输入 | 动作 |
| --- | --- |
| 左摇杆左右 / 上下 | 基座坐标系 X / Y 平移 |
| 右摇杆上下 | Z 升降 |
| 按住 LB | 使能手柄运动；松开立即暂停仿真 |
| WASD / QE | 无标准手柄时，键盘 XY / Z；Q 升、E 降 |
| 按住空格 | 使能键盘运动 |
| 拖动画面 / 滚轮 | 旋转视角 / 缩放 |
| DLS / SDLS、姿态滑块 | 比较求解方法及姿态偏好强度 |

先点击机械臂画面使页面获得焦点。手柄可能需要按一次实体按键才被浏览器识别。使用浏览器 `standard` 映射；非标准手柄提示后回退键盘。标准手柄连接时优先使用手柄。

切出窗口、手柄断开或指令超过 250 ms 未更新会停止输入。摇杆回中时冻结关节并重置目标，防止松手后继续追赶积累误差。实体手柄尚需按设备验证；网页和键盘路径可先独立运行。

## 算法结构

```mermaid
flowchart LR
    A[手柄或键盘] --> B[死区与 XYZ 速度映射]
    B --> C[积分目标及超前限幅]
    C --> D[SDLS / DLS]
    F[关节反馈与 FK / 位置雅可比] --> D
    D --> E[零空间姿态偏好]
    E --> G[速度、加速度与关节限位]
    G --> H[运动学仿真]
    H --> F
```

- 默认六旋转关节通用模型，长度单位 m、角度 rad，默认控制周期 10 ms。它不代表任何具体商品机械臂。
- 每周期只计算一次 3×6 雅可比的 SVD，并在求解、姿态投影和奇异性指标之间复用。
- SDLS 实现 Buss / Kim 论文公式 (3)、(4) 的单 TCP 旋转关节版本；采用论文的硬限幅，而非作者下载示例中的平滑缩放。
- 目标超前距离固定限制为 0.06 m，是本原型的工程选择，未完整复现论文的自适应目标距离策略。
- DLS 使用统一阻尼 λ=0.04 m/rad 作基线；并非论文全部实验的复现。
- 肘部滑块调节靠近初始姿态的零空间偏好，不是任意肘部位置/轨迹控制器。末端朝向不约束，夹爪不执行开合。

详见 [算法说明](docs/algorithm.md) 与 [验证记录](docs/validation.md)。

## 测试与性能对比

```bash
python -m unittest discover -s tests -v
python scripts/benchmark.py --steps 2000
python scripts/benchmark.py --steps 2000 --json
```

测试覆盖数值雅可比、DLS 手算结果、SDLS 奇异性、姿态投影、连续输入约束、松手暂停、输入校验、超时及 HTTP 接口。benchmark 使用相同确定性输入，输出均值、99 分位、最大耗时和位置误差；误差参照**受限目标**。

## 当前边界

1. **运动学仿真**：没有动力学、摩擦、碰撞检测、接触力、关节驱动器或 sim-to-real 标定。
2. **局部求解**：无法保证绕障、全局可达或到达所有目标；不可达输入会触发目标限幅。
3. **软实时循环**：100 Hz 是调度目标，Python、浏览器与系统调度不提供硬实时保证；页面耗时也不是端到端延迟。
4. **暂停与硬限位例外**：连续运动有速度/加速度限制；立即暂停及最终硬限位可能打断加速度连续性。这是仿真语义，不能直接用作真实制动控制。
5. **层级约束近似**：零空间项只保证线性化时的末端解耦；后续逐关节限幅会改变任务方向和优先级。严格约束应升级为 QP。

下一阶段应先指定真实机械臂型号/URDF、关节约束与反馈接口，再引入碰撞与动力学验证，随后才增加低速硬件适配。

## 项目结构

```text
arm_teleop/
  kinematics.py     # 六关节几何、FK、位置雅可比
  solvers.py        # DLS、SDLS、数值零空间投影
  controller.py     # 目标、姿态、限幅与仿真状态
  server.py         # 本地网页/API、100 Hz 调度、指令超时
  static/           # 中文界面、Canvas、Gamepad API
tests/              # 数值及接口验证
scripts/benchmark.py
docs/
```

## 参考

- Buss, S. R. & Kim, J.-S. (2005), *Selectively Damped Least Squares for Inverse Kinematics*, Journal of Graphics Tools 10(3), 37–49. [作者页面及原文](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/) · [DOI](https://doi.org/10.1080/2151237X.2005.10129202)
- Lynch & Park, *Modern Robotics*, [奇异性与雅可比](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/)

实现为独立编写，仓库不包含论文全文或作者的 C++ 源码。
