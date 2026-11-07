# 末端运动控制：参考论文与开源项目

检索日期：2026-11-06。这里列出与现有方案相关的研究材料；项目已加入约束 DLS 实验模式，并未集成下列外部项目。

当前系统的基线是手柄给出平移和转动意图，在 MuJoCo 模型上运行位姿闭环，并使用 DLS、位置优先或实验性六维 SDLS 求解关节参考。以下资料对应当前系统最需要对照的部分。

## 1. Selectively Damped Least Squares（SDLS）

- [Buss 和 Kim 的原论文](https://math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/SdlsPaper.pdf)；[作者提供的说明和示例](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/)。
- **相关之处**：与本项目的位置 SDLS 直接相关，按雅可比的奇异方向限制更新，并讨论不可达目标的目标距离限制。适合核对公式和设计奇异位形、不可达目标对照试验。
- **边界**：论文主要处理末端位置目标；本项目六维位置加姿态的 SDLS 是实验扩展，不能视为论文已验证的结果。论文结果也不直接覆盖本项目的碰撞、执行器和真实机器人性能。

## 2. Mink：MuJoCo 差分逆运动学

- [GitHub 项目](https://github.com/kevinzakka/mink/)；[官方文档](https://kevinzakka.github.io/mink/index.html)；[任务与约束示例](https://github.com/kevinzakka/mink/blob/main/docs/tutorial/tasks_and_limits.rst)。
- **相关之处**：直接使用 MuJoCo 模型；每周期将末端位姿、姿态偏好等任务与关节位置、速度和指定几何对的碰撞约束组合成二次规划。适合在相同模型和输入下，与当前逐步求解后再限幅的方法比较。
- **边界**：碰撞几何对、限值等需要按模型明确配置；差分 IK 是局部方法，不保证找到全局可达路径。上述文档所列限制不等于完整的加速度、驱动器和硬件安全系统。QP 引入求解器依赖，也需要在目标笔记本上实测耗时。

## 3. SNS：关节约束下的零空间饱和

- [Flacco、De Luca 和 Khatib 的论文](https://www.diag.uniroma1.it/~labrob/pub/papers/ICRA12_RedundancySNS.pdf)。
- **相关之处**：研究冗余机械臂的差分逆运动学，同时考虑关节位置、速度和加速度边界；在关节饱和时重新利用剩余自由度，并在无法完整执行任务时缩放任务。适合对比现有各关节单独限幅产生的末端轨迹偏差。
- **边界**：自由度和任务维数影响可用冗余；六轴机械臂执行满秩六维位姿任务时通常没有连续零空间。还需单独评估奇异位形、碰撞与控制器实际制动。

## 4. Ruckig：在线轨迹生成

- [RSS 2021 论文](https://www.roboticsproceedings.org/rss17/p015.pdf)；[GitHub 项目](https://github.com/pantor/ruckig)。
- **相关之处**：从当前关节位置、速度、加速度向目标状态生成满足速度、加速度和加加速度（jerk）约束的轨迹。适合对比摇杆回中、松开使能、方向反转时的平滑性和停止距离。
- **边界**：这是关节轨迹生成器，不负责求解末端 IK 或规划避碰；真实硬件需要使用其实际约束值，组合后的末端轨迹、停止距离和响应延迟仍需验证。

## 5. MoveIt Servo：末端速度遥操作参考

- [官方实时伺服教程](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)。
- **相关之处**：接受末端速度或位姿命令，包含关节约束、运动平滑，以及接近奇异位形或碰撞时的减速功能。其面向连续遥操作的设计可用于比较输入、奇异性提示和停止行为。
- **边界**：使用 ROS/MoveIt 生态，接入依赖和配置与当前独立 MuJoCo 程序不同；可选碰撞检查在接近障碍时减速，并不等价于全局避障规划。

## 6. MuJoCo Menagerie：多机型测试模型

- [官方模型仓库](https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md)。
- **相关之处**：包含多种机械臂与不同自由度的 MuJoCo 模型，可用于覆盖六轴、七轴和不同夹爪的适配测试。
- **边界**：每个模型应查看各自许可证；模型能够加载不代表真实机器人的零点、工具坐标、关节限值、驱动响应或碰撞几何已校准。

后续若要比较，应在相同模型、初始姿态、输入轨迹和限值下测量位置/姿态误差、轨迹偏差、关节限值、奇异位形表现、停止距离和每周期耗时，再结合实际手柄操作延迟判断。这些外部项目尚未集成或横向实测；项目内约束 DLS 与原 DLS 的离线对照见 `../optimized_algorithm/README.md`。
