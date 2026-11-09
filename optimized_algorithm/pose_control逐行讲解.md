# pose_control.py 逐行讲解

本文对应同目录下 `pose_control.py` 的 1–346 行，按源码顺序解释。行号是撰写时的源码行号；如果以后修改源码，行号可能变化。空行仅用于排版，不单独解释；同一条语句跨多行时合并说明。

## 先看整体：这份代码在做什么

这是**机械臂末端遥操作控制器**，不是 π0.5 模型。它接收世界坐标系下的平移、旋转指令和机械臂当前状态，计算发送给关节的参考位置。

流程：输入方向 → 积累末端目标位姿 → 计算与当前位姿的误差 → 利用雅可比矩阵求关节速度 → 限制速度、加速度和参考位置 → 输出 `q_ref`。

当前 MuJoCo 原生窗口通过 `native_control.py` 选择 `constrained_dls` 模式；本类本身的默认模式是 `dls`。其他模式的代码也会逐行讲解，但不表示它们都在当前窗口同时运行。

## 阅读前先认几个符号

| 名称 | 代表什么 |
|---|---|
| `q` | 当前测量到的各关节位置，通常是角度（弧度）；具体取决于关节类型 |
| `qvel` | 测量到的关节速度；这份 `step` 当前只检查它的格式，没有把它用于解算 |
| `tcp` | 工具中心点，也就是机械臂末端参考点的位置，单位米 |
| `rotation` | 末端当前姿态，使用 3×3 旋转矩阵表示 |
| `jacobian` 或 `j` | 雅可比矩阵，近似描述关节速度如何变成末端线速度与角速度 |
| `translation` | 用户输入的三维平移方向，各分量限制在 −1 到 1 |
| `rotation_input` | 用户输入的三维旋转方向，各分量限制在 −1 到 1 |
| `dt` | 本次控制计算的时间步，单位秒 |
| `q_ref` | 给下游执行器的关节位置参考，并非机械臂已经达到的真实位置 |
| `@` | Python/NumPy 中的矩阵乘法；`*` 通常是逐元素相乘 |
| `np` | NumPy 数值计算库的别名 |

举一个位置目标更新的例子：若使用当前原生窗口设置的最大平移速度 0.05 米/秒，输入为 `[1, 0, 0]`、`dt=0.01` 秒，那么本次目标位置沿世界 X 方向增加 0.0005 米，即 0.5 毫米。这只是目标更新，机械臂实际移动还取决于解算、限制、接触和执行器。

重要限制：这个文件约束的是参考指令，不保证仿真或实机的实际速度、碰撞安全或精确跟踪；运行真实设备仍需要相应的硬件安全措施。

# pose_control.py 第 1—152 行逐句讲解

下面保留原始代码的行号和语句；空行及仅用于结束多行字符串的引号不重复解释。缩进影响代码层级，阅读时请结合原文件。

**第 1 行**：`"""Model-independent, world-frame pose teleoperation using differential IK.`<br>
文件开头的多行说明文档：这是一个使用微分逆运动学的世界坐标系位姿遥操作控制器。模型无关指核心求解接受通用状态和雅可比，并不意味着任意机械臂无需配置即可使用。

**第 3 行**：`DLS is the 6D baseline. The experimental SDLS extends the original position`<br>
DLS 是六维位置加姿态控制的基础算法；SDLS 是实验性扩展：把位置和经过尺度调整的旋转看成两个三维块，并非原论文现成的六维求解器。

**第 4 行**：`method to a scaled six-dimensional pose task using two three-dimensional`<br>
接上行：六维姿态求解使用两个三维块；并非原论文直接提供的姿态求解器。

**第 5 行**：`blocks; it is not the original paper's pose solver. The position-priority mode projects angular motion`<br>
位置优先模式将角运动投影到位置雅可比的瞬时零空间，以优先保持平移；后续普通模式的逐关节裁剪可能改变任务优先级。

**第 6 行**：`into the exact instantaneous position nullspace. For the baseline modes,`<br>
接上行：角运动先投影到位置任务的瞬时零空间，基线模式还会在求解后裁剪关节值。

**第 7 行**：`joint clipping after solving can change task priorities. The optional`<br>
接上行：关节裁剪可能破坏原有优先级；可选约束 DLS 把约束放进求解过程。

**第 8 行**：`constrained DLS mode incorporates velocity and reference bounds in a convex`<br>
约束 DLS 使用凸二次优化，把关节速度和参考位置等边界纳入求解。

**第 9 行**：`quadratic solve; conflicting bounds relax the reference-lead constraint and`<br>
如果边界冲突，则放宽“参考不能领先当前状态太远”的约束，避免参考位置突然跳变。

**第 10 行**：`advance the reference without a jump. All limits apply to references, not a guarantee about physical robot motion.`<br>
这些限制约束的是控制器产生的参考指令，并不保证真实机械臂实际运动一定安全。

**第 12 行**：`from __future__ import annotations`<br>
启用延迟解析类型注解；这是 Python 兼容性设置，不直接控制机器人。

**第 14 行**：`import time`<br>
导入 time 模块，供文件后面的性能计时使用。

**第 15 行**：`import numpy as np`<br>
导入 NumPy 并简称 np，后面用它处理向量、矩阵和线性代数。

**第 18 行**：`def _vec(value, size, name):`<br>
定义内部辅助函数：要求 value 是长度为 size 的向量；name 用于构造报错。

**第 19 行**：`result = np.asarray(value, dtype=float)`<br>
把输入转换为浮点 NumPy 数组，保存为 result。

**第 20 行**：`if result.shape != (size,) or not np.isfinite(result).all():`<br>
检查形状是否严格为一维且长度正确，同时检查每个元素是否为有限数，排除 NaN 和无穷大。

**第 21 行**：`raise ValueError(f"{name} must be a finite vector of length {size}")`<br>
如果检查失败，抛出带变量名和所需长度的错误，避免非法数据进入控制计算。

**第 22 行**：`return result`<br>
返回通过验证的浮点向量；本函数检查形状和有限性，并不限制数值大小。

**第 25 行**：`def _bounded(vector, bound):`<br>
定义向量限幅函数：如果向量长度超出 bound，就按比例缩小。

**第 26 行**：`length = float(np.linalg.norm(vector))`<br>
计算向量的欧氏长度，也就是各分量平方和的平方根，并转成 Python 浮点数。

**第 27 行**：`return vector * min(1.0, bound / length) if length > 0 else vector.copy()`<br>
长度大于零时，乘以 min(1, bound/length)：未超限不变，超限按比例缩短并保持方向；零向量返回副本，避免除零。这里不是分别裁剪每一个轴。

**第 30 行**：`def rotation_exp(rotvec):`<br>
定义旋转指数映射：把旋转向量转换为 3×3 旋转矩阵。

**第 31 行**：`"""SO(3) exponential; rotation vector in radians."""`<br>
文档指出输入属于 SO(3) 旋转计算，旋转向量长度使用弧度。

**第 32 行**：`v = _vec(rotvec, 3, "rotation vector")`<br>
用 _vec 验证 rotvec 是有限的三维向量；方向对应旋转轴，长度对应旋转角。

**第 33 行**：`angle = float(np.linalg.norm(v))`<br>
计算旋转向量长度，即旋转角 angle，单位是弧度。

**第 34 行**：`k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])`<br>
由 v 构造反对称矩阵 K；对任意向量 a，K @ a 等于 v 与 a 的叉乘。

**第 35 行**：`if angle < 1e-8:`<br>
判断旋转角是否极小；直接计算后面的 sin(angle)/angle 等表达式可能损失数值精度。

**第 36 行**：`return np.eye(3) + k + .5 * k @ k`<br>
极小角度时使用矩阵指数的二阶近似 I + K + K²/2；np.eye(3) 是三维单位矩阵，@ 是矩阵乘法。

**第 37 行**：`return np.eye(3) + np.sin(angle) / angle * k + ((1 - np.cos(angle)) / angle**2) * (k @ k)`<br>
一般情况使用罗德里格公式 I + (sinθ/θ)K + ((1−cosθ)/θ²)K²，计算对应旋转矩阵。

**第 40 行**：`def rotation_log(matrix):`<br>
定义旋转对数映射：把旋转矩阵转换成最短旋转向量。

**第 41 行**：`"""Shortest world-frame rotation vector, stable near zero and pi.`<br>
函数文档说明输出最短旋转向量，并特别处理接近零和接近 π 的数值稳定性。具体属于哪个坐标系取决于传入矩阵怎样构造。

**第 43 行**：`At exactly pi either axis sign represents the same rotation; the chosen`<br>
文档提醒恰好转动 π 弧度时，轴的正反方向表示相同旋转，因此符号可能在这个固有不连续点发生变化。

**第 44 行**：`sign can change at that intrinsic discontinuity.`<br>
接上行：该符号跳变是旋转表示本身的问题，不代表算法必然让机械臂跳动。

**第 46 行**：`r = np.asarray(matrix, dtype=float)`<br>
把输入矩阵转为浮点 NumPy 数组。

**第 47 行**：`if r.shape != (3, 3) or not np.isfinite(r).all():`<br>
检查它是否为 3×3 且所有数均有限。

**第 48 行**：`raise ValueError("rotation matrix must be finite and 3x3")`<br>
条件不满足时抛出错误。

**第 49 行**：`if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:`<br>
进一步检查 RᵀR 是否近似单位矩阵，以及行列式是否接近 1；这是验证它近似为合法三维旋转矩阵。

**第 50 行**：`raise ValueError("rotation matrix must be orthonormal with determinant 1")`<br>
如果不是合法旋转矩阵，抛出明确错误，而不是继续算出错误姿态。

**第 51 行**：`cosine = np.clip((np.trace(r) - 1) * .5, -1., 1.)`<br>
利用旋转矩阵的迹计算 cosθ=(trace(R)−1)/2；clip 把结果限制在 [−1,1]，避免浮点误差令 arccos 无定义。

**第 52 行**：`angle = float(np.arccos(cosine))`<br>
对 cosθ 求反余弦，得到主值角度 θ，范围为 0 到 π。

**第 53 行**：`skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])`<br>
从 R−Rᵀ 的非重复元素中提取三维向量；这个向量包含旋转轴方向和 sinθ。

**第 54 行**：`if angle < 1e-7:`<br>
如果角度非常接近零，就使用小角度分支。

**第 55 行**：`return .5 * skew`<br>
小角近似 sinθ≈θ，因此旋转向量可近似写成 skew/2，避免除以接近零的 sinθ。

**第 56 行**：`if np.pi - angle < 1e-5:`<br>
如果角度非常接近 π，则使用另一种稳定的求轴方法。

**第 57 行**：`# Principal eigenvector of the symmetric component gives the axis.`<br>
代码注释：通过对称部分的主特征向量求旋转轴。

**第 58 行**：`_, axes = np.linalg.eigh((r + r.T + 2 * np.eye(3)) * .25)`<br>
对 (R+Rᵀ+2I)/4 做对称矩阵特征分解；忽略特征值，只保存按特征值排列的特征向量矩阵 axes。

**第 59 行**：`axis = axes[:, -1]`<br>
取最后一列，即对应最大特征值的特征向量，作为旋转轴。

**第 60 行**：`if np.dot(axis, skew) < 0:`<br>
计算旋转轴与 skew 的点积；为负时说明选择的轴方向与 skew 提供的方向相反。

**第 61 行**：`axis = -axis`<br>
把轴乘以 −1 调整符号。恰在 π 时两种轴符号本来就等价。

**第 62 行**：`return axis * angle`<br>
旋转轴乘以旋转角，得到接近 π 时的旋转向量。

**第 63 行**：`return angle / (2 * np.sin(angle)) * skew`<br>
一般情况直接使用 θ/(2sinθ)×skew，还原最短旋转向量。

**第 66 行**：`def _dls(j, velocity, damping):`<br>
定义阻尼最小二乘求解函数；j 是雅可比，velocity 是期望任务速度，damping 是阻尼系数。

**第 67 行**：`u, s, vh = np.linalg.svd(j, full_matrices=False)`<br>
对 j 做精简奇异值分解 J=UΣVᵀ，得到奇异方向和奇异值 s。

**第 68 行**：`return vh.T @ ((s / (s * s + damping * damping)) * (u.T @ velocity))`<br>
计算 V diag(s/(s²+阻尼²)) Uᵀ velocity，得到关节速度。阻尼可减少接近奇异位形时的小奇异值放大，但会牺牲部分跟踪精度，也不能消除所有机械限制。

**第 71 行**：`def _projector(j):`<br>
定义根据雅可比计算零空间投影矩阵的函数。

**第 72 行**：`_, s, vh = np.linalg.svd(j, full_matrices=False)`<br>
对 j 做精简奇异值分解；这里只使用奇异值 s 和右奇异向量 vh。

**第 73 行**：`cutoff = max(1e-10, s[0] * 1e-8) if s.size else 1e-10`<br>
构造识别非零奇异值的数值阈值：取 1e−10 与最大奇异值的 1e−8 倍中的较大者；无奇异值则取 1e−10。

**第 74 行**：`basis = vh[s > cutoff]`<br>
选择大于阈值的奇异值对应的右奇异向量，作为雅可比行空间的正交基。

**第 75 行**：`return np.eye(j.shape[1]) - basis.T @ basis`<br>
返回 P=I−basisᵀbasis。P 可把关节运动投影到数值意义下的位置零空间，使其对相应任务的瞬时影响接近零；效果受阈值、线性近似和后续裁剪影响。

**第 78 行**：`def _box_dls(j, target, damping, lower, upper, center):`<br>
定义带上下界的阻尼最小二乘函数：j 是任务雅可比，target 是目标速度，lower/upper 是逐关节边界，center 是阻尼项偏好的中心。

**第 79 行**：`"""Minimize damped task error with box bounds by a feasible active set.`<br>
多行文档概述：使用可行活动集方法，在盒形上下界内最小化阻尼任务误差。

**第 81 行**：`Returns (velocity, converged). The quadratic is strictly convex because`<br>
文档说明返回 (速度, 是否收敛)。在阻尼系数严格为正时，二次目标严格凸；迭代次数有上限。

**第 82 行**：`damping is positive. A bounded iteration count keeps control time finite.`<br>
接上行：限制求解迭代次数，避免控制周期无界等待；这不是实时性保证。

**第 84 行**：`h = j.T @ j + damping**2 * np.eye(j.shape[1])`<br>
计算二次优化的矩阵 H=JᵀJ+λ²I；其对应目标可写为 ||Jx−target||² + λ²||x−center||²，目标的整体系数不改变最优解。

**第 85 行**：`b = j.T @ target + damping**2 * center`<br>
计算右侧向量 b=Jᵀtarget+λ²center，后面梯度就是 Hx−b。

**第 86 行**：`x = np.clip(np.zeros(j.shape[1]), lower, upper)`<br>
以零关节速度为起点，并用 clip 裁入 lower、upper 范围，确保初始点可行。

**第 87 行**：`tol = 1e-10`<br>
设置求解器的数值容差 tol=1e−10。

**第 88 行**：`active = (x <= lower + tol) | (x >= upper - tol)`<br>
标记目前位于上下界附近的关节为活动约束 active，也就是暂时固定在边界上。

**第 89 行**：`fixed = np.abs(upper - lower) <= tol`<br>
标记上下界之差在容差内的关节为 fixed，这些变量近似没有自由空间。

**第 90 行**：`for _ in range(50 + 10 * j.shape[1]):`<br>
开始有上限的活动集迭代，次数为 50+10×关节数。

**第 91 行**：`gradient = h @ x - b`<br>
计算当前梯度 Hx−b；梯度反映增大各关节变量会如何改变目标值。

**第 92 行**：`free = ~active`<br>
取活动约束的逻辑非，得到目前可自由调整的关节。

**第 93 行**：`direction = np.zeros_like(x)`<br>
创建与 x 形状一致的零向量，准备保存本轮移动方向。

**第 94 行**：`if np.any(free):`<br>
只有存在自由变量时才求解自由子问题。

**第 95 行**：`direction[free] = np.linalg.solve(h[np.ix_(free, free)], -gradient[free])`<br>
抽取 H 中自由变量对应的行列，求解 H_ff × direction_f = −gradient_f；其余活动变量方向保持零。

**第 96 行**：`if np.max(np.abs(direction), initial=0.) <= tol:`<br>
如果方向所有分量都很小，表明当前活动集下已经不能明显改进，转去检查是否需要释放边界。

**第 97 行**：`# Release the active bound with the largest KKT violation.`<br>
注释说明下一步寻找 KKT 条件违反最大的活动边界；KKT 是判断受约束最优解的条件。

**第 98 行**：`violation = np.zeros_like(x)`<br>
创建全零 violation，准备记录每个活动边界的违反程度。

**第 99 行**：`at_lower = active & ~fixed & (x <= lower + tol)`<br>
找出位于下界、当前活动但不是固定变量的关节。

**第 100 行**：`at_upper = active & ~fixed & (x >= upper - tol)`<br>
同样找出位于上界、当前活动但不是固定变量的关节。

**第 101 行**：`violation[at_lower] = np.maximum(0., -gradient[at_lower])`<br>
在下界，如果梯度为负，向可行方向增大变量可降低目标，因此记录其违反程度 max(0,−gradient)。

**第 102 行**：`violation[at_upper] = np.maximum(0., gradient[at_upper])`<br>
在上界，如果梯度为正，向可行方向减小变量可降低目标，因此记录 max(0,gradient)。

**第 103 行**：`if np.max(violation, initial=0.) <= tol:`<br>
如果最大的违反程度也不超过容差，说明满足本求解器的数值终止条件。

**第 104 行**：`return x, True`<br>
返回当前可行解 x 和 True，表示活动集法已收敛。

**第 105 行**：`active[np.argmax(violation)] = False`<br>
否则释放违反最严重的边界，使这个变量下一轮可以移动。

**第 106 行**：`continue`<br>
跳过本轮余下计算，从新活动集重新开始下一轮。

**第 107 行**：`alpha = 1.`<br>
默认尝试完整牛顿步，步长 alpha=1。

**第 108 行**：`positive, negative = direction > tol, direction < -tol`<br>
分别标记移动方向明显为正或负的关节，避免对接近零的方向做除法。

**第 109 行**：`if np.any(positive):`<br>
如果有正方向分量，就检查到达各自上界前能移动多远。

**第 110 行**：`alpha = min(alpha, float(np.min((upper[positive] - x[positive]) / direction[positive])))`<br>
把 alpha 缩短到不会越过任何正方向变量上界的最大可行步长；float 把结果转成普通数值。

**第 111 行**：`if np.any(negative):`<br>
如果有负方向分量，就检查到达各自下界前能移动多远。

**第 112 行**：`alpha = min(alpha, float(np.min((lower[negative] - x[negative]) / direction[negative])))`<br>
同样进一步缩短步长，保证负方向变量不越过下界；分子和分母通常都是负数，因此比值非负。

**第 113 行**：`x = np.clip(x + max(0., alpha) * direction, lower, upper)`<br>
沿方向走 max(0,alpha) 步，再用 clip 防止浮点误差造成微小越界。

**第 114 行**：`if alpha < 1. - tol:`<br>
如果步长显著小于 1，说明移动被至少一个边界截住。

**第 115 行**：`active |= free & ((x <= lower + tol) | (x >= upper - tol))`<br>
把新碰到边界的自由变量加入活动集；|= 表示把原活动集与新条件合并。

**第 116 行**：`return x, False`<br>
若迭代次数耗尽，返回当前可行点和 False；调用方需要查看是否收敛，不能把它当作已证最优的解。

**第 119 行**：`def sdls_pose(jacobian, error, gamma_max=.2):`<br>
定义实验性的 SDLS 位姿求解函数；输入雅可比、任务空间误差增量，以及每个关节增量上限参数 gamma_max。

**第 120 行**：`"""Experimental scaled pose SDLS, extending the paper to 3D blocks.`<br>
文档说明这是扩展到三维分块的实验性缩放位姿 SDLS。

**第 122 行**：`The 6D version treats linear displacement and length-scaled rotation as`<br>
六维模式把线位移与按长度尺度调整后的旋转分别当作三维块；length_scale 由调用者预先选择和应用。

**第 123 行**：`two three-dimensional blocks; choose length_scale before calling. A 3D`<br>
三维模式采用单目标形式；error 和返回结果都是增量，不是速度。

**第 124 行**：`input has the original single-target definition. Error and the result`<br>
延续上行，明确这是启发式算法，而不是严格约束优化器。

**第 125 行**：`are increments, not velocities. This heuristic is not a constraint solver.`<br>
再次强调：不能因为函数内部有限幅，就认定它已满足所有关节位置和速度约束。

**第 127 行**：`j = np.asarray(jacobian, dtype=float)`<br>
把雅可比转换成浮点数组。

**第 128 行**：`e = np.asarray(error, dtype=float)`<br>
把任务误差转换成浮点数组。

**第 129 行**：`if j.ndim != 2 or j.shape[0] not in (3, 6) or j.shape[1] < 1 or e.shape != (j.shape[0],):`<br>
验证 j 为二维、行数为 3 或 6、至少有一个关节列，且 e 形状与任务维数相同。

**第 130 行**：`raise ValueError('SDLS requires a 3xn or 6xn Jacobian and matching increment')`<br>
维度不匹配时抛错，防止矩阵运算出错或结果无意义。

**第 131 行**：`if not np.isfinite(j).all() or not np.isfinite(e).all() or not np.isfinite(gamma_max) or gamma_max <= 0:`<br>
检查雅可比、误差和 gamma_max 均有限，且 gamma_max 大于零。

**第 132 行**：`raise ValueError('SDLS inputs must be finite and gamma_max positive')`<br>
条件不满足时抛出具体错误。

**第 133 行**：`u, s, vh = np.linalg.svd(j, full_matrices=False)`<br>
对雅可比进行精简奇异值分解，逐个分析任务空间与关节空间的对应方向。

**第 134 行**：`threshold = max(1e-10, float(s[0]) * 1e-8)`<br>
确定忽略数值近零奇异值的阈值，取绝对下限 1e−10 和最大奇异值的 1e−8 倍中的较大者。

**第 135 行**：`rho = sum(np.linalg.norm(j[start:start + 3], axis=0) for start in range(0, j.shape[0], 3))`<br>
对于 j 的每个关节列，分别计算各三维块的长度并求和，得到 rho；三维时只有一个块，六维时有两个块。

**第 136 行**：`result = np.zeros(j.shape[1])`<br>
创建长度为关节数的零向量 result，用于累加各奇异方向贡献。

**第 137 行**：`for i, sigma in enumerate(s):`<br>
逐一遍历各个奇异值 sigma 及其索引 i。

**第 138 行**：`if sigma <= threshold:`<br>
如果这个奇异值不大于阈值，说明该方向数值上过于接近奇异。

**第 139 行**：`continue`<br>
跳过该方向，避免除以极小 sigma 导致巨大关节增量。

**第 140 行**：`v = vh[i]`<br>
取出该方向的右奇异向量 v，它表示对应的关节空间方向。

**第 141 行**：`ni = sum(np.linalg.norm(u[start:start + 3, i]) for start in range(0, j.shape[0], 3))`<br>
计算 ni：把对应左奇异向量按每三个任务分量分块，分别求长度再相加。

**第 142 行**：`mi = float(np.abs(v) @ rho) / sigma`<br>
计算 mi=|v|·rho/sigma，估计该奇异方向下关节运动的尺度。

**第 143 行**：`bound = gamma_max * min(1., ni / mi) if mi > 0 else gamma_max`<br>
按 ni/mi 缩小该方向允许的最大关节增量，最大不超过 gamma_max；mi 不为正时直接使用 gamma_max。

**第 144 行**：`component = v * (u[:, i] @ e) / sigma`<br>
计算该奇异方向贡献 v×(uᵀe)/sigma：把误差投影到任务方向，再映射为关节增量。

**第 145 行**：`peak = np.max(np.abs(component), initial=0.)`<br>
寻找该贡献中绝对值最大的关节分量，作为 peak。

**第 146 行**：`if peak:`<br>
只有 peak 非零时才计算缩放，避免除零。

**第 147 行**：`component *= min(1., bound / peak)`<br>
如果 peak 超过 bound，将整个 component 按同一比例缩小；否则保持不变。

**第 148 行**：`result += component`<br>
把该奇异方向处理后的关节增量累加到 result。

**第 149 行**：`peak = np.max(np.abs(result), initial=0.)`<br>
所有方向累加完后，计算总结果中绝对值最大的分量。

**第 150 行**：`return result * min(1., gamma_max / peak) if peak else result`<br>
如果 peak 非零，把整个结果按比例缩小到最大分量不超过 gamma_max；若结果全零直接返回。它限制此函数输出增量，但并非通用硬约束求解器。

## 第 153–346 行：控制器主体

### 第 153–160 行：控制器是什么

- **153 `class PoseController:`**：定义位姿控制器类。一个对象保存控制过程中的目标位姿、关节参考值和上一周期速度。
- **154–160 文档字符串**：这是一个六维末端位姿参考生成器，六维指三维平移与三维旋转；限制目标领先于当前机械臂的程度。`spec` 提供关节界限等参数，`step()` 接收实测状态及世界坐标系雅可比矩阵，返回的 `q_ref` 是给位置执行器的参考值，不是实际关节位置。

### 第 162–186 行：创建控制器并读取设置

- **162 `def __init__(self, spec):`**：新建控制器时执行；`spec` 是设置字典。
- **163 `self.lower = np.asarray(spec['lower'], dtype=float)`**：读取每个关节位置下限并转换成浮点数组。
- **164–165 `if ...: raise ValueError(...)`**：下限必须是非空、一维、所有元素有限的向量，否则报错。
- **166 `self.n = self.lower.size`**：用下限向量长度确定参与控制的关节数量。
- **167 `self.upper = _vec(spec['upper'], self.n, 'upper').copy()`**：读取并校验每个关节的上限，复制一份避免调用方后来修改原数组影响控制器。
- **168 `self.vmax = _vec(spec['velocity_limit'], self.n, 'velocity_limit').copy()`**：读取各关节参考速度的最大值。
- **169 `self.amax = _vec(spec['acceleration_limit'], self.n, 'acceleration_limit').copy()`**：读取各关节参考加速度的最大值。
- **170 `self.home = _vec(spec.get('home', np.clip(np.zeros(self.n), self.lower, self.upper)), self.n, 'home').copy()`**：读取偏好的关节姿态 `home`；没提供时使用零姿态，但把超出上下限的零值裁到相应边界。`home` 是偏好姿态，不是自动回零指令。
- **171–172 `if np.any(...) ... raise ValueError(...)`**：任何关节下限不小于上限，或速度、加速度限制不大于零，都拒绝该配置。
- **173–174 `if np.any(self.home < self.lower) ...`**：偏好姿态必须落在每个关节上下限之内。
- **175–179 `defaults = dict(...)`**：列出默认调参：位置增益 `6`、姿态增益 `5`、末端平移速度上限 `0.12`、角速度上限 `0.7`、旋转长度缩放 `0.25`、阻尼 `0.04`、位置目标最大领先 `0.04`、姿态目标最大领先 `0.35`、关节参考最大领先 `0.12`、姿态偏好增益 `0.08`、SDLS 单步限制 `0.2`。常规情况下位置相关量用米，角度用弧度；速度分别是米/秒和弧度/秒。关节量的单位由关节类型决定，旋转关节通常为弧度。
- **180 `for key, default in defaults.items():`**：依次处理这些设置。
- **181 `value = float(spec.get(key, default))`**：如果 `spec` 提供了当前参数则用提供值，否则用默认值，并转成浮点数。
- **182–183 `if not np.isfinite(value) ... raise ValueError(...)`**：值必须有限、不可为负；只有 `posture_gain` 可以设为零。错误文字写 positive，但该项的零值实际被允许。
- **184 `setattr(self, key, value)`**：把每个数值按名称保存到对象，例如 `self.damping`。
- **185 `self.set_mode(spec.get('mode', 'dls'))`**：设置逆运动学算法模式，未指定则用 `dls`；这是类的默认值，具体程序仍可覆盖。
- **186 `self.reset()`**：清空上次控制状态，为首次更新做好准备。

### 第 188–198 行：切换模式和重置

- **188 `def set_mode(self, mode):`**：定义切换算法模式的方法。
- **189–190 `if mode not in (...) ... raise ValueError(...)`**：仅接受 `dls`、`sdls`、`position_priority`、`constrained_dls`，否则报错。
- **191 `self.mode = mode`**：保存模式；切换模式本身不重置目标或参考值。
- **193 `def reset(self):`**：清理控制器内部状态。
- **194 `self.target_position = None`**：清除世界坐标下的末端目标位置。
- **195 `self.target_rotation = None`**：清除末端目标姿态。
- **196 `self.q_ref = None`**：清除关节位置参考。
- **197 `self.velocity = np.zeros(self.n)`**：把上一周期的参考关节速度置零。
- **198 `self.was_enabled = False`**：标记上一周期没有启用，供下一周期重新对齐目标。

### 第 200–224 行：单周期输入与初始化

- **200 `def step(self, q, qvel, tcp, rotation, jacobian, translation, rotation_input, enabled, dt):`**：进行一个控制周期。`q` 是测得关节位置，`qvel` 是测得关节速度，`tcp` 是末端位置，`rotation` 是末端旋转矩阵，`jacobian` 是几何雅可比矩阵，`translation` 和 `rotation_input` 是三个轴的输入，`enabled` 是使能状态，`dt` 是本周期秒数。
- **201 `start = time.perf_counter()`**：记录高精度起始时间，最后计算本次求解耗时。
- **202 `q = _vec(q, self.n, 'q')`**：检查关节位置是否为长度为 `n` 的有限数值向量。
- **203 `qvel = _vec(qvel, self.n, 'qvel')`**：检查测得的关节速度。注意：本函数后续没有再使用 `qvel`；后面的加速度约束依据控制器保存的上一周期**参考速度**，不是依据测得速度。
- **204 `tcp = _vec(tcp, 3, 'tcp')`**：校验测得的末端三维位置。
- **205 `rotation = np.asarray(rotation, dtype=float)`**：把测得姿态转换成浮点矩阵。
- **206 `rotation_log(rotation)`**：调用旋转对数函数以检查矩阵是否为有限、正交且行列式接近 1 的 3×3 旋转矩阵；此处返回值被丢弃。先校验，避免无效输入改变内部目标。
- **207 `jacobian = np.asarray(jacobian, dtype=float)`**：把雅可比矩阵转为浮点数组。
- **208–209 `if jacobian.shape != (6, self.n) ...`**：雅可比必须是有限的 6×n 矩阵；这里约定前三行是平移、后三行是旋转。
- **210 `translation = _vec(translation, 3, 'translation')`**：校验平移输入，输入位于世界坐标轴方向。
- **211 `rotation_input = _vec(rotation_input, 3, 'rotation_input')`**：校验三轴旋转输入。
- **212–213 `if np.any(np.abs(translation) > 1) ...`**：每个输入轴都必须在 −1 到 1 之间。
- **214–215 `if not isinstance(enabled, (bool, np.bool_)) ...`**：使能值必须是 Python 或 NumPy 布尔值。
- **216–217 `if not np.isfinite(dt) or not 0 < dt <= .05 ...`**：周期必须是有限数，并处于 `(0, 0.05]` 秒；也就是本函数不能使用超过 50 毫秒的单次时间步。
- **218 `constraints = []`**：创建本周期约束或状态标记列表。
- **219 `relax_reference_lead = False`**：先假定不需要为了解决约束冲突而放宽关节参考领先范围。
- **220 `if self.q_ref is None:`**：如果这是重置后的第一个周期，初始化内部状态。
- **221 `self.q_ref = np.clip(q.copy(), self.lower, self.upper)`**：用测得的关节位置初始化参考，同时将其裁到模型关节限位内。
- **222 `self.target_position = tcp.copy()`**：把当前测得末端位置作为初始目标。
- **223 `self.target_rotation = rotation.copy()`**：把当前测得姿态作为初始目标。

### 第 225–244 行：计算末端目标与期望速度

- **225 `if enabled:`**：只有控制使能时才根据输入更新目标。
- **226 `if not self.was_enabled:`**：若上一周期未启用，本周期刚刚开始控制。
- **227–229 注释及赋值**：重新将目标位置和旋转设为测得的当前姿态，避免暂停前的旧目标导致恢复操作时跳动。
- **230 `self.target_position += _bounded(translation, 1.) * self.cartesian_speed * dt`**：将三轴输入整体长度限制在 1，再乘最大平移速度和本周期时间，累加到世界坐标中的位置目标。三个轴同时推满时会整体缩放，防止斜向速度超过设定幅值；机械臂不会被保证瞬时到达目标。
- **231 `self.target_rotation = rotation_exp(_bounded(rotation_input, 1.) * self.angular_speed * dt) @ self.target_rotation`**：同样限制旋转输入总幅值，将角速度乘时间得到本周期旋转向量，经 `rotation_exp` 转成旋转矩阵，并**左乘**目标姿态，代表按世界坐标系旋转。旋转不是将欧拉角逐项相加。
- **232 `e_pos = self.target_position - tcp`**：计算目标位置减去测量位置，得到三维位置误差。
- **233 `e_rot = rotation_log(self.target_rotation @ rotation.T)`**：计算从当前姿态到目标姿态的相对旋转，将其转换成世界坐标系的最短旋转向量。
- **234 `if np.linalg.norm(e_pos) > self.max_position_lead:`**：检查位置目标是否离当前实际位置太远。
- **235 `e_pos = _bounded(e_pos, self.max_position_lead)`**：超过时，保持误差方向并缩小其长度。
- **236 `self.target_position = tcp + e_pos`**：把目标拉回到允许距离内，避免累积很远的等待指令。
- **237 `constraints.append('position_target_lead')`**：记录本周期触发了位置目标领先限制。
- **238 `if np.linalg.norm(e_rot) > self.max_orientation_lead:`**：检查姿态目标相对实测姿态的角度是否过大。
- **239 `e_rot = _bounded(e_rot, self.max_orientation_lead)`**：保持旋转轴方向并限制角度。
- **240 `self.target_rotation = rotation_exp(e_rot) @ rotation`**：用截短后的旋转误差重新构造目标姿态。
- **241 `constraints.append('rotation_target_lead')`**：记录姿态目标领先限制被触发。
- **242 `jpos, jrot = jacobian[:3], jacobian[3:]`**：将雅可比分成位置和旋转两块。
- **243 `vpos = _bounded(self.position_gain * e_pos, self.cartesian_speed)`**：位置误差乘比例增益得到期望线速度，再限制其向量长度。
- **244 `vrot = _bounded(self.orientation_gain * e_rot, self.angular_speed)`**：姿态误差乘比例增益得到期望角速度，再限制其向量长度。

### 第 245–259 行：四种求解模式与姿态偏好

- **245 `if self.mode in ('dls', 'sdls', 'constrained_dls'):`**：这三种模式先把平移和旋转作为组合任务处理；第四种走位置优先分支。
- **246 `jscaled = np.vstack((jpos, self.length_scale * jrot))`**：把位置雅可比与缩放后的旋转雅可比上下拼接。`length_scale` 把旋转任务映射到与平移任务可共同权衡的尺度，不是机械臂真实长度。
- **247 `vscaled = np.r_[vpos, self.length_scale * vrot]`**：期望线速度与缩放后的角速度也上下拼接，使目标与矩阵尺度一致。
- **248–249 `desired = (_dls(...) if ... else sdls_pose(...) / dt)`**：普通 DLS 与受约束 DLS 先算阻尼最小二乘关节速度；SDLS 则把速度乘 `dt` 变成单步增量，用启发式 SDLS 解出关节增量后再除以 `dt` 转回速度。SDLS 不是严格约束求解器，六维版本是该项目对原始思路的扩展。
- **250 `null = _projector(jscaled)`**：计算组合任务雅可比的零空间投影。零空间内的小关节运动在当前姿态的一阶近似下不改变组合末端任务。
- **251 `else:`**：进入 `position_priority` 模式。
- **252 `pos = _dls(jpos, vpos, self.damping)`**：先只求满足位置目标的关节速度。
- **253 `p = _projector(jpos)`**：求位置任务的零空间投影。
- **254–255 注释**：在关节限幅等后处理之前，旋转修正被放在位置零空间中，所以对位置任务没有一阶干扰；后续裁剪仍可能破坏这一优先关系。
- **256 `jr = self.length_scale * jrot @ p`**：构造只允许利用位置零空间的旋转任务矩阵，并做长度缩放。
- **257 `desired = pos + p @ _dls(jr, self.length_scale * (vrot - jrot @ pos), self.damping)`**：先计算位置解已经带来的角速度 `jrot @ pos`，再尝试在不影响位置的一阶空间中补足剩余角速度。
- **258 `null = _projector(np.vstack((jpos, jrot)))`**：为后续姿态偏好寻找位置和姿态组合任务的零空间。
- **259 `desired += self.posture_gain * (null @ (self.home - q))`**：在零空间中添加朝偏好姿态移动的速度；增益为零时不添加。该项不表示实际运动一定不影响末端，也不保证回到 `home`。

### 第 260–289 行：受约束 DLS 求解

- **260 `if self.mode == 'constrained_dls':`**：此模式把参考速度相关的边界直接放进优化求解，而非只在求解后裁剪。项目原生控制配置选择该模式，但类自身默认模式仍是 `dls`。
- **261–262 注释**：将参考范围、制动和每周期加速度条件合并进求解。
- **263 `dist_lo = np.maximum(0., q - self.lower)`**：计算各实测关节位置到下限的非负距离。
- **264 `dist_hi = np.maximum(0., self.upper - q)`**：计算各实测关节位置到上限的非负距离。
- **265–266 `v_lo = ...; v_hi = ...`**：根据距离、加速度限制和离散时间步，计算朝下限、上限方向的参考速度制动包络。它是规划参考时的限制，不代表真实机械臂一定能按该减速度停下。
- **267 `ref_lo = np.maximum(self.lower, q - self.max_joint_reference_lead)`**：计算关节参考允许区间的下边界：既不能低于模型下限，也不能比实测位置落后超过给定量。
- **268 `ref_hi = np.minimum(self.upper, q + self.max_joint_reference_lead)`**：同理计算允许区间的上边界。
- **269 `ref_dist_lo = np.maximum(0., self.q_ref - ref_lo)`**：参考位置距允许区间下边界的非负距离。
- **270 `ref_dist_hi = np.maximum(0., ref_hi - self.q_ref)`**：参考位置距允许区间上边界的非负距离。
- **271–272 `ref_v_lo = ...; ref_v_hi = ...`**：根据到参考区间边界的距离，计算参考在两个方向的制动速度包络。
- **273–274 `lo = np.maximum.reduce((...))`**：逐关节取所有速度下界中最大的值，包括最大负速度、靠近物理下限的制动、靠近参考区间下边界的制动、由上一周期参考速度得到的加速度约束，以及本周期不能越过参考区间下界的条件。
- **275–276 `hi = np.minimum.reduce((...))`**：逐关节取所有速度上界中最小的值，对应正速度、两个上边界制动、加速度及参考区间上界。
- **277 `if np.any(lo > hi):`**：如果某关节的下界竟然大于上界，说明这个周期无法同时满足所有条件。
- **278–280 注释**：冲突时优先避免关节参考发生突然跳变，本周期放宽与参考领先和制动包络相关的部分要求。
- **281 `constraints.extend(('infeasible_velocity_bounds', 'reference_lead_and_braking_relaxed'))`**：记录上下界冲突及相关限制放宽。
- **282 `relax_reference_lead = True`**：标记后续使用连续参考的处理分支。
- **283 `else:`**：当速度上下界有可行解时，运行带盒约束的优化。
- **284–285 `desired, converged = _box_dls(jscaled, vscaled, self.damping, lo, hi, self.posture_gain * (self.home - q))`**：在每个关节速度都落在 `[lo, hi]` 的条件下，求兼顾末端速度误差和向偏好速度靠近的阻尼二次优化。该结果会覆盖前面计算的普通 DLS 加零空间项；这里的姿态偏好是优化目标中的正则项，不是完全相同的零空间相加。
- **286–287 `if not converged: constraints.append('solver_iteration_limit')`**：如果达到迭代上限还没确认收敛，记录这一情况，仍使用函数返回的可行候选值。
- **288–289 `if np.any(np.isclose(desired, lo, atol=1e-9)) ...`**：若某关节求解结果接近上下界，记录 `constrained_velocity`，表示边界可能对求解有影响。

### 第 290–310 行：速度限制、制动和松开使能

- **290–291 `if np.any(np.abs(desired) > self.vmax): constraints.append('velocity')`**：若期望关节参考速度超过限速，先记录。
- **292 `desired = np.clip(desired, -self.vmax, self.vmax)`**：逐关节将期望速度裁到允许范围。
- **293–295 注释**：接下来的制动包络依据实测关节位置；后续加速度上限在硬限位附近仍可能与制动要求冲突，物理后端还必须自行保证限位。
- **296–299 `dist_lo = ...; dist_hi = ...; v_lo = ...; v_hi = ...`**：依据当前实测位置重新计算距上下限距离和两个方向的制动速度。这里的计算在所有启用模式下执行，受约束模式前面也算过一次。
- **300 `limited = np.clip(desired, -v_lo, v_hi)`**：将期望速度裁进制动包络。
- **301–302 `if not np.allclose(limited, desired): constraints.append('braking')`**：如果裁剪使速度发生可检测变化，记录制动限制。
- **303 `desired = limited`**：把裁剪结果作为本周期目标速度。
- **304 `else:`**：若没有按住使能，执行停控分支。
- **305–306 注释**：松开使能时命令减速并把末端目标重新贴近实测位姿，避免下一次启用时继续执行旧目标。
- **307 `self.target_position = tcp.copy()`**：把位置目标设为当前实测位置。
- **308 `self.target_rotation = rotation.copy()`**：把姿态目标设为当前实测姿态。
- **309 `desired = np.zeros(self.n)`**：期望各关节参考速度为零；由于下一步还有加速度限制，它不一定立刻变成零。
- **310 `constraints.append('hold')`**：记录当前处于未使能保持状态。这里的 `hold` 是命令语义，不保证实际关节瞬间停止。

### 第 311–340 行：加速度限制与关节参考更新

- **311 `velocity = np.clip(desired, self.velocity - self.amax * dt, self.velocity + self.amax * dt)`**：将目标速度限制在上一周期参考速度上下 `amax × dt` 的范围内，限制参考速度每周期变化量。
- **312–313 `if not np.allclose(velocity, desired): constraints.append('acceleration')`**：加速度限制改变了目标时记录标记。
- **314 `previous_velocity = self.velocity.copy()`**：保存上一周期参考速度，用于后续检测极限处是否不得不放宽加速度限制。
- **315 `self.velocity = velocity`**：保存本周期参考速度。
- **316 `previous_ref = self.q_ref.copy()`**：记录更新前的关节位置参考。
- **317 `requested_ref = previous_ref + velocity * dt`**：将参考速度积分一个周期，得到尚未裁剪的新关节参考。
- **318 `if self.mode == 'constrained_dls' and not enabled:`**：在受约束模式并且松开使能时，检查渐进减速与移动中的参考窗口是否冲突。
- **319–320 `ref_lo = ...; ref_hi = ...`**：按当前实测关节位置计算新的参考允许范围。
- **321 `if np.any(requested_ref < ref_lo) or np.any(requested_ref > ref_hi):`**：若计划参考超出移动后的允许范围，说明二者存在冲突。
- **322–323 注释**：优先保持关节参考连续，不把它突然裁到随实际位置移动的窗口边缘。
- **324 `constraints.extend(('deadman_reference_lead_conflict', 'reference_lead_and_braking_relaxed'))`**：记录松开使能时的窗口冲突和相应放宽。
- **325 `relax_reference_lead = True`**：后面改用只受绝对关节边界约束的更新分支。
- **326 `if relax_reference_lead:`**：若本周期有需要放宽参考领先范围的冲突，进入该分支。
- **327–329 注释**：绝对关节位置上限优先于速度连续；固定边界不会像测量位置窗口一样移动，但在边界处可能需要放宽加速度约束。
- **330 `self.q_ref = np.clip(requested_ref, self.lower, self.upper)`**：只裁剪到固定模型关节上下限，不再强行裁进以实测位置为中心的领先窗口。
- **331 `velocity = (self.q_ref - previous_ref) / dt`**：用实际参考位置变化反推本周期参考速度。
- **332 `self.velocity = velocity`**：保存重新计算的速度。
- **333–334 `if np.any(np.abs(velocity - previous_velocity) > self.amax * dt + 1e-10): ...`**：若因绝对关节限位而不能维持参考加速度限制，记录 `acceleration_relaxed_at_joint_limit`。
- **335 `else:`**：无上述冲突时使用通常的参考范围裁剪。
- **336–337 `self.q_ref = np.clip(requested_ref, np.maximum(self.lower, q - self.max_joint_reference_lead), np.minimum(self.upper, q + self.max_joint_reference_lead))`**：把参考位置同时限制在模型关节范围及以当前实测 `q` 为中心的领先窗口内。注意：此通常分支裁剪后没有重新计算 `velocity`，所以返回的 `qdot_cmd` 与 `(新 q_ref − 旧 q_ref) / dt` 可能不同；`qdot_cmd` 也不是实测关节速度。
- **338–339 `if not np.allclose(self.q_ref, requested_ref): constraints.append('joint_reference_lead_or_limit')`**：参考位置若被上述裁剪改变，记录参考领先或绝对限位触发；此标记不区分具体是哪一种。
- **340 `self.was_enabled = bool(enabled)`**：记住本周期使能状态，下一周期可以判断是否刚恢复控制。

### 第 341–346 行：计算误差并返回

- **341 `error_position = self.target_position - tcp`**：用当前目标和实测末端位置计算返回用的位置误差向量。
- **342 `error_rotation = rotation_log(self.target_rotation @ rotation.T)`**：计算返回用的旋转误差向量。
- **343–346 `return dict(...)`**：返回各项结果：`q_ref` 是关节位置参考；`qdot_cmd` 是程序内部的参考速度，既不是实测速度，在普通参考裁剪分支下也不一定等于参考位置的实际变化率；`target_position` 和 `target_rotation` 是目标位姿；`error_m`、`error_rad` 分别是位置误差长度和角度误差大小；`constraints` 列出触发的约束或状态；`mode` 是当前算法；`solve_ms` 是本函数此次运行耗时，单位毫秒。返回数组使用 `.copy()`，避免外部直接修改控制器内部数组。

> **理解边界**：这些速度、加速度和位置限制主要作用于控制器生成的参考及求解过程，不保证物理机械臂实际运动、碰撞安全或精确跟踪；实机仍需独立限位、驱动器保护、安全控制以及实测验证。
