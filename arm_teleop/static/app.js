"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("robot-canvas");
  const ctx = canvas.getContext("2d");
  const state = {
    robot: null, connected: false, lastReceived: 0,
    focused: document.hasFocus() && !document.hidden,
    keys: new Set(), gamepad: null, method: "dls", elbowWeight: 0.35, robotId: "ur5e",
    inputMode: null, modeLocked: true, blockedAxes: [false, false, false, false], activePadId: null, needsRelease: true, configWaiters: [],
    lastInput: { direction: [0, 0, 0], rotation: [0, 0, 0], gripper: 0, enabled: false },
    camera: { yaw: -0.7, pitch: 0.47, zoom: 1 },
    inputEpoch: 0, inputController: null, inputTimer: null,
    configRunning: false, configPending: null, configDirty: false,
    resetting: false, width: 1, height: 1, drawNeeded: true,
  };
  const keyCodes = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "KeyI", "KeyK", "KeyJ", "KeyL", "KeyU", "KeyO", "KeyZ", "KeyX", "Space"]);
  const fmt = (v, digits = 2) => Number.isFinite(v) ? v.toFixed(digits) : "—";
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // ZYX roll/pitch/yaw are for display only; control uses rotation matrices.
  function orientationDegrees(matrix) {
    if (!Array.isArray(matrix) || matrix.length !== 3 ||
        !matrix.every((row) => Array.isArray(row) && row.length === 3 && row.every(Number.isFinite))) return null;
    const roll = Math.atan2(matrix[2][1], matrix[2][2]);
    const pitch = Math.atan2(-matrix[2][0], Math.hypot(matrix[0][0], matrix[1][0]));
    const yaw = Math.atan2(matrix[1][0], matrix[0][0]);
    return [roll, pitch, yaw].map((v) => v * 180 / Math.PI);
  }

  function updateJoints(joints) {
    if (!Array.isArray(joints)) return;
    const strip = $("joint-strip");
    if (strip.children.length !== joints.length) {
      strip.replaceChildren();
      strip.style.gridTemplateColumns = `repeat(${Math.max(1, joints.length)}, minmax(0,1fr))`;
      joints.forEach((_, i) => {
        const item = document.createElement("div");
        item.className = "joint-item";
        item.innerHTML = `<span>J${i + 1}</span><strong>—</strong>`;
        strip.appendChild(item);
      });
    }
    joints.forEach((joint, i) => {
      strip.children[i].querySelector("strong").textContent = Number.isFinite(joint) ? `${(joint * 180 / Math.PI).toFixed(1)}°` : "—";
    });
  }

  async function request(path, data, { signal, keepalive = false } = {}) {
    const response = await fetch(path, {
      method: data === undefined ? "GET" : "POST",
      headers: data === undefined ? undefined : { "Content-Type": "application/json" },
      body: data === undefined ? undefined : JSON.stringify(data),
      cache: "no-store", signal, keepalive,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.status === 204 ? null : response.json();
  }

  function gamepads() {
    try { return navigator.getGamepads ? Array.from(navigator.getGamepads()).filter(Boolean) : []; }
    catch (_) { return []; }
  }

  function sampleInput() {
    const pads = gamepads();
    const pad = pads.find((candidate) => candidate.mapping === "standard") || pads[0] || null;
    state.gamepad = pad;
    const safe = state.focused && !document.hidden && state.connected && !state.resetting;
    if (pad && pad.mapping === "standard") {
      const axis = (n) => {
        const value = clamp(Number(pad.axes[n]) || 0, -1, 1);
        const deadzone = 0.12;
        return Math.abs(value) <= deadzone ? 0 : Math.sign(value) * (Math.abs(value) - deadzone) / (1 - deadzone);
      };
      const mode = pad.buttons[5]?.pressed ? "rotation" : "translation";
      const padId = `${pad.index}:${pad.id}`;
      if (state.inputMode !== mode || state.activePadId !== padId) {
        state.inputMode = mode;
        state.activePadId = padId;
        // Block only displaced axes whose meanings changed. Each axis rearms
        // independently once centered; drift on one stick cannot block another.
        const used = mode === "rotation" ? [0, 1, 2, 3] : [0, 1, 3];
        state.blockedAxes = [false, false, false, false];
        for (const i of used) state.blockedAxes[i] = axis(i) !== 0;
      }
      const used = mode === "rotation" ? [0, 1, 2, 3] : [0, 1, 3];
      for (const i of used) if (state.blockedAxes[i] && axis(i) === 0) state.blockedAxes[i] = false;
      state.modeLocked = used.some((i) => state.blockedAxes[i]);
      const unlockedAxis = (i) => state.blockedAxes[i] ? 0 : axis(i);
      if (!pad.buttons[4]?.pressed) state.needsRelease = false;
      const enabled = !!(safe && !state.needsRelease && pad.buttons[4]?.pressed);
      const direction = mode === "translation" ? [unlockedAxis(0), -unlockedAxis(1), -unlockedAxis(3)] : [0, 0, 0];
      // Preserve the existing left-stick mapping and make right-stick up/down
      // control pitch too, so both axes of the right stick visibly rotate.
      const rotation = mode === "rotation" ? [-unlockedAxis(1), clamp(unlockedAxis(0) - unlockedAxis(3), -1, 1), unlockedAxis(2)] : [0, 0, 0];
      const gripper = Number(!!pad.buttons[0]?.pressed) - Number(!!pad.buttons[1]?.pressed);
      return enabled ? { direction, rotation, gripper, enabled } : { direction: [0, 0, 0], rotation: [0, 0, 0], gripper: 0, enabled: false };
    }
    state.activePadId = null;
    state.inputMode = "keyboard";
    const keys = state.keys;
    const direction = [Number(keys.has("KeyD")) - Number(keys.has("KeyA")), Number(keys.has("KeyW")) - Number(keys.has("KeyS")), Number(keys.has("KeyQ")) - Number(keys.has("KeyE"))];
    const rotation = [Number(keys.has("KeyI")) - Number(keys.has("KeyK")), Number(keys.has("KeyL")) - Number(keys.has("KeyJ")), Number(keys.has("KeyO")) - Number(keys.has("KeyU"))];
    for (const vector of [direction, rotation]) {
      const length = Math.hypot(...vector);
      if (length > 1) for (let i = 0; i < 3; i++) vector[i] /= length;
    }
    const enabled = !!(safe && keys.has("Space"));
    return enabled ? { direction, rotation, gripper: Number(keys.has("KeyZ")) - Number(keys.has("KeyX")), enabled } : { direction: [0, 0, 0], rotation: [0, 0, 0], gripper: 0, enabled: false };
  }

  // A single normal input request is in flight. Safety stops cancel it and send
  // a disabled sample immediately; the server also enforces an input watchdog.
  async function sendInput(epoch = state.inputEpoch, forcedStop = false) {
    if (epoch !== state.inputEpoch) return;
    clearTimeout(state.inputTimer);
    const started = performance.now();
    const controller = new AbortController();
    state.inputController = controller;
    const timeout = setTimeout(() => controller.abort(), 450);
    const input = forcedStop ? { direction: [0, 0, 0], rotation: [0, 0, 0], gripper: 0, enabled: false } : sampleInput();
    state.lastInput = input;
    try {
      await request("/api/input", input, { signal: controller.signal, keepalive: forcedStop });
      $("input-error").textContent = "";
    } catch (error) {
      if (error.name !== "AbortError" || epoch === state.inputEpoch) $("input-error").textContent = "输入指令未送达，请检查仿真服务。";
    }
    finally {
      clearTimeout(timeout);
      if (epoch === state.inputEpoch) {
        state.inputController = null;
        state.inputTimer = setTimeout(() => sendInput(epoch), Math.max(1, 33 - (performance.now() - started)));
      }
    }
  }

  function stopInput() {
    state.keys.clear();
    state.modeLocked = true;
    state.needsRelease = true;
    state.activePadId = null;
    clearTimeout(state.inputTimer);
    state.inputEpoch += 1;
    state.inputController?.abort();
    sendInput(state.inputEpoch, true);
    updateInputUI();
  }

  function setConnected(connected) {
    state.connected = connected;
    $("server-status").className = `status-pill ${connected ? "connected" : "disconnected"}`;
    $("server-status-text").textContent = connected ? "仿真已连接" : "连接已中断";
    if (!connected) {
      $("motion-status").textContent = "等待重新连接";
      $("motion-dot").className = "live-dot";
      $("run-message-text").textContent = "连接中断，输入已停止";
      $("run-message").className = "run-message warning";
      if (!state.robot) {
        $("loading-title").textContent = "暂时无法连接仿真";
        $("loading-detail").textContent = "请确认本地服务已启动，页面会自动重连";
      }
    }
  }

  async function pollState() {
    while (true) {
      const started = performance.now();
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1300);
      try {
        const robot = await request("/api/state", undefined, { signal: controller.signal });
        if (!robot || !Array.isArray(robot.points)) throw new Error("Invalid robot state");
        state.robot = robot;
        state.lastReceived = performance.now();
        setConnected(true);
        $("loading-overlay").classList.add("hidden");
        updateStateUI();
        state.drawNeeded = true;
      } catch (_) {
        const wasConnected = state.connected;
        setConnected(false);
        if (wasConnected) stopInput();
      } finally { clearTimeout(timeout); }
      await sleep(Math.max(state.connected ? 1 : 350, 50 - (performance.now() - started)));
    }
  }

  function updateStateUI() {
    const r = state.robot;
    $("error-value").textContent = fmt(r.error_m * 1000, 2);
    $("solve-value").textContent = fmt(r.solve_ms, 3);
    $("orientation-value").textContent = fmt(r.orientation_error_rad * 180 / Math.PI, 2);
    const rpy = orientationDegrees(r.tcp_rotation);
    ["roll", "pitch", "yaw"].forEach((name, i) => $("actual-" + name).textContent = rpy ? `${fmt(rpy[i], 1)}°` : "—");
    const gr = r.gripper_range;
    const aperture = Array.isArray(gr) && gr[1] > gr[0] && Number.isFinite(r.gripper_position) ? clamp((r.gripper_position - gr[0]) / (gr[1] - gr[0]), 0, 1) : null;
    $("gripper-status").textContent = aperture === null ? "夹爪状态：—" : `夹爪开度：${Math.round(aperture * 100)}%`;
    $("sigma-value").textContent = fmt(r.sigma_min, 4);
    ["x", "y", "z"].forEach((axis, i) => $("target-" + axis).textContent = fmt(r.target?.[i], 3));
    updateJoints(r.joints);
    const active = r.enabled && !r.input_stale && state.focused;
    $("motion-dot").className = `live-dot${active ? " active" : ""}`;
    $("motion-status").textContent = active ? "输入使能中" : "待机 · 保持关节位置";
    $("run-message").className = `run-message${active ? " active" : ""}`;
    $("run-message-text").textContent = r.fault ? `仿真故障：${typeof r.fault === "string" ? r.fault : JSON.stringify(r.fault)}` : !state.focused ? "窗口失焦，输入已停止" : r.input_stale ? "等待新输入 · 看门狗已停止输入" : active ? "正在跟踪末端目标" : "按住使能键开始移动";
    if (!state.configRunning && !state.configPending) {
      if (r.method === "dls" || r.method === "position_priority" || r.method === "sdls") state.method = r.method;
      if (r.robot_id === "ur5e" || r.robot_id === "panda") { state.robotId = r.robot_id; $("robot-select").value = r.robot_id; }
      if (!state.configDirty && Number.isFinite(r.elbow_weight)) {
        state.elbowWeight = r.elbow_weight;
        $("elbow-weight").value = String(r.elbow_weight);
      }
      updateConfigUI();
    }
    const c = r.constraints;
    let text = "关节限位与速度限制已启用；目标超前距离受限。";
    if (typeof c === "string" && c) text = c;
    else if (Array.isArray(c) && c.length) {
      const labels = { hold: "待机保持", target_lead: "目标超前限幅", position_target_lead: "位置目标超前限幅", rotation_target_lead: "朝向目标超前限幅", joint_reference_lead_or_limit: "关节参考受限", braking: "关节制动限幅", velocity: "速度限幅", acceleration: "加速度限幅", position_hard_stop: "关节限位保护" };
      text = c.map((key) => labels[key] || key).join(" · ");
    }
    else if (c && typeof c === "object") {
      const labels = { joint_limit: "关节限位", velocity_limit: "速度限制", acceleration_limit: "加速度限制", workspace: "工作区边界", singularity: "奇异区域减速", target_clamped: "目标已限制", joint_limits: "关节限位", velocity_limited: "速度限制", acceleration_limited: "加速度限制", joint_limit_active: "关节限位", workspace_clamped: "目标已限制" };
      const activeLabels = Object.entries(c).filter(([key, value]) => value === true && labels[key]).map(([key]) => labels[key]);
      if (activeLabels.length) text = "限制生效：" + [...new Set(activeLabels)].join(" · ");
    }
    $("constraint-message").textContent = text;
    updateInputUI();
  }

  function updateInputUI() {
    const pad = state.gamepad;
    if (pad?.mapping === "standard") {
      $("input-badge").textContent = "手柄优先";
      $("device-name").textContent = pad.id || "标准手柄已连接";
      $("device-name").title = pad.id;
      $("device-detail").textContent = "标准映射 · LB 使能 · RB 转动";
    } else if (pad) {
      $("input-badge").textContent = "键盘回退";
      $("device-name").textContent = "非标准手柄：映射未验证";
      $("device-name").title = pad.id;
      $("device-detail").textContent = "无法保证按键映射，当前使用键盘";
    } else {
      $("input-badge").textContent = "键盘";
      $("device-name").textContent = "未检测到手柄";
      $("device-name").title = "";
      $("device-detail").textContent = "连接后按任意键唤醒手柄";
    }
    const modeLabel = state.inputMode === "rotation" ? "转动模式" : state.inputMode === "translation" ? "平移模式" : "键盘输入";
    $("input-mode").textContent = state.modeLocked && pad?.mapping === "standard" ? `${modeLabel} · 偏离的摇杆请先回中` : state.needsRelease && pad?.mapping === "standard" ? `${modeLabel} · 请先松开 LB` : modeLabel;
    $("input-mode").classList.toggle("waiting", !!(state.modeLocked && pad?.mapping === "standard"));
    const input = state.lastInput;
    const rotation = input?.rotation || [0, 0, 0];
    $("rotation-input").textContent = input?.enabled ? `转动指令 X ${fmt(rotation[0], 2)} / Y ${fmt(rotation[1], 2)} / Z ${fmt(rotation[2], 2)}` : "转动：按住 LB + RB，回中后推动摇杆";
  }

  function updateConfigUI() {
    document.querySelectorAll("[data-method]").forEach((button) => {
      const selected = button.dataset.method === state.method;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    $("method-description").textContent = state.method === "position_priority" ? "优先跟踪位置，再利用剩余空间追踪朝向和姿态。" : state.method === "sdls" ? "姿态扩展 SDLS（实验），按不同方向调整步长。" : "同时跟踪位置与朝向，使用阻尼减少奇异位置的速度放大。";
    $("elbow-output").textContent = state.elbowWeight.toFixed(2);
    $("elbow-weight").style.background = `linear-gradient(to right,#5b9394 ${state.elbowWeight * 100}%,#e9eeee ${state.elbowWeight * 100}%)`;
  }

  async function queueConfig(changes) {
    state.configPending = { ...(state.configPending || {}), ...changes };
    if (state.configRunning) return new Promise((resolve) => state.configWaiters.push(resolve));
    state.configRunning = true;
    while (state.configPending) {
      const config = state.configPending;
      state.configPending = null;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      try {
        await request("/api/config", config, { signal: controller.signal });
        $("action-message").textContent = "";
        state.configDirty = false;
      } catch (_) {
        $("action-message").textContent = "参数未保存，请连接后重试。";
        state.configDirty = false;
      } finally { clearTimeout(timeout); }
    }
    state.configRunning = false;
    state.configWaiters.splice(0).forEach((resolve) => resolve());
  }

  document.querySelectorAll("[data-method]").forEach((button) => button.addEventListener("click", () => {
    state.method = button.dataset.method;
    updateConfigUI();
    queueConfig({ method: state.method });
  }));
  let configDebounce;
  $("elbow-weight").addEventListener("input", (event) => {
    state.configDirty = true;
    state.elbowWeight = Number(event.target.value);
    updateConfigUI();
    clearTimeout(configDebounce);
    configDebounce = setTimeout(() => queueConfig({ elbow_weight: state.elbowWeight }), 100);
  });
  $("elbow-weight").addEventListener("change", () => { clearTimeout(configDebounce); queueConfig({ elbow_weight: state.elbowWeight }); });
  $("robot-select").addEventListener("change", async (event) => {
    state.resetting = true; stopInput(); event.target.disabled = true;
    state.robotId = event.target.value;
    await queueConfig({ robot: state.robotId });
    state.resetting = false; event.target.disabled = false;
  });
  $("reset-arm").addEventListener("click", async () => {
    state.resetting = true;
    stopInput();
    $("reset-arm").disabled = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    try { await request("/api/reset", {}, { signal: controller.signal }); $("action-message").textContent = ""; }
    catch (_) { $("action-message").textContent = "复位失败，请检查本地服务后重试。"; }
    finally { clearTimeout(timeout); state.resetting = false; $("reset-arm").disabled = false; }
  });
  $("reset-view").addEventListener("click", () => {
    state.camera = { yaw: -0.7, pitch: 0.47, zoom: 1 };
    state.drawNeeded = true;
  });

  window.addEventListener("keydown", (event) => {
    if (!keyCodes.has(event.code)) return;
    if (/INPUT|TEXTAREA|SELECT|BUTTON/.test(event.target.tagName) || event.target.isContentEditable) return;
    event.preventDefault();
    state.keys.add(event.code);
  });
  window.addEventListener("keyup", (event) => {
    if (state.keys.has(event.code)) event.preventDefault();
    state.keys.delete(event.code);
  });
  window.addEventListener("blur", () => { state.focused = false; stopInput(); });
  window.addEventListener("focus", () => { state.focused = !document.hidden; state.keys.clear(); });
  document.addEventListener("visibilitychange", () => {
    state.focused = !document.hidden && document.hasFocus();
    if (!state.focused) stopInput();
  });
  window.addEventListener("pagehide", () => { state.focused = false; stopInput(); });
  window.addEventListener("gamepadconnected", () => { sampleInput(); updateInputUI(); });
  window.addEventListener("gamepaddisconnected", () => { stopInput(); sampleInput(); updateInputUI(); });
  document.addEventListener("click", (event) => {
    // Mouse users can resume keyboard teleoperation after changing a setting;
    // keyboard activation keeps focus for accessible button navigation.
    if (event.detail > 0) event.target.closest("button")?.blur();
  });

  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    canvas.focus({ preventScroll: true });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    state.camera.yaw += (event.clientX - drag.x) * 0.007;
    state.camera.pitch = clamp(state.camera.pitch + (event.clientY - drag.y) * 0.005, 0.1, 1.32);
    drag = { x: event.clientX, y: event.clientY };
    state.drawNeeded = true;
  });
  const endDrag = () => { drag = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.camera.zoom = clamp(state.camera.zoom * Math.exp(-event.deltaY * 0.001), 0.65, 1.7);
    state.drawNeeded = true;
  }, { passive: false });

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.width = rect.width; state.height = rect.height;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.drawNeeded = true;
  }
  new ResizeObserver(resize).observe($("scene-stage"));

  function project(point) {
    const [x, y, z] = point;
    const { yaw, pitch, zoom } = state.camera;
    const rx = Math.cos(yaw) * x - Math.sin(yaw) * y;
    const ry = Math.sin(yaw) * x + Math.cos(yaw) * y;
    const centeredZ = z - 0.25;
    const vertical = Math.cos(pitch) * centeredZ - Math.sin(pitch) * ry;
    const depth = Math.cos(pitch) * ry + Math.sin(pitch) * centeredZ;
    const scale = Math.min(state.width / 1.8, state.height / 1.1) * zoom;
    const perspective = 2.8 / (2.8 - depth);
    return { x: state.width * 0.5 + rx * scale * perspective, y: state.height * 0.53 - vertical * scale * perspective, depth, scale: scale * perspective };
  }

  function line(a, b, color, width = 1, dash = []) {
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  }

  function dot(p, radius, fill, stroke, strokeWidth = 1) {
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = strokeWidth; ctx.stroke(); }
  }

  function label(text, point, color, dx = 8, dy = -9) {
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = color;
    ctx.fillText(text, point.x + dx, point.y + dy);
  }

  function draw() {
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let i = -8; i <= 8; i++) {
      const a = i * 0.1;
      line(project([a, -0.8, 0]), project([a, 0.8, 0]), i === 0 ? "#c9d9d3" : "#e0e8e2", i === 0 ? 1.1 : 0.7);
      line(project([-0.8, a, 0]), project([0.8, a, 0]), i === 0 ? "#c9d9d3" : "#e0e8e2", i === 0 ? 1.1 : 0.7);
    }
    const base = project([0, 0, 0]);
    const xEnd = project([0.22, 0, 0]), yEnd = project([0, 0.22, 0]), zEnd = project([0, 0, 0.22]);
    line(base, xEnd, "#c68579", 1.7); line(base, yEnd, "#80a992", 1.7); line(base, zEnd, "#85a6c0", 1.7);
    label("X", xEnd, "#b47d75", 4, 3); label("Y", yEnd, "#709782", 4, 3); label("Z", zEnd, "#7b9cb8", 4, 3);
    if (!state.robot) return;
    const r = state.robot;
    const points = r.points.filter((p) => Array.isArray(p) && p.length >= 3 && p.every(Number.isFinite));
    if (points.length < 2) return;
    // The ground shadow is built from the same measured joint positions.
    for (let i = 1; i < points.length; i++) {
      const a = project([points[i - 1][0], points[i - 1][1], 0.001]);
      const b = project([points[i][0], points[i][1], 0.001]);
      line(a, b, "#47716a0a", 18); line(a, b, "#47716a09", 10);
    }
    const p = points.map(project);
    const basePoint = p[0];
    ctx.save(); ctx.translate(basePoint.x, basePoint.y);
    ctx.beginPath(); ctx.ellipse(0, 4, 23, 10, 0, 0, Math.PI * 2); ctx.fillStyle = "#738d8a"; ctx.fill();
    ctx.beginPath(); ctx.ellipse(0, 0, 23, 10, 0, 0, Math.PI * 2); ctx.fillStyle = "#9eafaa"; ctx.fill();
    ctx.beginPath(); ctx.ellipse(0, -1, 15, 6, 0, 0, Math.PI * 2); ctx.fillStyle = "#d9e3de"; ctx.fill();
    ctx.restore();
    const segments = [];
    for (let i = 1; i < p.length; i++) segments.push({ a: p[i - 1], b: p[i], index: i, depth: (p[i - 1].depth + p[i].depth) / 2 });
    segments.sort((a, b) => a.depth - b.depth);
    for (const segment of segments) {
      const { a, b, index } = segment;
      const width = Math.max(7, 16 - index * 1.5) * Math.min(1.15, state.width / 550);
      line({ x: a.x + 1, y: a.y + 2 }, { x: b.x + 1, y: b.y + 2 }, "#42615f18", width + 5);
      line(a, b, index < 4 ? "#688f89" : "#85a69a", width + 2);
      line(a, b, index < 4 ? "#bdd1c6" : "#cee0ce", width - 1);
      const jointRadius = Math.max(5, width * 0.58);
      dot(a, jointRadius + 2, "#668982");
      dot(a, jointRadius, "#e0ebe3", "#97b4a8", 1.2);
      dot(a, 2, "#7c9b91");
    }
    const tcp = project(r.tcp || points[points.length - 1]);
    // Draw a schematic gripper in the current TCP frame; it indicates aperture,
    // and is not a rendering of the robot collision geometry.
    const quat = r.tcp_quat;
    const position = r.tcp || points[points.length - 1];
    if (Array.isArray(quat) && quat.length === 4 && quat.every(Number.isFinite)) {
      const [w, x, y, z] = quat;
      const rotate = ([vx, vy, vz]) => {
        const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
        return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx];
      };
      const range = r.gripper_range;
      const ratio = Array.isArray(range) && range.length === 2 && range[1] > range[0] && Number.isFinite(r.gripper_position)
        ? clamp((r.gripper_position - range[0]) / (range[1] - range[0]), 0, 1) : 0.5;
      const offset = 0.008 + 0.027 * ratio;
      const at = (local) => { const v = rotate(local); return project(position.map((a, i) => a + v[i])); };
      // Tool-local axes make all three orientation changes visible, including
      // rotation around the gripper's own axis when its jaws are closed.
      const axisOrigin = at([0, 0, 0]);
      for (const [vector, color, name] of [
        [[0.13, 0, 0], "#cf554e", "工具 X"],
        [[0, 0.13, 0], "#30935a", "工具 Y"],
        [[0, 0, 0.13], "#367fc7", "工具 Z"],
      ]) {
        const endpoint = at(vector);
        line(axisOrigin, endpoint, color, 2.8);
        dot(endpoint, 3.5, color, "#ffffff", 1);
        label(name, endpoint, color, 5, -5);
      }
      const left = at([0, offset, 0]), right = at([0, -offset, 0]);
      line(left, right, "#436b69", 5);
      line(left, at([0, offset, 0.06]), "#436b69", 5);
      line(right, at([0, -offset, 0.06]), "#436b69", 5);
    }
    dot(tcp, 10, "#2f96a319"); dot(tcp, 5, "#268797", "#f5fffa", 2);
    label("TCP", tcp, "#43858b", 10, -12);
    if (Array.isArray(r.target) && r.target.length === 3 && r.target.every(Number.isFinite)) {
      const target = project(r.target);
      const ground = project([r.target[0], r.target[1], 0]);
      line(ground, target, "#c6ae7770", 1, [3, 4]);
      dot(ground, 3, "#d8bd8b55");
      if (Math.hypot(target.x - tcp.x, target.y - tcp.y) > 3) line(tcp, target, "#d3a85c", 1.2, [3, 4]);
      dot(target, 10, "#e8bc5a12", "#ce9c46", 1.2);
      line({ x: target.x - 14, y: target.y }, { x: target.x - 7, y: target.y }, "#cfa65c", 1);
      line({ x: target.x + 7, y: target.y }, { x: target.x + 14, y: target.y }, "#cfa65c", 1);
      line({ x: target.x, y: target.y - 14 }, { x: target.x, y: target.y - 7 }, "#cfa65c", 1);
      line({ x: target.x, y: target.y + 7 }, { x: target.x, y: target.y + 14 }, "#cfa65c", 1);
      label("TARGET", target, "#b89861", 15, 20);
    }
    // A small fixed camera compass helps make base-frame joystick mapping clear.
    ctx.font = "8px system-ui, sans-serif";
    ctx.fillStyle = "#a8b3ae";
    ctx.fillText("GRID  0.10 m", 24, state.height - 90);
  }

  function frame() {
    if (state.drawNeeded && state.width > 0 && state.height > 0) { draw(); state.drawNeeded = false; }
    requestAnimationFrame(frame);
  }
  updateConfigUI();
  resize();
  sendInput();
  pollState();
  requestAnimationFrame(frame);
})();
