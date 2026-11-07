"use strict";

// Browser-local input preferences; no commands are sent from this module.
(() => {
  const defaults = {
    keys: { xPlus: "KeyD", xMinus: "KeyA", yPlus: "KeyW", yMinus: "KeyS", zPlus: "KeyQ", zMinus: "KeyE", rollPlus: "KeyI", rollMinus: "KeyK", pitchPlus: "KeyL", pitchMinus: "KeyJ", yawPlus: "KeyO", yawMinus: "KeyU", open: "KeyZ", close: "KeyX", enable: "Space" },
    axes: [0, 1, 2, 3], invert: [false, false, false, false],
    buttons: { enable: 4, rotate: 5, open: 0, close: 1 }, deadzone: 0.12,
  };
  const keyLabels = { xPlus: "X 正", xMinus: "X 负", yPlus: "Y 正", yMinus: "Y 负", zPlus: "Z 正", zMinus: "Z 负", rollPlus: "滚转正", rollMinus: "滚转负", pitchPlus: "俯仰正", pitchMinus: "俯仰负", yawPlus: "偏航正", yawMinus: "偏航负", open: "夹爪打开", close: "夹爪关闭", enable: "按住使能" };
  const buttonLabels = { enable: "使能", rotate: "转动模式", open: "夹爪打开", close: "夹爪关闭" };
  const axisLabels = ["左摇杆左右", "左摇杆上下", "右摇杆左右", "右摇杆上下"];
  const keys = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"].map(c => `Key${c}`).concat(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"]);
  const clone = value => JSON.parse(JSON.stringify(value));
  function valid(value) {
    if (!value || !value.keys || !value.buttons || !Array.isArray(value.axes) || !Array.isArray(value.invert)) return false;
    const ks = Object.keys(defaults.keys).map(k => value.keys[k]);
    const bs = Object.keys(defaults.buttons).map(k => value.buttons[k]);
    return ks.every(k => keys.includes(k)) && new Set(ks).size === ks.length &&
      value.axes.length === 4 && value.axes.every(n => Number.isInteger(n) && n >= 0 && n < 4) && new Set(value.axes).size === 4 &&
      value.invert.length === 4 && value.invert.every(v => typeof v === "boolean") &&
      bs.every(n => Number.isInteger(n) && n >= 0 && n <= 15) && new Set(bs).size === bs.length &&
      Number.isFinite(value.deadzone) && value.deadzone >= 0 && value.deadzone <= 0.4;
  }
  let current = clone(defaults);
  try { const saved = JSON.parse(localStorage.getItem("arm-teleop-input-settings")); if (valid(saved)) current = saved; } catch (_) { /* Defaults still work without storage. */ }
  function save(next) {
    if (!valid(next)) return false;
    current = next;
    try { localStorage.setItem("arm-teleop-input-settings", JSON.stringify(next)); } catch (_) { /* Session-only settings. */ }
    return true;
  }
  function mount(onChange) {
    const fields = document.getElementById("input-settings-fields");
    const message = document.getElementById("input-settings-message");
    const reset = document.getElementById("input-settings-reset");
    function select(label, options, selected, change) {
      const row = document.createElement("label"); row.className = "mapping-row";
      const text = document.createElement("span"); text.textContent = label;
      const box = document.createElement("select"); box.setAttribute("aria-label", label);
      for (const [value, name] of options) { const option = document.createElement("option"); option.value = value; option.textContent = name; box.appendChild(option); }
      box.value = String(selected);
      box.addEventListener("change", () => { const next = clone(current); change(next, box.value); if (save(next)) { message.textContent = "设置已保存；请松开使能并回中。"; onChange(); } else { message.textContent = "同一组中的按键或轴不能重复。"; } render(); });
      row.append(text, box); fields.appendChild(row);
    }
    function heading(text) { const title = document.createElement("strong"); title.textContent = text; fields.appendChild(title); }
    function render() {
      fields.replaceChildren();
      heading("键盘");
      for (const [key, label] of Object.entries(keyLabels)) select(label, keys.map(k => [k, k.replace("Key", "").replace("Arrow", "方向 ")]), current.keys[key], (n, v) => { const other = Object.keys(n.keys).find(k => n.keys[k] === v); if (other) n.keys[other] = n.keys[key]; n.keys[key] = v; });
      heading("标准手柄");
      for (const [key, label] of Object.entries(buttonLabels)) select(label, Array.from({length: 16}, (_, i) => [i, `按钮 ${i}`]), current.buttons[key], (n, v) => { v = Number(v); const other = Object.keys(n.buttons).find(k => n.buttons[k] === v); if (other) n.buttons[other] = n.buttons[key]; n.buttons[key] = v; });
      axisLabels.forEach((label, i) => {
        select(label, Array.from({length: 4}, (_, j) => [j, `轴 ${j}`]), current.axes[i], (n, v) => { v = Number(v); const other = n.axes.indexOf(v); if (other >= 0) n.axes[other] = n.axes[i]; n.axes[i] = v; });
        select(`${label}方向`, [["normal", "正常"], ["inverted", "反向"]], current.invert[i] ? "inverted" : "normal", (n, v) => { n.invert[i] = v === "inverted"; });
      });
      select("摇杆死区", [["0.05", "5%"], ["0.12", "12%"], ["0.2", "20%"], ["0.3", "30%"]], current.deadzone, (n, v) => { n.deadzone = Number(v); });
    }
    reset.addEventListener("click", () => { save(clone(defaults)); message.textContent = "已恢复默认设置；请松开使能并回中。"; onChange(); render(); });
    document.getElementById("input-settings").addEventListener("toggle", onChange);
    render();
  }
  window.InputSettings = { get: () => current, defaults: () => clone(defaults), valid, mount };
})();
