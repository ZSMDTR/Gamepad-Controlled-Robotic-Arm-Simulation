// Exercise the real browser input mapper with simulated Gamepad API values.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../arm_teleop/static/app.js'), 'utf8');
const start = source.indexOf('  function sampleInput() {');
const end = source.indexOf('\n  // A single normal input request', start);
assert.ok(start >= 0 && end > start, 'input mapper must be present');
const state = { focused: true, connected: true, resetting: false, keys: new Set(),
  gamepad: null, inputMode: null, modeLocked: true, activePadId: null, needsRelease: true, keyboardNeedsRelease: false };
const pad = { index: 0, id: 'simulated standard pad', mapping: 'standard', axes: [0,0,0,0],
  buttons: Array.from({length: 17}, () => ({pressed: false})) };
let pads = [pad];
const settingsContext = vm.createContext({window: {}, localStorage: {getItem: () => null}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../arm_teleop/static/input-settings.js'), 'utf8'), settingsContext);
const InputSettings = settingsContext.window.InputSettings;
const settings = InputSettings.get();
const settingsPanel = {open: false};
const context = vm.createContext({ state, document: {hidden:false}, gamepads: () => pads,
  InputSettings, $: () => settingsPanel, clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)) });
vm.runInContext(source.slice(start, end) + '\nthis.sample = sampleInput;', context);
const sample = () => JSON.parse(JSON.stringify(context.sample()));
assert.equal(sample().enabled, false); // Release first; no spontaneous motion.
pad.buttons[4].pressed = true; // LB
pad.axes[0] = .56; pad.axes[1] = -.56; pad.axes[3] = -.56;
let input = sample();
assert.equal(input.enabled, true);
input.direction.forEach(v => assert.ok(Math.abs(v - .5) < 1e-12));
assert.deepEqual(input.rotation, [0,0,0]);
pad.buttons[5].pressed = true; // RB: changed meaning requires neutral first.
input = sample();
assert.deepEqual(input.rotation, [0,0,0]);
assert.equal(state.modeLocked, true);
pad.axes = [0,0,0,0]; sample();
assert.equal(state.modeLocked, false);
pad.axes = [.56,-.56,.56,0];
pad.buttons[0].pressed = true; // A opens.
input = sample();
input.rotation.forEach(v => assert.ok(Math.abs(v - .5) < 1e-12));
assert.equal(input.gripper, 1);
pad.buttons[1].pressed = true; // Contradictory buttons cancel.
assert.equal(sample().gripper, 0);
// Return to translation, then change mode with right-stick vertical displaced.
// One displaced axis must not block another centered axis in rotation mode.
pad.buttons[5].pressed = false; pad.axes = [0,0,0,.56]; sample();
pad.buttons[5].pressed = true; sample();
assert.equal(state.modeLocked, true);
pad.axes[2] = .56;
input = sample();
assert.ok(Math.abs(input.rotation[2] - .5) < 1e-12, 'unblocked yaw remains usable');
assert.equal(input.rotation[1], 0, 'displaced pitch axis waits for neutral');
pad.axes[3] = 0; sample();
pad.axes[3] = -.56;
input = sample();
assert.ok(Math.abs(input.rotation[1] - .5) < 1e-12, 'right stick vertical controls pitch');
state.focused = false;
assert.equal(sample().enabled, false);
state.focused = true; pads = []; state.keys = new Set(['Space', 'KeyZ']);
assert.equal(sample().gripper, 1); // Keyboard fallback.
settings.keys.enable = 'ArrowUp'; settings.keys.open = 'KeyP';
state.keys = new Set(['ArrowUp', 'KeyP']);
assert.equal(sample().enabled, true);
assert.equal(sample().gripper, 1);
state.keyboardNeedsRelease = true;
assert.equal(sample().enabled, false);
state.keyboardNeedsRelease = false; settingsPanel.open = true;
assert.equal(sample().enabled, false, 'settings panel blocks motion');
settingsPanel.open = false; pads = [pad]; pad.axes = [0, 0, 0, 0];
settings.buttons.enable = 6; settings.axes = [1, 0, 2, 3]; settings.invert[0] = true;
pad.buttons[4].pressed = false; pad.buttons[5].pressed = false; pad.buttons[6].pressed = false; state.needsRelease = true; state.inputMode = null;
sample(); pad.buttons[6].pressed = true; pad.axes[1] = .56;
assert.ok(Math.abs(sample().direction[0] + .5) < 1e-12, 'remapped and inverted axis applies');
assert.ok(InputSettings.valid(settings));
const orientationStart = source.indexOf('  function orientationDegrees(matrix) {');
const orientationEnd = source.indexOf('\n  function updateJoints(', orientationStart);
assert.ok(orientationStart >= 0 && orientationEnd > orientationStart);
vm.runInContext(source.slice(orientationStart, orientationEnd) + '\nthis.orientation = orientationDegrees;', context);
const identity = [[1,0,0],[0,1,0],[0,0,1]];
const yaw90 = [[0,-1,0],[1,0,0],[0,0,1]];
const pitch90 = [[0,0,1],[0,1,0],[-1,0,0]];
assert.deepEqual(JSON.parse(JSON.stringify(context.orientation(identity))), [0,0,0]);
assert.ok(Math.abs(context.orientation(yaw90)[2] - 90) < 1e-10);
assert.ok(Math.abs(context.orientation(pitch90)[1] - 90) < 1e-10);
assert.equal(context.orientation([[1]]), null);
console.log('Gamepad mapping, mode rearming, gripper and focus checks passed.');
