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
  gamepad: null, inputMode: null, modeLocked: true, activePadId: null, needsRelease: true };
const pad = { index: 0, id: 'simulated standard pad', mapping: 'standard', axes: [0,0,0,0],
  buttons: Array.from({length: 17}, () => ({pressed: false})) };
let pads = [pad];
const context = vm.createContext({ state, document: {hidden:false}, gamepads: () => pads,
  clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)) });
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
state.focused = false;
assert.equal(sample().enabled, false);
state.focused = true; pads = []; state.keys = new Set(['Space', 'KeyZ']);
assert.equal(sample().gripper, 1); // Keyboard fallback.
console.log('Gamepad mapping, mode rearming, gripper and focus checks passed.');
