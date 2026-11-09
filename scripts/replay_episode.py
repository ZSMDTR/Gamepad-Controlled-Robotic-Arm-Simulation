"""Inspect or visually replay a recorded episode's saved simulation states.

State playback displays recorded poses. It does not rerun recorded actions and
therefore does not verify that the controller can reproduce the demonstration.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer
import numpy as np

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.recorder import _sha256, model_fingerprints
from arm_teleop.world_axes import add_world_axes, update_tcp_axes


def records(path):
    """Read UTF-8 JSONL without holding an entire recording in memory."""
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'第 {line_number} 行不是有效的 JSON：{exc.msg}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'第 {line_number} 行必须是 JSON 对象')
            yield value


def _state_mj(state):
    if not isinstance(state, dict) or not isinstance(state.get('mj'), dict):
        raise ValueError('状态缺少 mj 数据')
    mj = state['mj']
    try:
        moment = float(mj['time'])
        if not math.isfinite(moment):
            raise ValueError('仿真时间必须是有限数值')
        for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat'):
            array = np.asarray(mj[name], dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f'{name} 包含非有限数值')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'仿真状态不完整：{exc}') from exc
    return mj


def _validate_action(action):
    if not isinstance(action, dict):
        raise ValueError('缺少有效动作')
    for name in ('direction', 'rotation'):
        try:
            value = np.asarray(action[name], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'动作 {name} 必须是 3 个有限数值') from exc
        if value.shape != (3,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            raise ValueError(f'动作 {name} 必须是 [-1, 1] 内的 3 个有限数值')
    if type(action.get('enabled')) is not bool:
        raise ValueError('动作 enabled 必须是布尔值')
    if type(action.get('gripper')) is not int or action['gripper'] not in (-1, 0, 1):
        raise ValueError('动作 gripper 必须是 -1、0 或 1')
    source = action.get('source', 'manual')
    if source not in ('manual', 'leader'):
        raise ValueError('动作来源不受支持')
    if source == 'leader':
        sample = action.get('leader_sample')
        if sample is None:
            if action['enabled']:
                raise ValueError('启用的主臂动作缺少关节读数')
        else:
            try:
                positions = np.asarray(sample['positions'], dtype=float)
                timestamp = float(sample['timestamp'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('主臂读数无效') from exc
            if positions.shape != (7,) or not np.isfinite(positions).all() or not math.isfinite(timestamp):
                raise ValueError('主臂读数必须包含 7 个有限角度和有效时间')


def inspect_episode(path):
    """Validate episode framing and return metadata plus its end record."""
    rows = records(path)
    metadata = next(rows, None)
    if not metadata or metadata.get('type') != 'metadata' or metadata.get('schema_version') != 1:
        raise ValueError('文件开头缺少支持的第 1 版元数据')
    if not isinstance(metadata.get('robot_id'), str) or not isinstance(metadata.get('scene_id'), str):
        raise ValueError('元数据缺少 robot_id 或 scene_id')
    count = 0
    end = None
    last_time = None
    for row in rows:
        if end is not None:
            raise ValueError('结束标记之后还有数据')
        if row.get('type') == 'end':
            end = row
            continue
        if row.get('type') != 'transition' or row.get('index') != count:
            raise ValueError(f'第 {count} 条 transition 缺失或序号不连续')
        _validate_action(row.get('action'))
        state = _state_mj(row.get('state'))
        next_state = _state_mj(row.get('next_state'))
        start, finish = float(state['time']), float(next_state['time'])
        if finish < start or (last_time is not None and start < last_time - 1e-8):
            raise ValueError('仿真时间倒退，不能按时间播放')
        last_time = finish
        count += 1
    if end is None:
        raise ValueError('文件没有结束标记，可能是未完成的采集')
    if end.get('outcome') not in ('success', 'failure', 'interrupted'):
        raise ValueError('结束标记的结果不受支持')
    if end.get('steps') != count:
        raise ValueError(f'结束标记的步数与实际条数不一致：{end.get("steps")} != {count}')
    if count == 0:
        raise ValueError('轨迹没有采集到任何控制周期')
    return metadata, end


def apply_state(model, data, state):
    """Restore a saved display state and recompute geometry for rendering."""
    mj = _state_mj(state)
    for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat'):
        destination = getattr(data, name)
        values = np.asarray(mj[name], dtype=float)
        if values.size != destination.size:
            raise ValueError(f'{name} 的尺寸与当前模型不一致；请使用采集时相同的模型和场景')
        destination[:] = values.reshape(destination.shape)
    data.time = float(mj['time'])
    mujoco.mj_forward(model, data)


def make_backend(metadata, directory):
    robot, scene = metadata['robot_id'], metadata['scene_id']
    expected = metadata.get('model_fingerprints')
    if isinstance(expected, dict):
        actual = model_fingerprints(robot)
        for name in ('profile_sha256', 'model_xml_sha256'):
            if expected.get(name) and expected[name] != actual.get(name):
                raise ValueError(f'本机模型 {name} 与录制时不同，请安装采集时使用的模型')
    else:
        print('警告：轨迹没有模型指纹，无法核对模型版本。', flush=True)
    if robot == 'b601_rs' and scene == 'b601_practice':
        hashes = metadata.get('code_sha256')
        expected_scene = hashes.get('arm_teleop/b601_scene.py') if isinstance(hashes, dict) else None
        scene_code = Path(__file__).resolve().parent.parent / 'arm_teleop/b601_scene.py'
        if expected_scene and _sha256(scene_code) != expected_scene:
            raise ValueError('本机场景生成器与录制时不同，请使用采集时的项目版本')
        source = Path(__file__).resolve().parent.parent / 'robot_models/b601_rs/profile.json'
        if not source.is_file():
            raise ValueError('本机尚未安装 B601-RS 模型，请先按项目说明安装')
        profile = make_b601_scene_profile(source, directory)
        return MujocoBackend(robot, profile_path=profile)
    if scene == 'robot_only':
        return MujocoBackend(robot)
    raise ValueError(f'不支持的场景 {scene!r}，无法安全地重建回放模型')


def latest_episode(directory):
    candidates = list(Path(directory).glob('episode_*.jsonl'))
    if not candidates:
        raise ValueError(f'在 {directory} 中没有找到示教轨迹')
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def playback(path, metadata, speed=1.0):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('播放速度必须大于零')
    with tempfile.TemporaryDirectory(prefix='episode-replay-') as directory:
        backend = make_backend(metadata, directory)
        model, data = backend.model, backend.data
        rows = records(path)
        next(rows)  # metadata was validated before launching
        first = next(rows)
        apply_state(model, data, first['state'])
        with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
            if metadata['robot_id'] == 'b601_rs':
                viewer.cam.lookat[:] = [.2, 0, .28]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            axes = add_world_axes(viewer.user_scn)
            if hasattr(viewer, 'set_texts'):
                viewer.set_texts((None, None, 'STATE PLAYBACK - not action replay', ''))
            update_tcp_axes(viewer.user_scn, axes, data.site_xpos[backend.tcp_id])
            viewer.sync()
            start_sim, start_wall = data.time, time.monotonic()
            last_sync = start_wall
            from itertools import chain
            for row in chain((first,), rows):
                if row.get('type') != 'transition' or not viewer.is_running():
                    break
                state = row['next_state']
                target = start_wall + (float(state['mj']['time']) - start_sim) / speed
                time.sleep(max(0, target - time.monotonic()))
                apply_state(model, data, state)
                now = time.monotonic()
                if now - last_sync >= 1 / 30:
                    update_tcp_axes(viewer.user_scn, axes, data.site_xpos[backend.tcp_id])
                    viewer.sync()
                    last_sync = now
            if viewer.is_running():
                update_tcp_axes(viewer.user_scn, axes, data.site_xpos[backend.tcp_id])
                viewer.sync()
                print('回放结束，关闭窗口退出。', flush=True)
                while viewer.is_running():
                    time.sleep(.1)


def main():
    parser = argparse.ArgumentParser(description='检查或逐帧显示 MuJoCo 示教轨迹（状态回放，不是动作重演）')
    parser.add_argument('episode', nargs='?', type=Path, help='示教轨迹 JSONL 文件')
    parser.add_argument('--latest', action='store_true', help='使用项目 recordings 文件夹中新录制的轨迹')
    parser.add_argument('--inspect', '--check-only', dest='check_only', action='store_true',
                        help='只检查轨迹和输出元数据，不打开窗口')
    parser.add_argument('--speed', type=float, default=1., help='状态回放速度倍率，默认 1')
    args = parser.parse_args()
    if bool(args.episode) == args.latest:
        parser.error('请指定一条轨迹文件，或使用 --latest')
    try:
        episode = (latest_episode(Path(__file__).resolve().parent.parent / 'recordings')
                   if args.latest else args.episode)
        metadata, end = inspect_episode(episode)
        print(json.dumps({'file': str(episode), 'metadata': metadata, 'end': end},
                         ensure_ascii=False, indent=2), flush=True)
        print('文件结构检查通过；状态回放不等于动作重演。', flush=True)
        if not args.check_only:
            playback(episode, metadata, args.speed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'轨迹检查或回放失败：{exc}\n')


if __name__ == '__main__':
    main()
