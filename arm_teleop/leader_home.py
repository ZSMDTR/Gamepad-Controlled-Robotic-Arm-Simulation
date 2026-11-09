"""Local, explicitly captured home pose for the read-only leader arm.

Angles are sensor positions, not commands to the hardware.  A file describes
one arm's installation and must be recalibrated if its setup changes.
"""
import json
import math
import os
from pathlib import Path
import tempfile


DEFAULT_TOLERANCE = math.radians(5)
MAX_SAMPLE_AGE = .15
MIN_STABLE_DURATION = .30
MAX_SAMPLE_GAP = .15
MAX_STABLE_MOVEMENT = math.radians(1)


def _vector(value, label, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(f'{label} must contain seven angles')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or abs(v) > math.pi or (positive and v <= 0)
           for v in value):
        raise ValueError(f'{label} must contain finite angles in the valid range')
    return [float(v) for v in value]


def _home(value):
    if not isinstance(value, dict) or value.get('version') != 1 or isinstance(value.get('version'), bool):
        raise ValueError('unsupported or invalid leader home file')
    return {'version': 1,
            'positions_rad': _vector(value.get('positions_rad'), 'positions_rad'),
            'tolerances_rad': _vector(value.get('tolerances_rad'), 'tolerances_rad', positive=True)}


def _sample(sample, now):
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError('invalid current time')
    try:
        positions, timestamp = sample.positions, sample.timestamp
    except AttributeError as exc:
        raise ValueError('leader sample is missing') from exc
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or
            not math.isfinite(timestamp) or timestamp > now + .01):
        raise ValueError('invalid leader sample timestamp')
    return _vector(positions, 'leader sample'), float(timestamp)


def _distance(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def save_home(path, samples, now, tolerances_rad=None):
    """Atomically save a stable, freshly sampled pose, preserving old data on failure."""
    if tolerances_rad is None:
        tolerances_rad = [DEFAULT_TOLERANCE] * 7
    tolerances = _vector(tolerances_rad, 'tolerances_rad', positive=True)
    samples = [_sample(sample, now) for sample in samples]
    if len(samples) < 2 or now - samples[-1][1] > MAX_SAMPLE_AGE:
        raise ValueError('fresh, stable leader samples are required')
    times = [sample[1] for sample in samples]
    if times[-1] - times[0] < MIN_STABLE_DURATION - 1e-9:
        raise ValueError('hold the leader still for at least 0.3 seconds')
    if any(not 0 < b - a <= MAX_SAMPLE_GAP for a, b in zip(times, times[1:])):
        raise ValueError('leader samples must be consecutive and time ordered')
    final = samples[-1][0]
    if any(_distance(a, b) > MAX_STABLE_MOVEMENT for positions, _ in samples
           for a, b in zip(positions, final)):
        raise ValueError('hold all leader joints still while saving home')
    home = {'version': 1, 'positions_rad': final, 'tolerances_rad': tolerances}
    path = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.leader-home-', suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(home, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return home


def load_home(path):
    """Load validated local calibration. Missing and corrupt files fail closed."""
    try:
        with Path(path).open(encoding='utf-8') as stream:
            return _home(json.load(stream))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError('invalid leader home file') from exc


def compare_home(home, sample, now):
    """Return (within all seven tolerances, angular errors in radians)."""
    home = _home(home)
    positions, timestamp = _sample(sample, now)
    if now - timestamp > MAX_SAMPLE_AGE:
        raise ValueError('leader sample is stale')
    errors = [_distance(a, b) for a, b in zip(positions, home['positions_rad'])]
    return all(error <= limit + 1e-12 for error, limit in zip(errors, home['tolerances_rad'])), errors
