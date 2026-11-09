"""Read the seven Fashion Star leader positions without commanding motion.

Only the single-turn angle query (0x0a) is sent.  Values have not been
calibrated or mapped to the simulated robot.  A complete sample's timestamp is
the start of its first query, so it is a conservative estimate of sample age.
"""
from dataclasses import dataclass
import math
import threading
import time

import serial


@dataclass(frozen=True)
class LeaderSample:
    positions: tuple[float, ...]
    timestamp: float


def _query(servo_id):
    frame = bytes((0x12, 0x4c, 0x0a, 0x01, servo_id))
    return frame + bytes((sum(frame) & 0xff,))


def _read_position(port, servo_id, stop, timeout=0.05):
    """Read one matching, checksummed reply; ignore corrupt and unrelated frames."""
    request = _query(servo_id)
    if port.write(request) != len(request):
        raise OSError('Incomplete serial query write')
    deadline = time.monotonic() + timeout
    data = bytearray()
    while not stop.is_set() and time.monotonic() < deadline:
        # A short serial timeout bounds shutdown and permits fragmented replies.
        data.extend(port.read(1))
        waiting = port.in_waiting
        if waiting:
            data.extend(port.read(waiting))
        while len(data) >= 2:
            if data[:2] != b'\x05\x1c':
                del data[0]
                continue
            if len(data) < 8:
                break
            frame = data[:8]
            if frame[2:4] != b'\x0a\x03' or sum(frame[:-1]) & 0xff != frame[-1]:
                del data[0]
                continue
            del data[:8]
            if frame[4] == servo_id:
                angle = int.from_bytes(frame[5:7], 'little', signed=True) / 10
                if -180 <= angle <= 180:
                    return math.radians(angle)
    return None


class LeaderReader:
    """Background reader. ``error`` explains a failed or closed connection.

    ``latest()`` returns None until seven valid responses arrive and immediately
    after any query fails. Consumers must also check the monotonic timestamp for
    staleness. A failed reader must be closed and replaced to reconnect.
    """
    def __init__(self, port):
        self.port = port
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample = None
        self._error = None
        self._thread = None

    @property
    def error(self):
        with self._lock:
            return self._error

    def start(self):
        if self._thread is not None:
            raise RuntimeError('LeaderReader can only be started once')
        self._thread = threading.Thread(target=self._run, name='leader-reader', daemon=True)
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._sample

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._sample = None

    def _run(self):
        device = None
        try:
            # Set modem lines before opening, rather than intentionally toggling
            # them on an attached serial adapter.
            device = serial.Serial(port=None, baudrate=1000000, timeout=0.01,
                                   write_timeout=0.05)
            device.dtr = False
            device.rts = False
            device.port = self.port
            device.open()
            device.reset_input_buffer()
            while not self._stop.is_set():
                positions = []
                started = time.monotonic()
                for servo_id in range(7):
                    value = _read_position(device, servo_id, self._stop)
                    if self._stop.is_set():
                        return
                    if value is None:
                        raise TimeoutError(f'No valid response from servo ID {servo_id}')
                    positions.append(value)
                    # Vendor recommends 5–10 ms between successive commands.
                    if self._stop.wait(0.005):
                        return
                with self._lock:
                    self._sample = LeaderSample(tuple(positions), started)
        except Exception as exc:
            with self._lock:
                self._sample = None
                self._error = str(exc)
        finally:
            if device is not None:
                device.close()
