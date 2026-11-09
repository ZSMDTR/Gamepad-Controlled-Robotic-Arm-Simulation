import math
import threading
import time
import unittest
from unittest.mock import patch

from arm_teleop.leader_serial import LeaderReader, _query, _read_position


def response(servo_id, tenths):
    frame = bytes((5, 0x1c, 0x0a, 3, servo_id)) + tenths.to_bytes(2, 'little', signed=True)
    return frame + bytes((sum(frame) & 0xff,))


class FakeSerial:
    def __init__(self, reply=None):
        self.reply = reply or (lambda servo_id, count: response(servo_id, servo_id * 100 - 300))
        self.data = bytearray()
        self.writes = []
        self.counts = [0] * 7
        self.closed = False
        self.dtr = self.rts = None

    def open(self):
        pass

    def close(self):
        self.closed = True

    def reset_input_buffer(self):
        self.data.clear()

    @property
    def in_waiting(self):
        # Emulate very fragmented serial delivery.
        return 0

    def write(self, frame):
        self.writes.append(frame)
        servo_id = frame[4]
        self.counts[servo_id] += 1
        self.data.extend(self.reply(servo_id, self.counts[servo_id]))
        return len(frame)

    def read(self, size):
        if not self.data:
            time.sleep(0.001)
            return b''
        data = bytes(self.data[:size])
        del self.data[:size]
        return data


def wait_until(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


class LeaderSerialTests(unittest.TestCase):
    def test_vendor_query_and_fragmented_response(self):
        self.assertEqual(_query(0), bytes.fromhex('12 4c 0a 01 00 69'))
        device = FakeSerial(lambda sid, count: response(sid, -123))
        result = _read_position(device, 0, threading.Event())
        self.assertAlmostEqual(result, math.radians(-12.3))

    def test_ignores_bad_checksum_wrong_id_and_noise(self):
        bad = bytearray(response(0, 999))
        bad[-1] ^= 1
        device = FakeSerial(lambda sid, count: b'\x98\x05' + bytes(bad) +
                            response(1, 888) + response(sid, 125))
        result = _read_position(device, 0, threading.Event())
        self.assertAlmostEqual(result, math.radians(12.5))

    def test_bad_or_wrong_response_times_out(self):
        bad = bytearray(response(0, 123))
        bad[-1] ^= 1
        device = FakeSerial(lambda sid, count: bytes(bad) + response(1, 123))
        self.assertIsNone(_read_position(device, 0, threading.Event(), timeout=0.02))

    def test_rejects_out_of_range_angle(self):
        device = FakeSerial(lambda sid, count: response(sid, 1801))
        self.assertIsNone(_read_position(device, 0, threading.Event(), timeout=0.02))

    def test_complete_sample_and_closes(self):
        device = FakeSerial()
        with patch('arm_teleop.leader_serial.serial.Serial', return_value=device):
            started = time.monotonic()
            reader = LeaderReader('test').start()
            self.assertTrue(wait_until(lambda: reader.latest() is not None))
            sample = reader.latest()
            self.assertEqual(len(sample.positions), 7)
            self.assertGreaterEqual(sample.timestamp, started)
            self.assertLessEqual(sample.timestamp, time.monotonic())
            for servo_id, value in enumerate(sample.positions):
                self.assertAlmostEqual(value, math.radians(servo_id * 10 - 30))
            self.assertIsNone(reader.error)
            reader.close()
        self.assertTrue(device.closed)
        self.assertIsNone(reader.latest())
        self.assertEqual(device.writes[:7], [_query(i) for i in range(7)])
        self.assertTrue(all(frame[2] == 0x0a for frame in device.writes))
        self.assertFalse(device.dtr)
        self.assertFalse(device.rts)

    def test_publishes_each_axis_after_first_complete_snapshot(self):
        device = FakeSerial(lambda sid, count: response(sid, (count - 1) * 100 + sid))
        reader = LeaderReader('test')
        clock = [100.]
        snapshots = []

        def after_query(seconds):
            self.assertEqual(seconds, .005)
            snapshots.append(reader.latest())
            clock[0] += seconds
            if len(snapshots) == 9:
                reader._stop.set()

        with patch('arm_teleop.leader_serial.serial.Serial', return_value=device), \
             patch('arm_teleop.leader_serial.time.monotonic', side_effect=lambda: clock[0]), \
             patch('arm_teleop.leader_serial.time.sleep', side_effect=after_query):
            reader._run()

        self.assertEqual(snapshots[:6], [None] * 6)
        first, second, third = snapshots[6:]
        self.assertAlmostEqual(first.timestamp, 100.)
        self.assertAlmostEqual(second.timestamp, 100.005)
        self.assertAlmostEqual(third.timestamp, 100.010)
        self.assertAlmostEqual(first.positions[0], 0.)
        self.assertAlmostEqual(second.positions[0], math.radians(10.))
        self.assertEqual(second.positions[1:], first.positions[1:])
        self.assertAlmostEqual(third.positions[1], math.radians(10.1))
        self.assertEqual(third.positions[2:], second.positions[2:])

    def test_oldest_axis_timestamp_reveals_delayed_snapshot(self):
        from arm_teleop.server import Simulation

        clock = [100.]

        class DelayedSerial(FakeSerial):
            def write(self, frame):
                if frame[4] == 6:
                    clock[0] += .6
                return super().write(frame)

        reader = LeaderReader('test')
        device = DelayedSerial()
        seen = []

        def after_query(seconds):
            clock[0] += seconds
            if reader.latest() is not None:
                seen.append((reader.latest(), clock[0]))
                reader._stop.set()

        with patch('arm_teleop.leader_serial.serial.Serial', return_value=device), \
             patch('arm_teleop.leader_serial.time.monotonic', side_effect=lambda: clock[0]), \
             patch('arm_teleop.leader_serial.time.sleep', side_effect=after_query):
            reader._run()
        self.assertEqual(len(seen), 1)
        snapshot, now = seen[0]
        self.assertAlmostEqual(snapshot.timestamp, 100.)
        self.assertGreater(now - snapshot.timestamp, .5)
        self.assertIsNone(Simulation._leader_values(snapshot, now))

    def test_disconnect_clears_last_sample_and_reports_error(self):
        device = FakeSerial(lambda sid, count: b'' if sid == 3 and count > 1
                            else response(sid, 100))
        with patch('arm_teleop.leader_serial.serial.Serial', return_value=device):
            reader = LeaderReader('test').start()
            self.assertTrue(wait_until(lambda: reader.latest() is not None))
            self.assertTrue(wait_until(lambda: reader.error is not None))
            self.assertIsNone(reader.latest())
            self.assertIn('ID 3', reader.error)
            reader.close()
        self.assertTrue(device.closed)

    def test_stop_during_query_interval_sends_no_more_commands(self):
        device = FakeSerial()
        reader = LeaderReader('test')

        def stop_during_sleep(seconds):
            self.assertEqual(seconds, 0.005)
            reader._stop.set()

        with patch('arm_teleop.leader_serial.serial.Serial', return_value=device), \
             patch('arm_teleop.leader_serial.time.sleep', side_effect=stop_during_sleep):
            reader._run()
        self.assertEqual(device.writes, [_query(0)])
        self.assertTrue(device.closed)
        self.assertIsNone(reader.latest())
        self.assertIsNone(reader.error)

    def test_open_failure_reported(self):
        with patch('arm_teleop.leader_serial.serial.Serial', side_effect=OSError('unavailable')):
            reader = LeaderReader('test').start()
            self.assertTrue(wait_until(lambda: reader.error is not None))
            self.assertIsNone(reader.latest())
            self.assertIn('unavailable', reader.error)
            reader.close()


if __name__ == '__main__':
    unittest.main()
