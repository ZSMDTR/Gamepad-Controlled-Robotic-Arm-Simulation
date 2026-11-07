"""Tests for the local-only B601 model installer, with synthetic inputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_b601.py"
spec = importlib.util.spec_from_file_location("install_b601", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class B601InstallerTests(unittest.TestCase):
    def test_rejects_changed_download(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            (source / "asset.stl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                installer.get_file("asset.stl", "0" * 64, source, Path(temp) / "out")
            self.assertFalse((Path(temp) / "out" / "asset.stl").exists())

    def test_adapts_locally_without_upstream_camera_and_produces_profile(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "source"
            (source / "mjcf").mkdir(parents=True)
            (source / "meshes").mkdir()
            (source / "meshes" / "example.stl").write_bytes(b"synthetic mesh")
            (source / "mjcf" / "rs_arm.xml").write_text('''<mujoco>
                <compiler meshdir="../meshes"/><asset><mesh name="wrist_test" file="x"/></asset>
                <worldbody><body name="gripper_end"><geom name="wrist_test"/>
                <camera name="wrist_rgb"/><site name="tcp" pos="0 0 0"/></body></worldbody>
                <contact/><actuator><motor name="old"/></actuator></mujoco>''', encoding="utf-8")
            output = temp / "result"
            installer.adapt(source, output)
            root = ET.parse(output / "model.xml").getroot()
            self.assertEqual(root.find("compiler").get("meshdir"), "meshes")
            self.assertIsNone(root.find(".//*[@name='wrist_test']"))
            self.assertIsNone(root.find(".//camera"))
            self.assertEqual(root.find(".//site[@name='tcp']").get("pos"), "-0.08 0 0")
            self.assertEqual(len(root.find("actuator")), 7)
            self.assertEqual(root.find("keyframe/key").get("name"), "home")
            profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["tcp"], "tcp")
            self.assertEqual(len(profile["joints"]), 6)


if __name__ == "__main__":
    unittest.main()
