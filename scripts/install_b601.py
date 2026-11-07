"""Fetch and adapt the upstream B601-RS model locally; never overwrite an installation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "39e19faf7e235ffb912ecaa2d2a9bc86cd5fa6ca"
URL = f"https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/{COMMIT}/Rebot_Arm_description/RS/"
ROOT = Path(__file__).resolve().parents[1]
HASHES = json.loads((Path(__file__).with_name("b601_source_hashes.json")).read_text(encoding="utf-8"))


def get_file(name, digest, source, destination):
    if source:
        data = (source / name).read_bytes()
    else:
        with urllib.request.urlopen(URL + name, timeout=90) as response:
            data = response.read()
    if name == "mjcf/rs_arm.xml":
        data = data.replace(b"\r\n", b"\n")  # Git on Windows may check out text as CRLF.
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch for {name}; installation stopped")
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def adapt(source, output):
    root = ET.parse(source / "mjcf/rs_arm.xml").getroot()
    root.find("compiler").set("meshdir", "meshes")
    root.insert(1, ET.Element("option", timestep="0.002", integrator="implicitfast"))
    assets = root.find("asset")
    for element in list(assets):
        if element.get("name", "").startswith("wrist_"):
            assets.remove(element)
    for body in root.iter("body"):
        body.set("gravcomp", "1")
        for geom in list(body.findall("geom")):
            if geom.get("name", "").startswith("wrist_"):
                body.remove(geom)
        for camera in list(body.findall("camera")):
            body.remove(camera)
    tcp = root.find(".//site[@name='tcp']")
    tcp.set("pos", "-0.08 0 0")
    tcp.set("rgba", "1 0 0 1")
    ET.SubElement(root.find("contact"), "exclude", body1="base_link", body2="link1")
    actuator = root.find("actuator")
    actuator.clear()
    for i, (kp, kv, limits, force) in enumerate([
        ("300", "20", "-2.8 2.8", "-36 36"),
        ("300", "20", "-0.2 3.14", "-36 36"),
        ("300", "20", "-0.2 3.14", "-36 36"),
        ("100", "8", "-1.57 1.57", "-14 14"),
        ("100", "8", "-1.57 1.57", "-14 14"),
        ("100", "8", "-3.14 3.14", "-14 14"),
    ], 1):
        ET.SubElement(actuator, "position", name=f"joint{i}_position", joint=f"joint{i}",
                      kp=kp, kv=kv, ctrlrange=limits, forcerange=force)
    ET.SubElement(actuator, "position", name="gripper_position", joint="joint7", kp="500", kv="20",
                  ctrlrange="0 0.05", forcerange="-100 100")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos="0 1 1.3 0.3 0 0 0.025 0.025 0.025",
                  ctrl="0 1 1.3 0.3 0 0 0.025")
    ET.indent(root, space="  ")
    output.mkdir()
    ET.ElementTree(root).write(output / "model.xml", encoding="unicode")
    shutil.copytree(source / "meshes", output / "meshes")
    profile = {
        "model": "model.xml", "tcp": "tcp", "home_key": "home",
        "joints": [f"joint{i}" for i in range(1, 7)],
        "actuators": [f"joint{i}_position" for i in range(1, 7)],
        "velocity": [0.4] * 6, "acceleration": [1.0] * 6,
        "gripper": {"joints": ["joint_left", "joint_right"],
                    "bodies": ["gripper_left", "gripper_right"], "closed": [0, 0],
                    "open": [0.05, 0.05], "actuators": ["gripper_position"],
                    "control_closed": [0], "control_open": [0.05]},
    }
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    (output / "SOURCE.txt").write_text(
        f"Locally adapted from Seeed-Projects/reBot-DevArm at {COMMIT}, Rebot_Arm_description/RS.\n"
        "https://github.com/Seeed-Projects/reBot-DevArm/tree/" + COMMIT + "/Rebot_Arm_description/RS\n"
        "The source README records no upstream LICENSE file at the original MJCF revision; rights remain unresolved. "
        "Do not redistribute the generated model or assets without resolving rights.\n"
        "Servo gains, gravity compensation, TCP and limits are simulation approximations, not hardware specifications.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rs", type=Path, help="optional already downloaded official RS directory (verified by SHA-256)")
    parser.add_argument("--output", type=Path, default=ROOT / "robot_models/b601_rs")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error(f"{target} already exists; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="b601-install-", dir=target.parent) as temp:
        temp = Path(temp)
        source = temp / "source"
        source.mkdir()
        print(f"Fetching and verifying {len(HASHES)} files from official commit {COMMIT}...", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(get_file, name, digest, args.source_rs, source)
                       for name, digest in HASHES.items()]
            for future in futures:
                future.result()
        generated = temp / "generated"
        adapt(source, generated)
        # Installation appears only when all downloads and transformations succeeded.
        generated.rename(target)
    print(f"Installed locally at {target}")


if __name__ == "__main__":
    main()
