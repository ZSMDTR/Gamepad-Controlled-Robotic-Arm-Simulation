"""Build a local B601 tabletop scene from the user's installed robot model.

The upstream model and meshes are only read on this machine. The generated
scene and profile are disposable files; no upstream assets are redistributed.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def make_b601_scene_profile(source_profile_path, output_directory):
    """Return a profile for a tabletop and a free 24 mm cube in MuJoCo."""
    source_profile_path = Path(source_profile_path).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    profile = json.loads(source_profile_path.read_text(encoding='utf-8'))
    robot_xml = (source_profile_path.parent / profile['model']).resolve()
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    root.set('model', 'B601-RS Dual View')
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '.')
    compiler.set('meshdir', str((robot_xml.parent / meshdir).resolve()))
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='practice_floor', type='plane',
                  size='2 2 .1', rgba='.35 .38 .4 1')
    ET.SubElement(world, 'geom', name='practice_table', type='box',
                  pos='.42 0 .255', size='.23 .25 .015',
                  rgba='.55 .43 .30 1', friction='1 .01 .001')
    cube = ET.SubElement(world, 'body', name='practice_cube', pos='.4 0 .282')
    ET.SubElement(cube, 'freejoint', name='practice_cube_free')
    ET.SubElement(cube, 'geom', name='practice_cube_geom', type='box',
                  size='.012 .012 .012', mass='.02', rgba='.95 .28 .08 1',
                  friction='1.5 .04 .01', condim='6')
    # A free joint adds seven qpos values to every model keyframe.
    for key in root.findall('./keyframe/key'):
        if key.get('qpos') is not None:
            key.set('qpos', key.get('qpos') + ' .4 0 .282 1 0 0 0')
    scene_file = output_directory / 'b601_practice_scene.xml'
    tree.write(scene_file, encoding='unicode')
    profile['model'] = str(scene_file)
    profile_file = output_directory / 'b601_practice_profile.json'
    profile_file.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    return profile_file
