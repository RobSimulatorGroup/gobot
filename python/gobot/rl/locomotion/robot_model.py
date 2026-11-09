"""Robot physics identity independent of the authored evaluation terrain."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def robot_physics_digest(artifact, robot_name):
    root = ET.fromstring(artifact.content)
    prefix = artifact.robot_prefix(robot_name)
    bodies = [body for body in root.findall('./worldbody/body') if body.get('name', '').startswith(prefix)]
    if not bodies:
        raise ValueError('compiled artifact has no robot subtree')
    def node(element):
        attrs = dict(sorted(element.attrib.items()))
        if 'file' in attrs:
            path = Path(attrs['file'])
            attrs['file'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [element.tag, attrs, [node(child) for child in element]]
    referenced = {element.get('mesh') for body in bodies for element in body.iter() if element.get('mesh')}
    assets = [node(a) for a in root.findall('./asset/mesh') if a.get('name') in referenced]
    exclusions = [node(a) for a in root.findall('./contact/exclude') if a.get('body1', '').startswith(prefix)]
    controls = [node(a) for a in root.findall('./actuator/*') if a.get('joint', '').startswith(prefix)]
    # Physics options are part of the contract. Runtime terrain geometry and origin are not.
    options = root.find('option')
    payload = [node(options) if options is not None else None, [node(b) for b in bodies], assets, controls, exclusions]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
