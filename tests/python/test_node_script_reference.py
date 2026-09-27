"""Python-authored script references use scene serialization and command history."""
import json
from pathlib import Path
import tempfile
import unittest

import gobot


class NodeScriptReferenceTest(unittest.TestCase):
    def test_attach_save_reload_clear_and_undo(self):
        context = gobot.app.context()
        previous_project = context.project_path or str(Path.cwd())
        with tempfile.TemporaryDirectory(prefix="gobot-script-reference-") as directory:
            project = Path(directory)
            # Resource loading and scene authoring must not execute the script.
            (project / "control.py").write_text('raise RuntimeError("must not execute while authoring")\n')
            context.set_project_path(directory)
            try:
                root = gobot.create_node("Node3D", "root")
                child = gobot.create_node("Node3D", "child")
                root.add_child(child)
                root.set("script", "res://control.py")
                child.set_property("script", "res://control.py")

                def saved(node):
                    gobot.save_scene(node, "res://scene.jscn")
                    return json.loads((project / "scene.jscn").read_text())

                def script_path(scene, index=0):
                    reference = scene["__NODES__"][index]["properties"]["script"]
                    if reference is None:
                        return None
                    return next(resource["__PATH__"] for resource in scene["__EXT_RESOURCES__"]
                                if reference == f'ExtResource({resource["__ID__"]})')

                scene = saved(root)
                references = scene["__EXT_RESOURCES__"]
                self.assertEqual(len(references), 1)
                self.assertEqual(references[0]["__TYPE__"], "PythonScript")
                self.assertEqual(references[0]["__PATH__"], "res://control.py")
                reference = f'ExtResource({references[0]["__ID__"]})'
                self.assertTrue(all(node["properties"]["script"] == reference for node in scene["__NODES__"]))

                root = context.load_scene("res://scene.jscn")
                child = root.find("child")
                with self.assertRaises(RuntimeError):
                    root.set("script", "res://missing.py")
                self.assertEqual(script_path(saved(root)), "res://control.py")
                root.set("script", None)
                self.assertIsNone(saved(root)["__NODES__"][0]["properties"]["script"])
                self.assertTrue(context.undo())
                self.assertEqual(script_path(saved(root)), "res://control.py")
                self.assertTrue(context.redo())
                child.set("script", None)
                scene = saved(root)
                self.assertEqual(scene["__EXT_RESOURCES__"], [])
                self.assertTrue(all(node["properties"]["script"] is None for node in scene["__NODES__"]))
            finally:
                context.clear_scene()
                context.set_project_path(previous_project)


if __name__ == "__main__":
    unittest.main()
