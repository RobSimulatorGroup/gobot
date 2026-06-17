#include "manual_bindings_internal.hpp"

namespace gobot::python {

void RegisterManualNodeBindings(PyNodeClass& node_class, PyNode3DClass& node3d_class) {
    node_class
            .def_property_readonly("id", &NodeGetId)
            .def_property("name", &NodeGetName, &NodeSetName)
            .def_property_readonly("type", &NodeTypeName)
            .def_property_readonly("type_name", &NodeTypeName)
            .def_property_readonly("path", &NodeGetPath)
            .def_property_readonly("valid", &NodeIsValid)
            .def_property_readonly("child_count", [](const PyNodeHandle& handle) {
                return handle.Resolve()->GetChildCount();
            })
            .def_property_readonly("children", &GetNodeChildren)
            .def_property_readonly("parent", &NodeGetParent)
            .def("child", &GetNodeChild, py::arg("index"))
            .def("find", &FindNode, py::arg("path"))
            .def("add_child", &NodeAddChild, py::arg("child"), py::arg("force_readable_name") = false)
            .def("remove_child", &NodeRemoveChild, py::arg("child"), py::arg("delete") = false)
            .def("remove", [](PyNodeHandle& handle, bool delete_node) -> py::object {
                Node* node = handle.Resolve();
                Node* parent = node->GetParent();
                if (parent == nullptr) {
                    if (delete_node) {
                        throw std::invalid_argument("cannot remove the root node through a node handle");
                    }
                    return py::none();
                }
                PyNodeHandle parent_handle = MakeTypedNodeHandle(parent);
                return NodeRemoveChild(parent_handle, handle, delete_node);
            }, py::arg("delete") = true)
            .def("reparent", &NodeReparent, py::arg("parent"))
            .def("get", &NodeGetProperty, py::arg("property"))
            .def("get_property", &NodeGetProperty, py::arg("property"))
            .def("set", &NodeSetProperty, py::arg("property"), py::arg("value"))
            .def("set_property", &NodeSetProperty, py::arg("property"), py::arg("value"))
            .def("property_names", &NodeGetPropertyNames)
            .def("to_dict", &NodeToDict)
            .def("__repr__", [](const PyNodeHandle& handle) {
                if (!NodeIsValid(handle)) {
                    return std::string("<gobot.Node invalid>");
                }
                return "<gobot." + NodeTypeName(handle) + " name='" + NodeGetName(handle) + "'>";
            });

    node3d_class
            .def_property("position",
                          [](const PyNode3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Node3D>()->GetPosition());
                          },
                          [](PyNode3DHandle& handle, const py::handle& value) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSetNodeProperty(node, "position", Variant(PythonToVector3(value)));
                          })
            .def_property("rotation_degrees",
                          [](const PyNode3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Node3D>()->GetEulerDegree());
                          },
                          [](PyNode3DHandle& handle, const py::handle& value) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSetNodeProperty(node, "rotation_degrees", Variant(PythonToVector3(value)));
                          })
            .def_property("scale",
                          [](const PyNode3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Node3D>()->GetScale());
                          },
                          [](PyNode3DHandle& handle, const py::handle& value) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSetNodeProperty(node, "scale", Variant(PythonToVector3(value)));
                          })
            .def_property("visible",
                          [](const PyNode3DHandle& handle) {
                              return handle.ResolveAs<Node3D>()->IsVisible();
                          },
                          [](PyNode3DHandle& handle, bool visible) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSetNodeProperty(node, "visible", Variant(visible));
                          })
            .def("set_global_transform",
                          [](PyNode3DHandle& handle,
                             const py::handle& position,
                             const py::handle& orientation) {
                              handle.ResolveAs<Node3D>()->SetGlobalTransform(PythonToTransformWxyz(position, orientation));
                          })
            .def("set_transform",
                          [](PyNode3DHandle& handle,
                             const py::handle& position,
                             const py::handle& orientation) {
                              handle.ResolveAs<Node3D>()->SetTransform(PythonToTransformWxyz(position, orientation));
                          });
}

} // namespace gobot::python
