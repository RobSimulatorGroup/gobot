#include "manual_bindings_internal.hpp"

namespace gobot::python {

void RegisterManualCommonBindings(py::module_& module) {

    module.attr("__version__") = Engine::GetVersionString();
    module.def("version", []() {
        return Engine::GetVersionString();
    });
    module.def("version_info", []() {
        py::dict info;
        info["major"] = GOBOT_VERSION_MAJOR;
        info["minor"] = GOBOT_VERSION_MINOR;
        info["patch"] = GOBOT_VERSION_PATCH;
        info["commit"] = Engine::GetBuildCommit();
        return info;
    });

    py::exec(R"(
class NodeScript:
    """Base class for Python scripts attached to Gobot scene nodes."""

    def __init__(self):
        self.node = None
        self.root = None
        self.context = None

    def _attach(self, node, root, context):
        self.node = node
        self.root = root
        self.context = context

    def get_node(self, path: str):
        if path == "":
            return self.node
        if path.startswith("/"):
            if self.root is None:
                return None
            if path == "/":
                return self.root
            return self.root.find(path[1:])
        if self.node is None:
            return None
        return self.node.find(path)

    def get_root(self):
        return self.root
)",
             module.attr("__dict__"),
             module.attr("__dict__"));

    py::enum_<JointType>(module, "JointType")
            .value("Fixed", JointType::Fixed)
            .value("Revolute", JointType::Revolute)
            .value("Continuous", JointType::Continuous)
            .value("Prismatic", JointType::Prismatic)
            .value("Floating", JointType::Floating)
            .value("Planar", JointType::Planar)
            .export_values();

    py::enum_<JointDriveMode>(module, "JointDriveMode")
            .value("Passive", JointDriveMode::Passive)
            .value("Motor", JointDriveMode::Motor)
            .value("Position", JointDriveMode::Position)
            .value("Velocity", JointDriveMode::Velocity)
            .export_values();

    py::enum_<RobotMode>(module, "RobotMode")
            .value("Assembly", RobotMode::Assembly)
            .value("Motion", RobotMode::Motion)
            .export_values();

    py::enum_<RayReductionMode>(module, "RayReductionMode")
            .value("None_", RayReductionMode::None)
            .value("Min", RayReductionMode::Min)
            .value("Max", RayReductionMode::Max)
            .value("Mean", RayReductionMode::Mean)
            .export_values();

    py::enum_<RayPatternMode>(module, "RayPatternMode")
            .value("Custom", RayPatternMode::Custom)
            .value("Grid", RayPatternMode::Grid)
            .export_values();

    py::enum_<RayAlignmentMode>(module, "RayAlignmentMode")
            .value("World", RayAlignmentMode::World)
            .value("Base", RayAlignmentMode::Base)
            .value("Yaw", RayAlignmentMode::Yaw)
            .export_values();

    py::class_<Input>(module, "Input")
            .def_property_readonly("has_control_focus", &Input::HasControlFocus)
            .def("is_key_pressed", &Input::IsKeyPressedByName, py::arg("key_name"))
            .def("is_key_held", &Input::IsKeyHeldByName, py::arg("key_name"));

    py::enum_<LinkRole>(module, "LinkRole")
            .value("Physical", LinkRole::Physical)
            .value("VirtualRoot", LinkRole::VirtualRoot)
            .export_values();

    py::enum_<TerrainColorMode>(module, "TerrainColorMode")
            .value("SurfaceColor", TerrainColorMode::SurfaceColor)
            .value("HeightRamp", TerrainColorMode::HeightRamp)
            .value("Palette", TerrainColorMode::Palette)
            .export_values();
}

} // namespace gobot::python
