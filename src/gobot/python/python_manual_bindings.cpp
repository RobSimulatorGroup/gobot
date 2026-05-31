#include "gobot/python/python_binding_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/python/python_script_runner.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/node_creation_registry.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_command.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/scene/sensor_3d.hpp"
#include "gobot/scene/terrain_3d.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {
namespace {

enum class PyNodeOwnership {
    Borrowed,
    DetachedOwned,
    TreeOwned
};

struct PyNodeHandleState {
    ObjectID id;
    std::uint64_t scene_epoch{0};
    std::string name_snapshot;
    PyNodeOwnership ownership{PyNodeOwnership::Borrowed};
};

struct PyNodeHandle {
    std::shared_ptr<PyNodeHandleState> state;
    std::string expected_type;

    static std::shared_ptr<PyNodeHandleState> MakeState(Node* node,
                                                        std::uint64_t epoch,
                                                        PyNodeOwnership node_ownership) {
        auto state = std::make_shared<PyNodeHandleState>();
        state->id = node != nullptr ? node->GetInstanceId() : ObjectID();
        state->scene_epoch = epoch;
        state->name_snapshot = node != nullptr ? node->GetName() : std::string();
        state->ownership = node_ownership;
        return state;
    }

    PyNodeHandle() = default;

    PyNodeHandle(Node* node,
                 std::string expected_type_name,
                 std::uint64_t epoch,
                 PyNodeOwnership node_ownership)
        : state(MakeState(node, epoch, node_ownership)),
          expected_type(std::move(expected_type_name)) {
    }

    PyNodeHandle(std::shared_ptr<PyNodeHandleState> handle_state,
                 std::string expected_type_name)
        : state(std::move(handle_state)),
          expected_type(std::move(expected_type_name)) {
    }

    ~PyNodeHandle() {
        ReleaseDetached();
    }

    void ReleaseDetached() {
        if (!state || state.use_count() != 1 || state->ownership != PyNodeOwnership::DetachedOwned) {
            return;
        }

        auto* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(state->id));
        if (node != nullptr && node->GetParent() == nullptr) {
            Object::Delete(node);
        }
        state->ownership = PyNodeOwnership::Borrowed;
        state->id = ObjectID();
    }

    Node* TryResolve() const {
        if (!state) {
            return nullptr;
        }
        auto* object = ObjectDB::GetInstance(state->id);
        auto* node = Object::PointerCastTo<Node>(object);
        if (node == nullptr) {
            return nullptr;
        }
        if (!expected_type.empty()) {
            const Type type = Type::get_by_name(expected_type);
            if (type.is_valid() && node->GetType() != type && !node->GetType().is_derived_from(type)) {
                return nullptr;
            }
        }
        return node;
    }

    Node* Resolve() const;

    template <typename T>
    T* ResolveAs() const {
        Node* node = Resolve();
        auto* typed = Object::PointerCastTo<T>(node);
        if (typed == nullptr) {
            throw py::type_error("Gobot node handle expected type '" + std::string(Type::get<T>().get_name().data()) +
                                 "' but resolved '" + std::string(node->GetClassStringName()) + "'");
        }
        return typed;
    }

    void RefreshSnapshot(Node* node) {
        if (state && node != nullptr) {
            state->name_snapshot = node->GetName();
        }
    }

    void TransferToTree() {
        if (state) {
            state->ownership = PyNodeOwnership::TreeOwned;
        }
    }

    void TransferToDetachedOwned(Node* node);

    void Invalidate() {
        if (state) {
            state->ownership = PyNodeOwnership::Borrowed;
            state->id = ObjectID();
        }
    }
};

struct PyNode3DHandle : public PyNodeHandle {
    using PyNodeHandle::PyNodeHandle;
};

struct PyRobot3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyLink3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyJoint3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyCollisionShape3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyMeshInstance3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyTerrain3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PySensor3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

struct PyIMUSensor3DHandle : public PySensor3DHandle {
    using PySensor3DHandle::PySensor3DHandle;
};

struct PyAngularMomentumSensor3DHandle : public PySensor3DHandle {
    using PySensor3DHandle::PySensor3DHandle;
};

struct PyContactSensor3DHandle : public PySensor3DHandle {
    using PySensor3DHandle::PySensor3DHandle;
};

struct PyTerrainHeightSensor3DHandle : public PySensor3DHandle {
    using PySensor3DHandle::PySensor3DHandle;
};

std::string ExpectedTypeForNode(Node* node);
PyNodeHandle MakeTypedNodeHandle(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed);
py::object MakeTypedNodeObject(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed);

template <typename Func>
void ExecuteSceneMutation(const std::string&, Func&& func) {
    std::unique_ptr<SceneCommand> command = std::forward<Func>(func)();
    if (!GetActiveAppContext().ExecuteSceneCommand(std::move(command))) {
        throw std::runtime_error("failed to execute Gobot scene command");
    }
}

void ExecuteSetNodeProperty(Node* node, const std::string& property_name, Variant value);

std::uint64_t ActiveSceneEpoch() {
    const std::uint64_t scene_script_epoch = PythonScriptRunner::GetExecutingSceneScriptEpoch();
    if (scene_script_epoch != 0) {
        return scene_script_epoch;
    }
    return GetActiveAppContext().GetSceneEpoch();
}

Node* ActiveSceneRoot() {
    if (Node* scene_script_root = PythonScriptRunner::GetExecutingSceneScriptRoot()) {
        return scene_script_root;
    }
    return GetActiveAppContext().GetSceneRoot();
}

bool IsSceneScriptRuntimeMutation() {
    return PythonScriptRunner::IsExecutingSceneScript();
}

[[noreturn]] void ThrowReferenceError(const std::string& message) {
    PyErr_SetString(PyExc_ReferenceError, message.c_str());
    throw py::error_already_set();
}

Robot3D* CreateTestRobotScene() {
    auto* robot = Object::New<Robot3D>();
    robot->SetName("robot");

    auto* base_link = Object::New<Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.0, 0.0, 1.0});
    base_link->SetHasInertial(true);
    base_link->SetMass(1.0);

    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto box = MakeRef<BoxShape3D>();
    box->SetSize({0.5, 0.5, 0.5});
    collision_shape->SetShape(box);
    base_link->AddChild(collision_shape);

    auto* joint = Object::New<Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);

    auto* tip_link = Object::New<Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(base_link);
    robot->AddChild(joint);
    joint->AddChild(tip_link);
    return robot;
}

CollisionShape3D* CreateBoxCollision(const std::string& name,
                                      const Vector3& size,
                                      const Vector3& position = Vector3::Zero()) {
    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName(name);
    collision_shape->SetPosition(position);
    auto box = MakeRef<BoxShape3D>();
    box->SetSize(size);
    collision_shape->SetShape(box);
    return collision_shape;
}

MeshInstance3D* CreateBoxVisual(const std::string& name,
                                const Vector3& size,
                                const Vector3& position = Vector3::Zero()) {
    auto* mesh_instance = Object::New<MeshInstance3D>();
    mesh_instance->SetName(name);
    mesh_instance->SetPosition(position);
    auto box = MakeRef<BoxMesh>();
    box->SetSize(size);
    mesh_instance->SetMesh(box);
    return mesh_instance;
}

Node* LoadSceneRoot(const std::string& scene_path) {
    Ref<Resource> resource =
            ResourceLoader::Load(scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to load PackedScene from '" + scene_path + "'");
    }

    Node* scene_root = packed_scene->Instantiate();
    if (scene_root == nullptr) {
        throw std::runtime_error("failed to instantiate PackedScene from '" + scene_path + "'");
    }

    return scene_root;
}

bool SaveSceneRoot(Node* root, const std::string& path) {
    if (root == nullptr) {
        throw std::invalid_argument("cannot save a null Gobot scene root");
    }
    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    if (!packed_scene->Pack(root)) {
        return false;
    }
    USING_ENUM_BITWISE_OPERATORS;
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    return ResourceSaver::Save(packed_scene,
                               global_path,
                               ResourceSaverFlags::ReplaceSubResourcePaths |
                                       ResourceSaverFlags::ChangePath);
}

std::string LocalizeResourcePath(const std::string& path) {
    return ProjectSettings::GetInstance()->LocalizePath(path);
}

std::string GlobalizeResourcePath(const std::string& path) {
    return ProjectSettings::GetInstance()->GlobalizePath(LocalizeResourcePath(path));
}

std::string ResourcePathStem(const std::string& path) {
    return std::filesystem::path(GlobalizeResourcePath(path)).stem().string();
}

std::optional<std::string> ReadFirstMJCFIncludeFile(const std::string& xml_path) {
    std::ifstream input(GlobalizeResourcePath(xml_path));
    if (!input.is_open()) {
        return std::nullopt;
    }

    const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    static const std::regex include_regex(R"gobot(<\s*include\b([^>]*)>)gobot", std::regex::icase);
    static const std::regex file_regex(R"gobot(file\s*=\s*["']([^"']+)["'])gobot", std::regex::icase);

    std::smatch include_match;
    if (!std::regex_search(xml, include_match, include_regex)) {
        return std::nullopt;
    }

    std::smatch file_match;
    const std::string attributes = include_match.size() > 1 ? include_match[1].str() : std::string();
    if (!std::regex_search(attributes, file_match, file_regex) || file_match.size() < 2) {
        return std::nullopt;
    }

    return file_match[1].str();
}

std::string GetXmlAttribute(const std::string& attributes, const std::string& name) {
    const std::regex attr_regex("(^|[[:space:]])" + name + R"(\s*=\s*["']([^"']*)["'])", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(attributes, match, attr_regex) || match.size() < 3) {
        return {};
    }
    return match[2].str();
}

std::vector<RealType> ParseRealList(const std::string& text) {
    std::vector<RealType> values;
    std::istringstream input(text);
    RealType value{};
    while (input >> value) {
        values.push_back(value);
    }
    return values;
}

Color ParseColorOrDefault(const std::string& text, const Color& fallback) {
    const std::vector<RealType> values = ParseRealList(text);
    if (values.size() < 3) {
        return fallback;
    }
    return Color(static_cast<float>(values[0]),
                 static_cast<float>(values[1]),
                 static_cast<float>(values[2]),
                 values.size() >= 4 ? static_cast<float>(values[3]) : 1.0f);
}

std::vector<std::pair<std::string, std::string>> ReadMJCFPlaneGeoms(const std::string& xml_path) {
    std::ifstream input(GlobalizeResourcePath(xml_path));
    if (!input.is_open()) {
        return {};
    }

    const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    static const std::regex geom_regex(R"gobot(<\s*geom\b([^>]*)>)gobot", std::regex::icase);

    std::vector<std::pair<std::string, std::string>> plane_geoms;
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), geom_regex);
    auto end = std::sregex_iterator();
    for (auto iter = begin; iter != end; ++iter) {
        const std::string attributes = iter->size() > 1 ? (*iter)[1].str() : std::string();
        if (ToLower(GetXmlAttribute(attributes, "type")) != "plane") {
            continue;
        }

        std::string name = GetXmlAttribute(attributes, "name");
        if (name.empty()) {
            name = "ground";
        }
        plane_geoms.emplace_back(std::move(name), attributes);
    }
    return plane_geoms;
}

std::optional<std::string> ResolveFirstMJCFIncludeResourcePath(const std::string& xml_path) {
    std::optional<std::string> include_file = ReadFirstMJCFIncludeFile(xml_path);
    if (!include_file.has_value() || include_file->empty()) {
        return std::nullopt;
    }

    if (include_file->find("://") != std::string::npos || IsAbsolutePath(*include_file)) {
        return LocalizeResourcePath(*include_file);
    }

    const std::string local_xml_path = LocalizeResourcePath(xml_path);
    return SimplifyPath(PathJoin(GetBaseDir(local_xml_path), *include_file));
}

std::string MakeIncludedScenePath(const std::string& scene_path, const std::string& included_xml_path) {
    const std::string local_scene_path = LocalizeResourcePath(scene_path);
    return SimplifyPath(PathJoin(GetBaseDir(local_scene_path), ResourcePathStem(included_xml_path) + ".jscn"));
}

void AddMJCFScenePlaneGeoms(Node3D* scene_root, const std::string& xml_path) {
    if (scene_root == nullptr) {
        return;
    }

    for (const auto& [name, attributes] : ReadMJCFPlaneGeoms(xml_path)) {
        const std::vector<RealType> size_values = ParseRealList(GetXmlAttribute(attributes, "size"));
        const RealType size_x = size_values.size() >= 1 ? size_values[0] : 50.0;
        const RealType size_y = size_values.size() >= 2 ? size_values[1] : size_x;
        constexpr RealType kGroundThickness = 0.02;
        const Vector3 ground_size{
                std::max<RealType>(size_x * 2.0, 0.01),
                std::max<RealType>(size_y * 2.0, 0.01),
                kGroundThickness};
        const std::vector<RealType> pos_values = ParseRealList(GetXmlAttribute(attributes, "pos"));
        Vector3 position{0.0, 0.0, -kGroundThickness * 0.5};
        if (pos_values.size() >= 3) {
            position = Vector3{pos_values[0],
                               pos_values[1],
                               static_cast<RealType>(pos_values[2] - kGroundThickness * 0.5)};
        }

        const Color color = ParseColorOrDefault(GetXmlAttribute(attributes, "rgba"), Color(0.45f, 0.45f, 0.45f, 1.0f));

        auto* ground = Object::New<Node3D>();
        ground->SetName(name);
        ground->SetPosition(position);

        auto* visual = Object::New<MeshInstance3D>();
        visual->SetName(name + "_visual");
        visual->SetPosition(Vector3{0.0, 0.0, kGroundThickness * 0.5});
        auto mesh = MakeRef<PlaneMesh>();
        mesh->SetSize(Vector2{ground_size.x(), ground_size.y()});
        auto material = MakeRef<PBRMaterial3D>();
        material->SetAlbedo(color);
        mesh->SetMaterial(dynamic_pointer_cast<Material>(material));
        visual->SetMesh(mesh);
        visual->SetSurfaceColor(color);
        ground->AddChild(visual);

        auto* collision = Object::New<CollisionShape3D>();
        collision->SetName(name + "_collision");
        auto shape = MakeRef<BoxShape3D>();
        shape->SetSize(ground_size);
        collision->SetShape(shape);
        ground->AddChild(collision);

        scene_root->AddChild(ground);
    }
}

Ref<PythonScript> LoadPythonScriptResource(const std::string& script_path) {
    Ref<PythonScript> script = dynamic_pointer_cast<PythonScript>(
            ResourceLoader::Load(script_path, "PythonScript", ResourceFormatLoader::CacheMode::Reuse));
    if (!script.IsValid()) {
        throw std::runtime_error("failed to load Python script '" + script_path + "'");
    }
    return script;
}

void SetOptionalNodeScript(Node* root, const std::optional<std::string>& script_path) {
    if (script_path.has_value() && !script_path->empty()) {
        root->SetScript(LoadPythonScriptResource(*script_path));
    }
}

void CopyMJCFDynamicProperties(Node* target, const Node* source) {
    if (target == nullptr || source == nullptr) {
        return;
    }

    if (auto* target_joint = Object::PointerCastTo<Joint3D>(target)) {
        const auto* source_joint = Object::PointerCastTo<Joint3D>(source);
        if (source_joint != nullptr) {
            target_joint->SetJointPosition(source_joint->GetJointPosition());
            target_joint->SetInitialPosition(source_joint->GetInitialPosition());
            target_joint->SetDriveMode(source_joint->GetDriveMode());
            target_joint->SetDriveStiffness(source_joint->GetDriveStiffness());
            target_joint->SetDriveDamping(source_joint->GetDriveDamping());
            target_joint->SetControlLowerLimit(source_joint->GetControlLowerLimit());
            target_joint->SetControlUpperLimit(source_joint->GetControlUpperLimit());
            target_joint->SetForceLowerLimit(source_joint->GetForceLowerLimit());
            target_joint->SetForceUpperLimit(source_joint->GetForceUpperLimit());
            target_joint->SetGear(source_joint->GetGear());
        }
    }

    if (auto* target_collision = Object::PointerCastTo<CollisionShape3D>(target)) {
        const auto* source_collision = Object::PointerCastTo<CollisionShape3D>(source);
        if (source_collision != nullptr) {
            target_collision->SetFriction(source_collision->GetFriction());
            target_collision->SetContactType(source_collision->GetContactType());
            target_collision->SetContactAffinity(source_collision->GetContactAffinity());
            target_collision->SetContactDimension(source_collision->GetContactDimension());
            target_collision->SetSolref(source_collision->GetSolref());
            target_collision->SetSolimp(source_collision->GetSolimp());
            target_collision->SetMargin(source_collision->GetMargin());
            target_collision->SetGap(source_collision->GetGap());
        }
    }

    std::unordered_map<std::string, const Node*> source_children_by_name;
    for (std::size_t i = 0; i < source->GetChildCount(); ++i) {
        const Node* source_child = source->GetChild(static_cast<int>(i));
        source_children_by_name[source_child->GetName()] = source_child;
    }

    for (std::size_t i = 0; i < target->GetChildCount(); ++i) {
        Node* target_child = target->GetChild(static_cast<int>(i));
        auto source_iter = source_children_by_name.find(target_child->GetName());
        if (source_iter != source_children_by_name.end()) {
            CopyMJCFDynamicProperties(target_child, source_iter->second);
        }
    }
}

Node* InstantiateMJCFRoot(const std::string& xml_path) {
    Ref<Resource> resource =
            ResourceLoader::Load(xml_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to import MJCF scene from '" + xml_path + "'");
    }

    Node* root = packed_scene->Instantiate();
    if (root == nullptr) {
        throw std::runtime_error("failed to instantiate MJCF scene from '" + xml_path + "'");
    }

    if (auto* robot = Object::PointerCastTo<Robot3D>(root)) {
        robot->SetSourcePath(LocalizeResourcePath(xml_path));
    }

    return root;
}

void SaveSingleMJCFScene(const std::string& xml_path,
                         const std::string& scene_path,
                         const std::optional<std::string>& name,
                         const std::optional<std::string>& script_path) {
    Node* root = InstantiateMJCFRoot(xml_path);

    try {
        if (name.has_value() && !name->empty()) {
            root->SetName(*name);
        }

        SetOptionalNodeScript(root, script_path);

        if (!SaveSceneRoot(root, scene_path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + scene_path + "'");
        }
    } catch (...) {
        Object::Delete(root);
        throw;
    }

    Object::Delete(root);
}

bool TrySaveSplitMJCFScene(const std::string& xml_path,
                           const std::string& scene_path,
                           const std::optional<std::string>& name,
                           const std::optional<std::string>& script_path) {
    std::optional<std::string> included_xml_path = ResolveFirstMJCFIncludeResourcePath(xml_path);
    if (!included_xml_path.has_value()) {
        return false;
    }

    const std::string robot_scene_path = MakeIncludedScenePath(scene_path, *included_xml_path);
    if (LocalizeResourcePath(robot_scene_path) == LocalizeResourcePath(scene_path)) {
        return false;
    }

    Node* robot_root = InstantiateMJCFRoot(*included_xml_path);
    Node* compiled_scene_root = InstantiateMJCFRoot(xml_path);
    std::string robot_name = robot_root->GetName().empty() ? ResourcePathStem(*included_xml_path) : robot_root->GetName();

    try {
        CopyMJCFDynamicProperties(robot_root, compiled_scene_root);
        if (robot_root->GetName().empty()) {
            robot_root->SetName(robot_name);
        }
        if (!SaveSceneRoot(robot_root, robot_scene_path)) {
            throw std::runtime_error("failed to save Gobot robot asset to '" + robot_scene_path + "'");
        }
    } catch (...) {
        Object::Delete(robot_root);
        Object::Delete(compiled_scene_root);
        throw;
    }
    Object::Delete(robot_root);
    Object::Delete(compiled_scene_root);

    Ref<PackedScene> robot_scene = dynamic_pointer_cast<PackedScene>(
            ResourceLoader::Load(robot_scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Replace));
    if (!robot_scene.IsValid()) {
        throw std::runtime_error("failed to load generated Gobot robot asset '" + robot_scene_path + "'");
    }
    if (robot_scene->GetPath().empty()) {
        throw std::runtime_error("generated Gobot robot asset has no resource path '" + robot_scene_path + "'");
    }

    auto* scene_root = Object::New<Node3D>();
    try {
        scene_root->SetName(name.has_value() && !name->empty() ? *name : ResourcePathStem(scene_path));
        SetOptionalNodeScript(scene_root, script_path);
        AddMJCFScenePlaneGeoms(scene_root, xml_path);

        Node* robot_instance = robot_scene->Instantiate();
        if (robot_instance == nullptr) {
            throw std::runtime_error("failed to instantiate generated Gobot robot asset '" + robot_scene_path + "'");
        }
        robot_instance->SetName(robot_name);
        robot_instance->SetSceneInstance(robot_scene);
        scene_root->AddChild(robot_instance);

        if (!SaveSceneRoot(scene_root, scene_path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + scene_path + "'");
        }
    } catch (...) {
        Object::Delete(scene_root);
        throw;
    }

    Object::Delete(scene_root);
    return true;
}

void ImportMJCFScene(const std::string& xml_path,
                     const std::string& scene_path,
                     const std::optional<std::string>& name,
                     const std::optional<std::string>& script_path) {
    if (TrySaveSplitMJCFScene(xml_path, scene_path, name, script_path)) {
        return;
    }

    SaveSingleMJCFScene(xml_path, scene_path, name, script_path);
}

PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return PhysicsBackendType::Null;
    }
    if (backend == "mujoco" || backend == "mujoco_cpu") {
        return PhysicsBackendType::MuJoCoCpu;
    }
    if (backend == "mujoco_warp" || backend == "warp") {
        return PhysicsBackendType::MuJoCoWarp;
    }
    throw std::invalid_argument("unknown Gobot physics backend '" + backend + "'");
}

struct PyScene {
    Node* root{nullptr};
    std::uint64_t scene_epoch{0};

    explicit PyScene(Node* root_node)
        : root(root_node),
          scene_epoch(ActiveSceneEpoch()) {
    }

    ~PyScene() {
        if (root != nullptr) {
            Object::Delete(root);
            root = nullptr;
        }
    }

    PyScene(const PyScene&) = delete;
    PyScene& operator=(const PyScene&) = delete;
};

struct PySceneTransaction {
    std::string name;
    bool active{false};

    explicit PySceneTransaction(std::string transaction_name)
        : name(std::move(transaction_name)) {
    }

    PySceneTransaction& Enter() {
        if (!GetActiveAppContext().BeginSceneTransaction(name)) {
            throw std::runtime_error("failed to begin Gobot scene transaction '" + name + "'");
        }
        active = true;
        return *this;
    }

    bool Exit(const py::object& exc_type, const py::object&, const py::object&) {
        if (!active) {
            return false;
        }

        active = false;
        if (exc_type.is_none()) {
            if (!GetActiveAppContext().CommitSceneTransaction()) {
                throw std::runtime_error("failed to commit Gobot scene transaction '" + name + "'");
            }
        } else {
            GetActiveAppContext().CancelSceneTransaction();
        }
        return false;
    }
};

std::string NodeTypeName(const PyNodeHandle& handle) {
    return std::string(handle.Resolve()->GetClassStringName());
}

py::list GetNodeChildren(PyNodeHandle& handle) {
    Node* node = handle.Resolve();
    py::list children;
    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        children.append(MakeTypedNodeObject(node->GetChild(static_cast<int>(index))));
    }
    return children;
}

py::object GetNodeChild(PyNodeHandle& handle, int index) {
    Node* child = handle.Resolve()->GetChild(index);
    if (child == nullptr) {
        throw py::index_error("Gobot node child index is out of range");
    }
    return MakeTypedNodeObject(child);
}

py::object FindNode(PyNodeHandle& handle, const std::string& path) {
    Node* node = handle.Resolve()->GetNodeOrNull(NodePath(path));
    if (node == nullptr) {
        return py::none();
    }
    return MakeTypedNodeObject(node);
}

py::object NodeGetProperty(const PyNodeHandle& handle, const std::string& name) {
    const Node* node = handle.Resolve();
    Variant value = node->Get(name);
    if (!value.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    return VariantToPython(value);
}

void NodeSetProperty(PyNodeHandle& handle, const std::string& name, const py::handle& value) {
    Node* node = handle.Resolve();
    const Property property = node->GetType().get_property(name);
    if (!property.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    Variant variant_value = PythonToVariantForType(value, property.get_type());
    if (IsSceneScriptRuntimeMutation()) {
        if (!node->Set(name, variant_value)) {
            throw std::runtime_error("failed to set Gobot runtime node property '" + name + "'");
        }
    } else {
        ExecuteSetNodeProperty(node, name, std::move(variant_value));
    }
    handle.RefreshSnapshot(node);
}

py::object NodeAddChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool force_readable_name) {
    Node* node = handle.Resolve();
    Node* child = child_handle.Resolve();
    if (IsSceneScriptRuntimeMutation()) {
        node->AddChild(child, force_readable_name);
    } else {
        ExecuteSceneMutation("add_child", [&]() {
            return std::make_unique<AddChildNodeCommand>(node->GetInstanceId(),
                                                        child->GetInstanceId(),
                                                        force_readable_name);
        });
    }
    child_handle.TransferToTree();
    child_handle.RefreshSnapshot(child);
    return MakeTypedNodeObject(child);
}

py::object NodeRemoveChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool delete_child) {
    Node* node = handle.Resolve();
    Node* child = child_handle.Resolve();
    if (child->GetParent() != node) {
        throw std::invalid_argument("Gobot node '" + child->GetName() + "' is not a child of '" +
                                    node->GetName() + "'");
    }

    if (delete_child) {
        if (IsSceneScriptRuntimeMutation()) {
            node->RemoveChild(child);
            Object::Delete(child);
        } else {
            ExecuteSceneMutation("remove_child_delete", [&]() {
                return std::make_unique<RemoveChildNodeCommand>(node->GetInstanceId(), child->GetInstanceId(), true);
            });
        }
        child_handle.Invalidate();
        return py::none();
    }

    std::shared_ptr<PyNodeHandleState> detached_state = child_handle.state;
    if (IsSceneScriptRuntimeMutation()) {
        node->RemoveChild(child);
    } else {
        ExecuteSceneMutation("remove_child_detach", [&]() {
            return std::make_unique<RemoveChildNodeCommand>(node->GetInstanceId(), child->GetInstanceId(), false);
        });
    }
    child_handle.TransferToDetachedOwned(child);
    return py::cast(PyNodeHandle(detached_state, ExpectedTypeForNode(child)));
}

void NodeReparent(PyNodeHandle& handle, PyNodeHandle& parent_handle) {
    Node* node = handle.Resolve();
    Node* parent = parent_handle.Resolve();
    if (IsSceneScriptRuntimeMutation()) {
        node->Reparent(parent);
    } else {
        ExecuteSceneMutation("reparent", [&]() {
            return std::make_unique<ReparentNodeCommand>(node->GetInstanceId(), parent->GetInstanceId());
        });
    }
    handle.TransferToTree();
    handle.RefreshSnapshot(node);
}

py::object NodeGetParent(PyNodeHandle& handle) {
    Node* parent = handle.Resolve()->GetParent();
    if (parent == nullptr) {
        return py::none();
    }
    return MakeTypedNodeObject(parent);
}

py::list NodeGetPropertyNames(const PyNodeHandle& handle) {
    const Node* node = handle.Resolve();
    py::list names;
    for (const Property& property : node->GetType().get_properties()) {
        names.append(py::str(property.get_name().data()));
    }
    return names;
}

std::string NodeGetName(const PyNodeHandle& handle) {
    return handle.Resolve()->GetName();
}

void NodeSetName(PyNodeHandle& handle, const std::string& name) {
    Node* node = handle.Resolve();
    if (IsSceneScriptRuntimeMutation()) {
        node->SetName(name);
    } else {
        ExecuteSceneMutation("rename_node", [&]() {
            return std::make_unique<RenameNodeCommand>(node->GetInstanceId(), name);
        });
    }
    handle.RefreshSnapshot(node);
}

std::string NodeGetPath(const PyNodeHandle& handle) {
    Node* node = handle.Resolve();
    if (node->IsInsideTree()) {
        return node->GetPath().operator std::string();
    }
    return node->GetName();
}

std::uint64_t NodeGetId(const PyNodeHandle& handle) {
    return handle.state ? static_cast<std::uint64_t>(handle.state->id) : 0;
}

bool NodeIsValid(const PyNodeHandle& handle) {
    try {
        return handle.TryResolve() != nullptr &&
               (!handle.state || handle.state->ownership == PyNodeOwnership::DetachedOwned ||
                handle.state->scene_epoch == ActiveSceneEpoch());
    } catch (...) {
        return false;
    }
}

py::dict NodeToDict(const PyNodeHandle& handle) {
    const Node* node = handle.Resolve();
    py::dict result;
    result["id"] = NodeGetId(handle);
    result["name"] = node->GetName();
    result["path"] = NodeGetPath(handle);
    result["type"] = std::string(node->GetClassStringName());
    py::dict properties;
    for (const Property& property : node->GetType().get_properties()) {
        Variant value = property.get_value(node);
        if (value.is_valid()) {
            properties[py::str(property.get_name().data())] = VariantToPython(value);
        }
    }
    result["properties"] = properties;
    return result;
}

Node* CreateNode(const std::string& type_name, const std::string& name) {
    Node* node = NodeCreationRegistry::CreateNode(type_name);
    if (node == nullptr) {
        throw std::invalid_argument("unknown Gobot node type '" + type_name + "'");
    }
    if (!name.empty()) {
        node->SetName(name);
    }
    return node;
}

py::dict ResourceToPythonDict(const Ref<Resource>& resource) {
    py::dict result;
    if (!resource.IsValid()) {
        return result;
    }

    result["type"] = py::str(resource->GetClassStringName().data());
    result["path"] = resource->GetPath();
    result["name"] = resource->GetName();
    py::dict properties;
    Instance instance(resource.Get());
    const Type type = resource->GetType();
    for (const Property& property : type.get_properties()) {
        Variant value = property.get_value(instance);
        if (!value.is_valid()) {
            continue;
        }
        properties[py::str(property.get_name().data())] = VariantToPython(value);
    }
    result["properties"] = properties;
    for (const auto& item : properties) {
        result[item.first] = item.second;
    }
    return result;
}

py::dict TransformToPythonDict(const Affine3& transform) {
    const Vector3 position = transform.translation();
    const Quaternion rotation(transform.linear());

    py::dict result;
    result["position"] = py::make_tuple(position.x(), position.y(), position.z());
    result["quaternion"] = py::make_tuple(rotation.w(), rotation.x(), rotation.y(), rotation.z());
    return result;
}

Quaternion PythonToQuaternionWxyz(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 4) {
        throw std::invalid_argument("expected a 4-element quaternion in [w, x, y, z] order");
    }

    Quaternion quaternion(py::cast<RealType>(sequence[0]),
                          py::cast<RealType>(sequence[1]),
                          py::cast<RealType>(sequence[2]),
                          py::cast<RealType>(sequence[3]));
    if (quaternion.norm() <= CMP_EPSILON) {
        return Quaternion::Identity();
    }
    quaternion.normalize();
    return quaternion;
}

Vector2 PythonToVector2(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 2) {
        throw std::invalid_argument("expected a 2-element vector");
    }
    return {
            py::cast<RealType>(sequence[0]),
            py::cast<RealType>(sequence[1])
    };
}

py::tuple Vector2ToPython(const Vector2& value) {
    return py::make_tuple(value.x(), value.y());
}

Color PythonToColor4(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 4) {
        throw std::invalid_argument("expected a 4-element color");
    }
    return Color(static_cast<float>(py::cast<RealType>(sequence[0])),
                 static_cast<float>(py::cast<RealType>(sequence[1])),
                 static_cast<float>(py::cast<RealType>(sequence[2])),
                 static_cast<float>(py::cast<RealType>(sequence[3])));
}

py::tuple ColorToPython(const Color& color) {
    return py::make_tuple(color.red(), color.green(), color.blue(), color.alpha());
}

std::vector<Vector3> PythonToVector3List(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    std::vector<Vector3> values;
    values.reserve(static_cast<std::size_t>(sequence.size()));
    for (py::handle item : sequence) {
        values.push_back(PythonToVector3(item));
    }
    return values;
}

py::list Vector3ListToPython(const std::vector<Vector3>& values) {
    py::list result;
    for (const Vector3& value : values) {
        result.append(Vector3ToPython(value));
    }
    return result;
}

std::string PhysicsJointControlModeName(PhysicsJointControlMode mode) {
    switch (mode) {
        case PhysicsJointControlMode::Passive:
            return "passive";
        case PhysicsJointControlMode::Position:
            return "position";
        case PhysicsJointControlMode::Velocity:
            return "velocity";
        case PhysicsJointControlMode::Effort:
            return "effort";
    }

    return "unknown";
}

std::string PhysicsSensorTypeName(PhysicsSensorType type) {
    switch (type) {
        case PhysicsSensorType::IMU:
            return "imu";
        case PhysicsSensorType::AngularMomentum:
            return "angular_momentum";
        case PhysicsSensorType::Contact:
            return "contact";
        case PhysicsSensorType::TerrainHeight:
            return "terrain_height";
        case PhysicsSensorType::Unknown:
            break;
    }

    return "unknown";
}

py::dict JointSnapshotToPythonDict(const PhysicsJointSnapshot& joint) {
    py::dict result;
    result["name"] = joint.name;
    result["parent_link"] = joint.parent_link;
    result["child_link"] = joint.child_link;
    result["type"] = static_cast<JointType>(joint.joint_type);
    result["axis"] = Vector3ToPython(joint.axis);
    result["lower_limit"] = joint.lower_limit;
    result["upper_limit"] = joint.upper_limit;
    result["effort_limit"] = joint.effort_limit;
    result["velocity_limit"] = joint.velocity_limit;
    result["damping"] = joint.damping;
    result["joint_position"] = joint.joint_position;
    result["initial_position"] = joint.initial_position;
    result["drive_mode"] = joint.drive_mode;
    result["drive_stiffness"] = joint.drive_stiffness;
    result["drive_damping"] = joint.drive_damping;
    result["control_lower_limit"] = joint.control_lower_limit;
    result["control_upper_limit"] = joint.control_upper_limit;
    result["force_lower_limit"] = joint.force_lower_limit;
    result["force_upper_limit"] = joint.force_upper_limit;
    result["gear"] = joint.gear;
    result["global_transform"] = TransformToPythonDict(joint.global_transform);
    return result;
}

py::dict LinkSnapshotToPythonDict(const PhysicsLinkSnapshot& link) {
    py::dict result;
    result["name"] = link.name;
    result["role"] = link.role == PhysicsLinkRole::VirtualRoot ? "virtual_root" : "physical";
    result["mass"] = link.mass;
    result["center_of_mass"] = Vector3ToPython(link.center_of_mass);
    result["inertia_diagonal"] = Vector3ToPython(link.inertia_diagonal);
    result["inertia_off_diagonal"] = Vector3ToPython(link.inertia_off_diagonal);
    result["global_transform"] = TransformToPythonDict(link.global_transform);
    result["collision_shape_count"] = link.collision_shapes.size();
    return result;
}

py::dict SensorSnapshotToPythonDict(const PhysicsSensorSnapshot& sensor) {
    py::dict result;
    result["name"] = sensor.name;
    result["sensor_name"] = sensor.name;
    result["link_name"] = sensor.link_name;
    result["type"] = PhysicsSensorTypeName(sensor.type);
    result["enabled"] = sensor.enabled;
    result["sensor_period"] = sensor.sensor_period;
    result["noise_stddev"] = sensor.noise_stddev;
    result["visualize_debug"] = sensor.visualize_debug;
    result["radius"] = sensor.radius;
    result["min_threshold"] = sensor.min_threshold;
    result["max_threshold"] = sensor.max_threshold;
    result["sample_offsets"] = Vector3ListToPython(sensor.sample_offsets);
    result["channel_names"] = sensor.channel_names;
    result["global_transform"] = TransformToPythonDict(sensor.global_transform);
    result["local_transform"] = TransformToPythonDict(sensor.local_transform);
    return result;
}

py::dict RuntimeNameMapToPythonDict(const PhysicsSceneSnapshot& snapshot) {
    py::dict result;
    py::list robots;

    for (const PhysicsRobotSnapshot& robot : snapshot.robots) {
        py::dict robot_dict;
        robot_dict["name"] = robot.name;
        robot_dict["source_path"] = robot.source_path;

        py::list links;
        py::list link_names;
        for (const PhysicsLinkSnapshot& link : robot.links) {
            links.append(LinkSnapshotToPythonDict(link));
            link_names.append(link.name);
        }
        robot_dict["links"] = links;
        robot_dict["link_names"] = link_names;

        py::list joints;
        py::list joint_names;
        py::list controllable_joint_names;
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            joints.append(JointSnapshotToPythonDict(joint));
            joint_names.append(joint.name);

            const auto joint_type = static_cast<JointType>(joint.joint_type);
            if (joint_type == JointType::Revolute ||
                joint_type == JointType::Continuous ||
                joint_type == JointType::Prismatic) {
                controllable_joint_names.append(joint.name);
            }
        }
        robot_dict["joints"] = joints;
        robot_dict["joint_names"] = joint_names;
        robot_dict["controllable_joint_names"] = controllable_joint_names;

        py::list sensors;
        py::list sensor_names;
        for (const PhysicsSensorSnapshot& sensor : robot.sensors) {
            sensors.append(SensorSnapshotToPythonDict(sensor));
            sensor_names.append(sensor.name);
        }
        robot_dict["sensors"] = sensors;
        robot_dict["sensor_names"] = sensor_names;
        robots.append(robot_dict);
    }

    result["robots"] = robots;
    result["total_link_count"] = snapshot.total_link_count;
    result["total_joint_count"] = snapshot.total_joint_count;
    result["total_collision_shape_count"] = snapshot.total_collision_shape_count;
    result["total_sensor_count"] = snapshot.total_sensor_count;
    return result;
}

py::dict JointStateToPythonDict(const PhysicsJointState& joint) {
    py::dict result;
    result["robot_name"] = joint.robot_name;
    result["name"] = joint.joint_name;
    result["joint_name"] = joint.joint_name;
    result["type"] = static_cast<JointType>(joint.joint_type);
    result["position"] = joint.position;
    result["velocity"] = joint.velocity;
    result["effort"] = joint.effort;
    result["control_mode"] = PhysicsJointControlModeName(joint.control_mode);
    result["target_position"] = joint.target_position;
    result["target_velocity"] = joint.target_velocity;
    result["target_effort"] = joint.target_effort;
    return result;
}

py::dict LinkStateToPythonDict(const PhysicsLinkState& link) {
    py::dict result;
    result["robot_name"] = link.robot_name;
    result["name"] = link.link_name;
    result["link_name"] = link.link_name;
    result["role"] = link.role == PhysicsLinkRole::VirtualRoot ? "virtual_root" : "physical";
    result["global_transform"] = TransformToPythonDict(link.global_transform);
    result["linear_velocity"] = Vector3ToPython(link.linear_velocity);
    result["angular_velocity"] = Vector3ToPython(link.angular_velocity);
    return result;
}

py::dict ContactStateToPythonDict(const PhysicsContactState& contact) {
    py::dict result;
    result["robot_name"] = contact.robot_name;
    result["link_name"] = contact.link_name;
    result["other_robot_name"] = contact.other_robot_name;
    result["other_link_name"] = contact.other_link_name;
    result["position"] = Vector3ToPython(contact.position);
    result["normal"] = Vector3ToPython(contact.normal);
    result["distance"] = contact.distance;
    return result;
}

py::dict SensorStateToPythonDict(const PhysicsSensorState& sensor) {
    py::dict result;
    result["robot_name"] = sensor.robot_name;
    result["link_name"] = sensor.link_name;
    result["name"] = sensor.sensor_name;
    result["sensor_name"] = sensor.sensor_name;
    result["type"] = PhysicsSensorTypeName(sensor.type);
    result["enabled"] = sensor.enabled;
    result["global_transform"] = TransformToPythonDict(sensor.global_transform);
    result["values"] = sensor.values;
    result["channel_names"] = sensor.channel_names;
    result["timestamp"] = sensor.timestamp;
    return result;
}

py::dict RuntimeStateToPythonDict(const PhysicsSceneState& state) {
    py::dict result;
    py::list robots;
    for (const PhysicsRobotState& robot : state.robots) {
        py::dict robot_dict;
        robot_dict["name"] = robot.name;

        py::list links;
        for (const PhysicsLinkState& link : robot.links) {
            links.append(LinkStateToPythonDict(link));
        }
        robot_dict["links"] = links;

        py::list joints;
        for (const PhysicsJointState& joint : robot.joints) {
            joints.append(JointStateToPythonDict(joint));
        }
        robot_dict["joints"] = joints;

        py::list sensors;
        for (const PhysicsSensorState& sensor : robot.sensors) {
            sensors.append(SensorStateToPythonDict(sensor));
        }
        robot_dict["sensors"] = sensors;
        robots.append(robot_dict);
    }

    py::list contacts;
    for (const PhysicsContactState& contact : state.contacts) {
        contacts.append(ContactStateToPythonDict(contact));
    }

    result["robots"] = robots;
    result["contacts"] = contacts;
    result["total_link_count"] = state.total_link_count;
    result["total_joint_count"] = state.total_joint_count;
    result["total_sensor_count"] = state.total_sensor_count;
    return result;
}

} // namespace

namespace {

EngineContext& EnsureRuntimeContext() {
    return GetActiveAppContext();
}

Node* PyNodeHandle::Resolve() const {
    if (!state || state->id.IsNull()) {
        ThrowReferenceError("Gobot node handle is invalid");
    }

    if (state->ownership != PyNodeOwnership::DetachedOwned &&
        state->scene_epoch != 0 &&
        state->scene_epoch != ActiveSceneEpoch()) {
        ThrowReferenceError("Gobot node handle '" + state->name_snapshot +
                            "' is from an inactive scene epoch");
    }

    Node* node = TryResolve();
    if (node == nullptr) {
        ThrowReferenceError("Gobot node handle '" + state->name_snapshot +
                            "' id=" + std::to_string(static_cast<std::uint64_t>(state->id)) +
                            " no longer resolves to a live node");
    }
    return node;
}

void PyNodeHandle::TransferToDetachedOwned(Node* node) {
    if (!state) {
        state = MakeState(node, ActiveSceneEpoch(), PyNodeOwnership::DetachedOwned);
        return;
    }
    state->id = node != nullptr ? node->GetInstanceId() : ObjectID();
    state->scene_epoch = ActiveSceneEpoch();
    state->ownership = PyNodeOwnership::DetachedOwned;
    RefreshSnapshot(node);
}

std::string ExpectedTypeForNode(Node* node) {
    return node == nullptr ? "Node" : std::string(node->GetClassStringName());
}

PyNodeHandle MakeNodeHandle(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyNodeHandle(node, ExpectedTypeForNode(node), ActiveSceneEpoch(), ownership);
}

PyNode3DHandle MakeNode3DHandle(Node3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyNode3DHandle(node, ExpectedTypeForNode(node), ActiveSceneEpoch(), ownership);
}

PyRobot3DHandle MakeRobot3DHandle(Robot3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyRobot3DHandle(node, "Robot3D", ActiveSceneEpoch(), ownership);
}

PyLink3DHandle MakeLink3DHandle(Link3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyLink3DHandle(node, "Link3D", ActiveSceneEpoch(), ownership);
}

PyJoint3DHandle MakeJoint3DHandle(Joint3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyJoint3DHandle(node, "Joint3D", ActiveSceneEpoch(), ownership);
}

PyCollisionShape3DHandle MakeCollisionShape3DHandle(CollisionShape3D* node,
                                                    PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyCollisionShape3DHandle(node, "CollisionShape3D", ActiveSceneEpoch(), ownership);
}

PyMeshInstance3DHandle MakeMeshInstance3DHandle(MeshInstance3D* node,
                                                PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyMeshInstance3DHandle(node, "MeshInstance3D", ActiveSceneEpoch(), ownership);
}

PyTerrain3DHandle MakeTerrain3DHandle(Terrain3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyTerrain3DHandle(node, "Terrain3D", ActiveSceneEpoch(), ownership);
}

PySensor3DHandle MakeSensor3DHandle(Sensor3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PySensor3DHandle(node, "Sensor3D", ActiveSceneEpoch(), ownership);
}

PyIMUSensor3DHandle MakeIMUSensor3DHandle(IMUSensor3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyIMUSensor3DHandle(node, "IMUSensor3D", ActiveSceneEpoch(), ownership);
}

PyAngularMomentumSensor3DHandle MakeAngularMomentumSensor3DHandle(AngularMomentumSensor3D* node,
                                                                  PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyAngularMomentumSensor3DHandle(node, "AngularMomentumSensor3D", ActiveSceneEpoch(), ownership);
}

PyContactSensor3DHandle MakeContactSensor3DHandle(ContactSensor3D* node,
                                                  PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyContactSensor3DHandle(node, "ContactSensor3D", ActiveSceneEpoch(), ownership);
}

PyTerrainHeightSensor3DHandle MakeTerrainHeightSensor3DHandle(
        TerrainHeightSensor3D* node,
        PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyTerrainHeightSensor3DHandle(node, "TerrainHeightSensor3D", ActiveSceneEpoch(), ownership);
}

PyNodeHandle MakeTypedNodeHandle(Node* node, PyNodeOwnership ownership) {
    if (auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return MakeMeshInstance3DHandle(mesh_instance, ownership);
    }
    if (auto* terrain = Object::PointerCastTo<Terrain3D>(node)) {
        return MakeTerrain3DHandle(terrain, ownership);
    }
    if (auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(node)) {
        return MakeContactSensor3DHandle(contact_sensor, ownership);
    }
    if (auto* terrain_height_sensor = Object::PointerCastTo<TerrainHeightSensor3D>(node)) {
        return MakeTerrainHeightSensor3DHandle(terrain_height_sensor, ownership);
    }
    if (auto* angular_momentum_sensor = Object::PointerCastTo<AngularMomentumSensor3D>(node)) {
        return MakeAngularMomentumSensor3DHandle(angular_momentum_sensor, ownership);
    }
    if (auto* imu_sensor = Object::PointerCastTo<IMUSensor3D>(node)) {
        return MakeIMUSensor3DHandle(imu_sensor, ownership);
    }
    if (auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        return MakeSensor3DHandle(sensor, ownership);
    }
    if (auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        return MakeCollisionShape3DHandle(collision_shape, ownership);
    }
    if (auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        return MakeJoint3DHandle(joint, ownership);
    }
    if (auto* link = Object::PointerCastTo<Link3D>(node)) {
        return MakeLink3DHandle(link, ownership);
    }
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        return MakeRobot3DHandle(robot, ownership);
    }
    if (auto* node_3d = Object::PointerCastTo<Node3D>(node)) {
        return MakeNode3DHandle(node_3d, ownership);
    }
    return MakeNodeHandle(node, ownership);
}

py::object MakeTypedNodeObject(Node* node, PyNodeOwnership ownership) {
    if (auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return py::cast(MakeMeshInstance3DHandle(mesh_instance, ownership));
    }
    if (auto* terrain = Object::PointerCastTo<Terrain3D>(node)) {
        return py::cast(MakeTerrain3DHandle(terrain, ownership));
    }
    if (auto* contact_sensor = Object::PointerCastTo<ContactSensor3D>(node)) {
        return py::cast(MakeContactSensor3DHandle(contact_sensor, ownership));
    }
    if (auto* terrain_height_sensor = Object::PointerCastTo<TerrainHeightSensor3D>(node)) {
        return py::cast(MakeTerrainHeightSensor3DHandle(terrain_height_sensor, ownership));
    }
    if (auto* angular_momentum_sensor = Object::PointerCastTo<AngularMomentumSensor3D>(node)) {
        return py::cast(MakeAngularMomentumSensor3DHandle(angular_momentum_sensor, ownership));
    }
    if (auto* imu_sensor = Object::PointerCastTo<IMUSensor3D>(node)) {
        return py::cast(MakeIMUSensor3DHandle(imu_sensor, ownership));
    }
    if (auto* sensor = Object::PointerCastTo<Sensor3D>(node)) {
        return py::cast(MakeSensor3DHandle(sensor, ownership));
    }
    if (auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node)) {
        return py::cast(MakeCollisionShape3DHandle(collision_shape, ownership));
    }
    if (auto* joint = Object::PointerCastTo<Joint3D>(node)) {
        return py::cast(MakeJoint3DHandle(joint, ownership));
    }
    if (auto* link = Object::PointerCastTo<Link3D>(node)) {
        return py::cast(MakeLink3DHandle(link, ownership));
    }
    if (auto* robot = Object::PointerCastTo<Robot3D>(node)) {
        return py::cast(MakeRobot3DHandle(robot, ownership));
    }
    if (auto* node_3d = Object::PointerCastTo<Node3D>(node)) {
        return py::cast(MakeNode3DHandle(node_3d, ownership));
    }
    return py::cast(MakeNodeHandle(node, ownership));
}

void ExecuteSetNodeProperty(Node* node, const std::string& property_name, Variant value) {
    if (node == nullptr) {
        throw std::invalid_argument("cannot set property on a null Gobot node");
    }
    if (IsSceneScriptRuntimeMutation()) {
        if (!node->Set(property_name, std::move(value))) {
            throw std::runtime_error("failed to set Gobot runtime node property '" + property_name + "'");
        }
        return;
    }
    ExecuteSceneMutation("set_property", [&]() {
        return std::make_unique<SetNodePropertyCommand>(node->GetInstanceId(), property_name, std::move(value));
    });
}

} // namespace

void RegisterRuntime(py::module_& module) {
    module.doc() = "Gobot robotics scene, simulation, and rendering engine bindings.";
}

void RegisterManualApis(py::module_& module) {
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
            .value("MjLab", TerrainColorMode::MjLab)
            .export_values();

    auto node_class = py::class_<PyNodeHandle>(module, "Node");
    auto node3d_class = py::class_<PyNode3DHandle, PyNodeHandle>(module, "Node3D");
    auto robot3d_class = py::class_<PyRobot3DHandle, PyNode3DHandle>(module, "Robot3D");
    auto link3d_class = py::class_<PyLink3DHandle, PyNode3DHandle>(module, "Link3D");
    auto joint3d_class = py::class_<PyJoint3DHandle, PyNode3DHandle>(module, "Joint3D");
    auto collision_shape_class =
            py::class_<PyCollisionShape3DHandle, PyNode3DHandle>(module, "CollisionShape3D");
    auto mesh_instance_class =
            py::class_<PyMeshInstance3DHandle, PyNode3DHandle>(module, "MeshInstance3D");
    auto terrain3d_class = py::class_<PyTerrain3DHandle, PyNode3DHandle>(module, "Terrain3D");
    auto sensor3d_class = py::class_<PySensor3DHandle, PyNode3DHandle>(module, "Sensor3D");
    auto imu_sensor3d_class = py::class_<PyIMUSensor3DHandle, PySensor3DHandle>(module, "IMUSensor3D");
    auto angular_momentum_sensor3d_class =
            py::class_<PyAngularMomentumSensor3DHandle, PySensor3DHandle>(module, "AngularMomentumSensor3D");
    auto contact_sensor3d_class =
            py::class_<PyContactSensor3DHandle, PySensor3DHandle>(module, "ContactSensor3D");
    auto terrain_height_sensor3d_class =
            py::class_<PyTerrainHeightSensor3DHandle, PySensor3DHandle>(module, "TerrainHeightSensor3D");

    py::implicitly_convertible<PyRobot3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyLink3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyJoint3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyCollisionShape3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyMeshInstance3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PySensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyIMUSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyAngularMomentumSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyContactSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyTerrainHeightSensor3DHandle, PyNodeHandle>();

    py::class_<EngineContext>(module, "AppContext")
            .def_property_readonly("project_path", &EngineContext::GetProjectPath)
            .def_property_readonly("scene_path", &EngineContext::GetScenePath)
            .def_property_readonly("scene_epoch", &EngineContext::GetSceneEpoch)
            .def_property_readonly("scene_dirty", &EngineContext::IsSceneDirty)
            .def_property_readonly("can_undo", &EngineContext::CanUndoSceneCommand)
            .def_property_readonly("can_redo", &EngineContext::CanRedoSceneCommand)
            .def_property_readonly("undo_name", &EngineContext::GetUndoSceneCommandName)
            .def_property_readonly("redo_name", &EngineContext::GetRedoSceneCommandName)
            .def_property_readonly("root", [](EngineContext& context) -> py::object {
                Node* root = ActiveSceneRoot();
                if (root == nullptr) {
                    return py::none();
                }
                return MakeTypedNodeObject(root);
            })
            .def_property_readonly("input", [](EngineContext&) -> Input* {
                return Input::GetInstanceOrNull();
            }, py::return_value_policy::reference)
            .def_property("backend_type",
                          &EngineContext::GetBackendType,
                          &EngineContext::SetBackendType)
            .def_property_readonly("has_scene", &EngineContext::HasScene)
            .def_property_readonly("has_world", &EngineContext::HasWorld)
            .def_property_readonly("simulation_time", &EngineContext::GetSimulationTime)
            .def_property_readonly("frame_count", &EngineContext::GetFrameCount)
            .def_property("fixed_time_step",
                          &EngineContext::GetFixedTimeStep,
                          [](EngineContext& context, RealType fixed_time_step) {
                              if (!context.SetFixedTimeStep(fixed_time_step)) {
                                  throw std::runtime_error(context.GetLastError());
                              }
                          })
            .def_property_readonly("gravity", [](const EngineContext& context) {
                return Vector3ToPython(context.GetGravity());
            })
            .def("set_project_path", [](EngineContext& context, const std::string& project_path) {
                if (!context.SetProjectPath(project_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("project_path"))
            .def("load_scene", [](EngineContext& context, const std::string& scene_path) {
                if (!context.LoadScene(scene_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
                return MakeTypedNodeObject(context.GetSceneRoot());
            }, py::arg("scene_path"))
            .def("clear_scene", &EngineContext::ClearScene)
            .def("notify_scene_changed", &EngineContext::NotifySceneChanged)
            .def("mark_scene_clean", &EngineContext::MarkSceneClean)
            .def("undo", [](EngineContext& context) {
                return context.UndoSceneCommand();
            })
            .def("redo", [](EngineContext& context) {
                return context.RedoSceneCommand();
            })
            .def("begin_transaction", [](EngineContext& context, const std::string& name) {
                if (!context.BeginSceneTransaction(name)) {
                    throw std::runtime_error("failed to begin Gobot scene transaction '" + name + "'");
                }
            }, py::arg("name") = "Scene Transaction")
            .def("commit_transaction", [](EngineContext& context) {
                if (!context.CommitSceneTransaction()) {
                    throw std::runtime_error("failed to commit Gobot scene transaction");
                }
            })
            .def("cancel_transaction", [](EngineContext& context) {
                if (!context.CancelSceneTransaction()) {
                    throw std::runtime_error("failed to cancel Gobot scene transaction");
                }
            })
            .def("transaction", [](EngineContext&, const std::string& name) {
                return PySceneTransaction(name);
            }, py::arg("name") = "Scene Transaction")
            .def("build_world", [](EngineContext& context, PhysicsBackendType backend_type) {
                context.SetBackendType(backend_type);
                if (!context.BuildWorld()) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("backend_type") = PhysicsBackendType::Null)
            .def("rebuild_world", [](EngineContext& context, bool preserve_state) {
                if (!context.RebuildWorld(preserve_state)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("preserve_state") = true)
            .def("clear_world", &EngineContext::ClearWorld)
            .def("reset_simulation", [](EngineContext& context) {
                if (!context.ResetSimulation()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step_once", [](EngineContext& context) {
                if (!context.StepOnce()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step", [](EngineContext& context, std::uint64_t ticks) {
                if (!context.StepTicks(ticks)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("ticks") = 1)
            .def("configure_batch_world", [](EngineContext& context, std::size_t num_envs) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ConfigureEnvironmentBatch(num_envs)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("num_envs"))
            .def_property_readonly("batch_env_count", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    return static_cast<std::size_t>(0);
                }
                return simulation->GetEnvironmentCount();
            })
            .def("reset_batch_env", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ResetEnvironment(env_id)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"))
            .def("step_batch_env", [](EngineContext& context, std::size_t env_id, std::uint64_t ticks) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->StepEnvironment(env_id, ticks)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"), py::arg("ticks") = 1)
            .def("set_batch_joint_position_target", [](EngineContext& context,
                                                       std::size_t env_id,
                                                       const std::string& robot,
                                                       const std::string& joint,
                                                       RealType target_position) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetEnvironmentJointPositionTarget(env_id, robot, joint, target_position)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("target_position"))
            .def("reset_batch_joint_state", [](EngineContext& context,
                                               std::size_t env_id,
                                               const std::string& robot,
                                               const std::string& joint,
                                               RealType position,
                                               RealType velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentJointState(env_id, robot, joint, position, velocity)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("position"), py::arg("velocity") = 0.0)
            .def("reset_batch_link_state", [](EngineContext& context,
                                              std::size_t env_id,
                                              const std::string& robot,
                                              const std::string& link,
                                              const py::object& position,
                                              const py::object& orientation,
                                              const py::object& linear_velocity,
                                              const py::object& angular_velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentLinkState(env_id,
                                                              robot,
                                                              link,
                                                              PythonToVector3(position),
                                                              PythonToQuaternionWxyz(orientation),
                                                              PythonToVector3(linear_velocity),
                                                              PythonToVector3(angular_velocity))) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"),
               py::arg("robot"),
               py::arg("link"),
               py::arg("position"),
               py::arg("orientation") = py::make_tuple(1.0, 0.0, 0.0, 0.0),
               py::arg("linear_velocity") = py::make_tuple(0.0, 0.0, 0.0),
               py::arg("angular_velocity") = py::make_tuple(0.0, 0.0, 0.0))
            .def("set_robot_action", [](EngineContext& context,
                                        const std::string& robot,
                                        const std::vector<RealType>& action) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetRobotJointPositionTargetsFromNormalizedAction(robot, action)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("action"))
            .def("set_robot_named_action", [](EngineContext& context,
                                              const std::string& robot,
                                              const std::vector<std::string>& joint_names,
                                              const std::vector<RealType>& action) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetRobotJointPositionTargetsFromNormalizedAction(robot, joint_names, action)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint_names"), py::arg("action"))
            .def("set_default_joint_gains", [](EngineContext& context, py::dict gains) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                simulation->SetDefaultJointGains(DictToReflected<JointControllerGains>(gains));
            }, py::arg("gains"))
            .def("get_default_joint_gains", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return ReflectedToPythonDict(simulation->GetDefaultJointGains());
            })
            .def("set_joint_position_target", [](EngineContext& context,
                                                 const std::string& robot,
                                                 const std::string& joint,
                                                 RealType target_position) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetJointPositionTarget(robot, joint, target_position)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("target_position"))
            .def("set_joint_velocity_target", [](EngineContext& context,
                                                 const std::string& robot,
                                                 const std::string& joint,
                                                 RealType target_velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetJointVelocityTarget(robot, joint, target_velocity)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("target_velocity"))
            .def("set_joint_effort_target", [](EngineContext& context,
                                               const std::string& robot,
                                               const std::string& joint,
                                               RealType target_effort) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetJointEffortTarget(robot, joint, target_effort)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("target_effort"))
            .def("set_joint_passive", [](EngineContext& context,
                                         const std::string& robot,
                                         const std::string& joint) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetJointPassive(robot, joint)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"))
            .def("reset_joint_state", [](EngineContext& context,
                                         const std::string& robot,
                                         const std::string& joint,
                                         RealType position,
                                         RealType velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetJointState(robot, joint, position, velocity)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("position"), py::arg("velocity") = 0.0)
            .def("reset_link_state", [](EngineContext& context,
                                        const std::string& robot,
                                        const std::string& link,
                                        const py::object& position,
                                        const py::object& orientation,
                                        const py::object& linear_velocity,
                                        const py::object& angular_velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetLinkState(robot,
                                                   link,
                                                   PythonToVector3(position),
                                                   PythonToQuaternionWxyz(orientation),
                                                   PythonToVector3(linear_velocity),
                                                   PythonToVector3(angular_velocity))) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"),
               py::arg("link"),
               py::arg("position"),
               py::arg("orientation") = py::make_tuple(1.0, 0.0, 0.0, 0.0),
               py::arg("linear_velocity") = py::make_tuple(0.0, 0.0, 0.0),
               py::arg("angular_velocity") = py::make_tuple(0.0, 0.0, 0.0))
            .def("get_runtime_name_map", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                Ref<PhysicsWorld> world = simulation->GetWorld();
                if (!world.IsValid()) {
                    throw std::runtime_error("simulation world has not been built from a scene");
                }
                return RuntimeNameMapToPythonDict(world->GetSceneSnapshot());
            })
            .def("get_runtime_state", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                Ref<PhysicsWorld> world = simulation->GetWorld();
                if (!world.IsValid()) {
                    throw std::runtime_error("simulation world has not been built from a scene");
                }
                return RuntimeStateToPythonDict(world->GetSceneState());
            })
            .def("get_batch_runtime_state", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                const PhysicsSceneState* state = simulation->GetEnvironmentState(env_id);
                if (state == nullptr) {
                    throw std::runtime_error("simulation environment state is not available");
                }
                return RuntimeStateToPythonDict(*state);
            }, py::arg("env_id"))
            .def("get_last_error", &EngineContext::GetLastError);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return MakeTypedNodeObject(scene.root);
            })
            .def_property_readonly("scene_epoch", [](const PyScene& scene) {
                return scene.scene_epoch;
            });

    py::class_<PySceneTransaction>(module, "SceneTransaction")
            .def("__enter__", &PySceneTransaction::Enter, py::return_value_policy::reference_internal)
            .def("__exit__", &PySceneTransaction::Exit,
                 py::arg("exc_type"),
                 py::arg("exc_value"),
                 py::arg("traceback"));

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
                          });

    robot3d_class
            .def_property("source_path",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetSourcePath();
                          },
                          [](PyRobot3DHandle& handle, const std::string& source_path) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSetNodeProperty(robot, "source_path", Variant(source_path));
                          })
            .def_property("mode",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetMode();
                          },
                          [](PyRobot3DHandle& handle, RobotMode mode) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSetNodeProperty(robot, "mode", Variant(mode));
                          });

    link3d_class
            .def_property("has_inertial",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->HasInertial();
                          },
                          [](PyLink3DHandle& handle, bool has_inertial) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "has_inertial", Variant(has_inertial));
                          })
            .def_property("mass",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetMass();
                          },
                          [](PyLink3DHandle& handle, RealType mass) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "mass", Variant(mass));
                          })
            .def_property("center_of_mass",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetCenterOfMass());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "center_of_mass", Variant(PythonToVector3(value)));
                          })
            .def_property("inertia_diagonal",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetInertiaDiagonal());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "inertia_diagonal", Variant(PythonToVector3(value)));
                          })
            .def_property("role",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetRole();
                          },
                          [](PyLink3DHandle& handle, LinkRole role) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "role", Variant(role));
                          });

    joint3d_class
            .def_property("joint_type",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointType();
                          },
                          [](PyJoint3DHandle& handle, JointType joint_type) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "joint_type", Variant(joint_type));
                          })
            .def_property("parent_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetParentLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& parent_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "parent_link", Variant(parent_link));
                          })
            .def_property("child_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetChildLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& child_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "child_link", Variant(child_link));
                          })
            .def_property("axis",
                          [](const PyJoint3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Joint3D>()->GetAxis());
                          },
                          [](PyJoint3DHandle& handle, const py::handle& value) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "axis", Variant(PythonToVector3(value)));
                          })
            .def_property("lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "lower_limit", Variant(lower_limit));
                          })
            .def_property("upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "upper_limit", Variant(upper_limit));
                          })
            .def_property("effort_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetEffortLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType effort_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "effort_limit", Variant(effort_limit));
                          })
            .def_property("velocity_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetVelocityLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType velocity_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "velocity_limit", Variant(velocity_limit));
                          })
            .def_property("damping",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDamping();
                          },
                          [](PyJoint3DHandle& handle, RealType damping) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "damping", Variant(damping));
                          })
            .def_property("joint_position",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointPosition();
                          },
                          [](PyJoint3DHandle& handle, RealType joint_position) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "joint_position", Variant(joint_position));
                          })
            .def_property("initial_position",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetInitialPosition();
                          },
                          [](PyJoint3DHandle& handle, RealType initial_position) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "initial_position", Variant(initial_position));
                          })
            .def_property("drive_mode",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveMode();
                          },
                          [](PyJoint3DHandle& handle, JointDriveMode drive_mode) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_mode", Variant(drive_mode));
                          })
            .def_property("drive_stiffness",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveStiffness();
                          },
                          [](PyJoint3DHandle& handle, RealType stiffness) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_stiffness", Variant(stiffness));
                          })
            .def_property("drive_damping",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveDamping();
                          },
                          [](PyJoint3DHandle& handle, RealType damping) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_damping", Variant(damping));
                          })
            .def_property("control_lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetControlLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "control_lower_limit", Variant(lower_limit));
                          })
            .def_property("control_upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetControlUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "control_upper_limit", Variant(upper_limit));
                          })
            .def_property("force_lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetForceLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "force_lower_limit", Variant(lower_limit));
                          })
            .def_property("force_upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetForceUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "force_upper_limit", Variant(upper_limit));
                          })
            .def_property("gear",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetGear();
                          },
                          [](PyJoint3DHandle& handle, const std::vector<RealType>& gear) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "gear", Variant(gear));
                          });

    collision_shape_class
            .def_property("disabled",
                          [](const PyCollisionShape3DHandle& handle) {
                              return handle.ResolveAs<CollisionShape3D>()->IsDisabled();
                          },
                          [](PyCollisionShape3DHandle& handle, bool disabled) {
                              CollisionShape3D* collision_shape = handle.ResolveAs<CollisionShape3D>();
                              ExecuteSetNodeProperty(collision_shape, "disabled", Variant(disabled));
                          });

    terrain3d_class
            .def("clear_terrain",
                 [](PyTerrain3DHandle& handle) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     ExecuteSetNodeProperty(terrain, "boxes", Variant(std::vector<TerrainBox>{}));
                     ExecuteSetNodeProperty(terrain, "heightfields", Variant(std::vector<TerrainHeightField>{}));
                     ExecuteSetNodeProperty(terrain, "mesh_patches", Variant(std::vector<TerrainMeshPatch>{}));
                     ExecuteSetNodeProperty(terrain, "spawn_origins", Variant(std::vector<Vector3>{}));
                 })
            .def("add_box",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& size,
                    const py::handle& rotation_degrees,
                    const py::handle& color) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainBox> boxes = terrain->GetBoxes();
                     TerrainBox box;
                     box.center = PythonToVector3(center);
                     box.size = PythonToVector3(size);
                     box.rotation_degrees = PythonToVector3(rotation_degrees);
                     box.color = PythonToColor4(color);
                     boxes.push_back(box);
                     ExecuteSetNodeProperty(terrain, "boxes", Variant(boxes));
                 },
                 py::arg("center"),
                 py::arg("size"),
                 py::arg("rotation_degrees") = py::make_tuple(0.0, 0.0, 0.0),
                 py::arg("color") = py::make_tuple(1.0, 1.0, 1.0, 1.0))
            .def("add_heightfield",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& size,
                    int rows,
                    int cols,
                    const std::vector<RealType>& heights,
                    RealType base_thickness,
                    const std::vector<RealType>& normalized_elevation,
                    RealType z_offset) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainHeightField> heightfields = terrain->GetHeightFields();
                     TerrainHeightField heightfield;
                     heightfield.center = PythonToVector3(center);
                     heightfield.size = PythonToVector2(size);
                     heightfield.rows = rows;
                     heightfield.cols = cols;
                     heightfield.heights = heights;
                     heightfield.normalized_elevation = normalized_elevation;
                     heightfield.base_thickness = base_thickness;
                     heightfield.z_offset = z_offset;
                     heightfields.push_back(std::move(heightfield));
                     ExecuteSetNodeProperty(terrain, "heightfields", Variant(heightfields));
                 },
                 py::arg("center"),
                 py::arg("size"),
                 py::arg("rows"),
                 py::arg("cols"),
                 py::arg("heights"),
                 py::arg("base_thickness") = 0.1,
                 py::arg("normalized_elevation") = std::vector<RealType>{},
                 py::arg("z_offset") = 0.0)
            .def("add_mesh_patch",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& vertices,
                    const std::vector<std::uint32_t>& indices,
                    const py::handle& rotation_degrees,
                    const py::handle& color) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainMeshPatch> mesh_patches = terrain->GetMeshPatches();
                     TerrainMeshPatch mesh_patch;
                     mesh_patch.center = PythonToVector3(center);
                     mesh_patch.vertices = PythonToVector3List(vertices);
                     mesh_patch.indices = indices;
                     mesh_patch.rotation_degrees = PythonToVector3(rotation_degrees);
                     mesh_patch.color = PythonToColor4(color);
                     mesh_patches.push_back(std::move(mesh_patch));
                     ExecuteSetNodeProperty(terrain, "mesh_patches", Variant(mesh_patches));
                 },
                 py::arg("center"),
                 py::arg("vertices"),
                 py::arg("indices"),
                 py::arg("rotation_degrees") = py::make_tuple(0.0, 0.0, 0.0),
                 py::arg("color") = py::make_tuple(1.0, 1.0, 1.0, 1.0))
            .def_property_readonly("box_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetBoxes().size();
                                   })
            .def_property_readonly("heightfield_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetHeightFields().size();
                                   })
            .def_property_readonly("mesh_patch_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetMeshPatches().size();
                                   })
            .def("get_heightfield_heights",
                 [](const PyTerrain3DHandle& handle, std::size_t index) {
                     const auto& heightfields = handle.ResolveAs<Terrain3D>()->GetHeightFields();
                     if (index >= heightfields.size()) {
                         throw py::index_error("Terrain3D heightfield index out of range");
                     }
                     return heightfields[index].heights;
                 },
                 py::arg("index"))
            .def_property("spawn_origins",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector3ListToPython(handle.ResolveAs<Terrain3D>()->GetSpawnOrigins());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "spawn_origins", Variant(PythonToVector3List(value)));
                          })
            .def_property("surface_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetSurfaceColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "surface_color", Variant(PythonToColor4(value)));
                          })
            .def_property("color_mode",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetColorMode();
                          },
                          [](PyTerrain3DHandle& handle, TerrainColorMode value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "color_mode", Variant(value));
                          })
            .def_property("height_low_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetHeightLowColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_low_color", Variant(PythonToColor4(value)));
                          })
            .def_property("height_high_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetHeightHighColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_high_color", Variant(PythonToColor4(value)));
                          })
            .def_property("height_range_min",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetHeightRangeMin();
                          },
                          [](PyTerrain3DHandle& handle, RealType value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_range_min", Variant(value));
                          })
            .def_property("height_range_max",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetHeightRangeMax();
                          },
                          [](PyTerrain3DHandle& handle, RealType value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_range_max", Variant(value));
                          })
            .def_property("friction",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Terrain3D>()->GetFriction());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "friction", Variant(PythonToVector3(value)));
                          })
            .def_property("solref",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector2ToPython(handle.ResolveAs<Terrain3D>()->GetSolref());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "solref", Variant(PythonToVector2(value)));
                          })
            .def_property("solimp",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetSolimp();
                          },
                          [](PyTerrain3DHandle& handle, const std::vector<RealType>& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "solimp", Variant(value));
                          });

    sensor3d_class
            .def_property("enabled",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->IsEnabled();
                          },
                          [](PySensor3DHandle& handle, bool enabled) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "enabled", Variant(enabled));
                          })
            .def_property("sensor_period",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->GetSensorPeriod();
                          },
                          [](PySensor3DHandle& handle, RealType sensor_period) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "sensor_period", Variant(sensor_period));
                          })
            .def_property("noise_stddev",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->GetNoiseStddev();
                          },
                          [](PySensor3DHandle& handle, RealType noise_stddev) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "noise_stddev", Variant(noise_stddev));
                          })
            .def_property("visualize_debug",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->ShouldVisualizeDebug();
                          },
                          [](PySensor3DHandle& handle, bool visualize_debug) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "visualize_debug", Variant(visualize_debug));
                          });

    contact_sensor3d_class
            .def_property("radius",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetRadius();
                          },
                          [](PyContactSensor3DHandle& handle, RealType radius) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "radius", Variant(radius));
                          })
            .def_property("min_threshold",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetMinThreshold();
                          },
                          [](PyContactSensor3DHandle& handle, RealType min_threshold) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "min_threshold", Variant(min_threshold));
                          })
            .def_property("max_threshold",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetMaxThreshold();
                          },
                          [](PyContactSensor3DHandle& handle, RealType max_threshold) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "max_threshold", Variant(max_threshold));
                          });

    terrain_height_sensor3d_class
            .def_property("sample_offsets",
                          [](const PyTerrainHeightSensor3DHandle& handle) {
                              return Vector3ListToPython(
                                      handle.ResolveAs<TerrainHeightSensor3D>()->GetSampleOffsets());
                          },
                          [](PyTerrainHeightSensor3DHandle& handle, const py::handle& value) {
                              TerrainHeightSensor3D* sensor = handle.ResolveAs<TerrainHeightSensor3D>();
                              ExecuteSetNodeProperty(sensor, "sample_offsets", Variant(PythonToVector3List(value)));
                          });

    GOB_UNUSED(imu_sensor3d_class);
    GOB_UNUSED(angular_momentum_sensor3d_class);

    mesh_instance_class
            .def_property("surface_color",
                          [](const PyMeshInstance3DHandle& handle) {
                              const Color color = handle.ResolveAs<MeshInstance3D>()->GetSurfaceColor();
                              return py::make_tuple(color.red(), color.green(), color.blue(), color.alpha());
                          },
                          [](PyMeshInstance3DHandle& handle, const py::handle& value) {
                              py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
                              if (sequence.size() != 4) {
                                  throw std::invalid_argument("expected a 4-element RGBA color");
                              }
                              MeshInstance3D* mesh_instance = handle.ResolveAs<MeshInstance3D>();
                              ExecuteSetNodeProperty(mesh_instance, "surface_color", Variant(Color{
                                      py::cast<float>(sequence[0]),
                                      py::cast<float>(sequence[1]),
                                      py::cast<float>(sequence[2]),
                                      py::cast<float>(sequence[3])
                              }));
                          });

    module.def("set_project_path", [](const std::string& project_path) {
        EngineContext& context = GetActiveAppContext();
        if (!context.SetProjectPath(project_path)) {
            throw std::runtime_error("failed to set Gobot project path '" + project_path + "'");
        }
    });

    py::module_ app_module = module.def_submodule("app", "Gobot application runtime context.");
    app_module.def("context", []() -> EngineContext& {
        return GetActiveAppContext();
    }, py::return_value_policy::reference);

    module.def("load_scene", [](const std::string& scene_path) {
        EnsureRuntimeContext();
        return std::make_unique<PyScene>(LoadSceneRoot(scene_path));
    }, py::arg("scene_path"));

    module.def("create_node", [](const std::string& type_name, const std::string& name) {
        EnsureRuntimeContext();
        return MakeTypedNodeObject(CreateNode(type_name, name), PyNodeOwnership::DetachedOwned);
    }, py::arg("type_name"), py::arg("name") = "");

    module.def("transaction", [](const std::string& name) {
        EnsureRuntimeContext();
        return PySceneTransaction(name);
    }, py::arg("name") = "Scene Transaction");

    module.def("undo", []() {
        EnsureRuntimeContext();
        return GetActiveAppContext().UndoSceneCommand();
    });

    module.def("redo", []() {
        EnsureRuntimeContext();
        return GetActiveAppContext().RedoSceneCommand();
    });

    module.def("create_box_collision", [](const std::string& name,
                                          const py::handle& size,
                                          const py::handle& position) {
        EnsureRuntimeContext();
        return MakeCollisionShape3DHandle(CreateBoxCollision(name, PythonToVector3(size), PythonToVector3(position)),
                                          PyNodeOwnership::DetachedOwned);
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::move);

    module.def("create_box_visual", [](const std::string& name,
                                       const py::handle& size,
                                       const py::handle& position) {
        EnsureRuntimeContext();
        return MakeMeshInstance3DHandle(CreateBoxVisual(name, PythonToVector3(size), PythonToVector3(position)),
                                        PyNodeOwnership::DetachedOwned);
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::move);

    module.def("save_scene", [](PyNodeHandle& root, const std::string& path) {
        EnsureRuntimeContext();
        if (!SaveSceneRoot(root.Resolve(), path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + path + "'");
        }
    }, py::arg("root"), py::arg("path"));

    module.def("import_mjcf_scene", [](const std::string& xml_path,
                                       const std::string& scene_path,
                                       const std::optional<std::string>& name,
                                       const std::optional<std::string>& script) {
        EnsureRuntimeContext();
        ImportMJCFScene(xml_path, scene_path, name, script);
    }, py::arg("xml_path"), py::arg("scene_path"), py::arg("name") = py::none(), py::arg("script") = py::none());

    module.def("load_resource", [](const std::string& path, const std::string& type_hint) {
        EnsureRuntimeContext();
        return ResourceToPythonDict(ResourceLoader::Load(path, type_hint));
    }, py::arg("path"), py::arg("type_hint") = "");

    module.def("_node_from_id", [](std::uint64_t id) -> py::object {
        EnsureRuntimeContext();
        auto* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(ObjectID(id)));
        if (node == nullptr) {
            return py::none();
        }
        return MakeTypedNodeObject(node);
    }, py::arg("id"));

    module.def("create_test_scene", []() {
        EnsureRuntimeContext();
        return std::make_unique<PyScene>(CreateTestRobotScene());
    });

    module.def("backend_infos", []() {
        EnsureRuntimeContext();
        py::list infos;
        for (const PhysicsBackendInfo& info : PhysicsServer::GetBackendInfosForAllBackends()) {
            infos.append(ReflectedToPythonDict(info));
        }
        return infos;
    });

}

void RegisterModule(py::module_& module) {
    RegisterRuntime(module);
    RegisterReflectedTypes(module);
    RegisterManualApis(module);
}

} // namespace gobot::python
