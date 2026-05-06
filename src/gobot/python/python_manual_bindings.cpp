#include "gobot/python/python_binding_registry.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/node_creation_registry.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {
namespace {

EngineContext* s_active_app_context = nullptr;

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

std::string ExpectedTypeForNode(Node* node);
PyNodeHandle MakeTypedNodeHandle(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed);
py::object MakeTypedNodeObject(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed);

template <typename Func>
void ExecuteSceneMutation(const std::string&, Func&& func);

[[noreturn]] void ThrowReferenceError(const std::string& message) {
    PyErr_SetString(PyExc_ReferenceError, message.c_str());
    throw py::error_already_set();
}

struct GobotRuntime {
    ProjectSettings* project_settings{nullptr};
    PhysicsServer* physics_server{nullptr};
    SimulationServer* app_simulation_server{nullptr};
    std::unique_ptr<EngineContext> app_context;
    bool scene_initializer_ready{false};

    GobotRuntime() {
        project_settings = Object::New<ProjectSettings>();
        physics_server = Object::New<PhysicsServer>();
        app_simulation_server = Object::New<SimulationServer>();
        app_context = std::make_unique<EngineContext>(project_settings,
                                                      physics_server,
                                                      app_simulation_server);
        SceneInitializer::Init();
        scene_initializer_ready = true;
    }

    ~GobotRuntime() {
        app_context.reset();
        if (app_simulation_server != nullptr) {
            Object::Delete(app_simulation_server);
            app_simulation_server = nullptr;
        }
        if (scene_initializer_ready) {
            SceneInitializer::Destroy();
            scene_initializer_ready = false;
        }
        if (physics_server != nullptr) {
            Object::Delete(physics_server);
            physics_server = nullptr;
        }
        if (project_settings != nullptr) {
            Object::Delete(project_settings);
            project_settings = nullptr;
        }
    }

    EngineContext& GetAppContext() {
        return *app_context;
    }
};

GobotRuntime& Runtime() {
    static GobotRuntime runtime;
    return runtime;
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

PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return PhysicsBackendType::Null;
    }
    if (backend == "mujoco") {
        return PhysicsBackendType::MuJoCoCpu;
    }
    throw std::invalid_argument("unknown Gobot physics backend '" + backend + "'");
}

struct PyRLEnvironment {
    SimulationServer* simulation{nullptr};
    RLEnvironment* environment{nullptr};
    Node* scene_root{nullptr};

    PyRLEnvironment(const std::string& scene_path,
                    const std::string& robot,
                    const std::string& backend) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        simulation = Object::New<SimulationServer>(ParseBackend(backend));
        environment = Object::New<RLEnvironment>(simulation);
        scene_root = scene_path.empty() ? static_cast<Node*>(CreateTestRobotScene()) : LoadSceneRoot(scene_path);
        environment->SetSceneRoot(scene_root);
        environment->SetRobotName(robot);
    }

    ~PyRLEnvironment() {
        if (environment != nullptr) {
            Object::Delete(environment);
            environment = nullptr;
        }
        if (simulation != nullptr) {
            Object::Delete(simulation);
            simulation = nullptr;
        }
        if (scene_root != nullptr) {
            Object::Delete(scene_root);
            scene_root = nullptr;
        }
    }

    RLEnvironmentResetResult Reset(std::uint32_t seed) {
        return environment->Reset(seed);
    }

    RLEnvironmentStepResult Step(const std::vector<RealType>& action) {
        return environment->Step(action);
    }

    void SetDefaultJointGains(const JointControllerGains& gains) {
        simulation->SetDefaultJointGains(gains);
    }

    const JointControllerGains& GetDefaultJointGains() const {
        return simulation->GetDefaultJointGains();
    }
};

struct PyRLControllerConfig {
    std::vector<std::string> controlled_joints;
    std::vector<RealType> default_action;
    JointControllerGains joint_gains{100.0, 10.0, 0.0, 0.0};
};

py::dict ControllerConfigToDict(const PyRLControllerConfig& config) {
    py::dict result;
    result["controlled_joints"] = config.controlled_joints;
    result["default_action"] = config.default_action;
    result["joint_gains"] = ReflectedToPythonDict(config.joint_gains);
    return result;
}

PyRLControllerConfig ControllerConfigFromDict(py::dict dict) {
    PyRLControllerConfig config;
    if (dict.contains("controlled_joints")) {
        config.controlled_joints = py::cast<std::vector<std::string>>(dict["controlled_joints"]);
    }
    if (dict.contains("default_action")) {
        config.default_action = py::cast<std::vector<RealType>>(dict["default_action"]);
    }
    if (dict.contains("joint_gains")) {
        config.joint_gains = DictToReflected<JointControllerGains>(
                py::reinterpret_borrow<py::dict>(dict["joint_gains"]));
    }
    return config;
}

PyRLControllerConfig MakeControllerConfigFromEnv(PyRLEnvironment& env) {
    PyRLControllerConfig config;
    config.controlled_joints = env.environment->GetControlledJointNames();
    config.default_action = env.environment->GetDefaultAction();
    if (config.default_action.empty()) {
        config.default_action.assign(config.controlled_joints.size(), 0.0);
    }
    config.joint_gains = env.GetDefaultJointGains();
    return config;
}

void ApplyControllerConfigToEnv(PyRLEnvironment& env, py::dict dict) {
    const PyRLControllerConfig config = ControllerConfigFromDict(dict);
    env.environment->SetConfiguredControlledJointNames(config.controlled_joints);
    env.environment->SetDefaultAction(config.default_action);
    env.SetDefaultJointGains(config.joint_gains);
    if (!config.default_action.empty() &&
        config.default_action.size() != config.controlled_joints.size()) {
        throw std::invalid_argument("default_action size must match controlled_joints size");
    }
}

struct PyScene {
    Node* root{nullptr};
    std::uint64_t scene_epoch{0};

    explicit PyScene(Node* root_node)
        : root(root_node),
          scene_epoch(GetActiveAppContext().GetSceneEpoch()) {
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
    ExecuteSceneMutation("set_property", [&]() {
        if (!node->Set(name, PythonToVariantForType(value, property.get_type()))) {
            throw std::runtime_error("failed to set Gobot property '" + name + "'");
        }
        handle.RefreshSnapshot(node);
    });
}

py::object NodeAddChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool force_readable_name) {
    Node* node = handle.Resolve();
    Node* child = child_handle.Resolve();
    ExecuteSceneMutation("add_child", [&]() {
        node->AddChild(child, force_readable_name);
        child_handle.TransferToTree();
        child_handle.RefreshSnapshot(child);
    });
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
        ExecuteSceneMutation("remove_child_delete", [&]() {
            node->RemoveChild(child);
            Object::Delete(child);
            child_handle.Invalidate();
        });
        return py::none();
    }

    std::shared_ptr<PyNodeHandleState> detached_state = child_handle.state;
    ExecuteSceneMutation("remove_child_detach", [&]() {
        node->RemoveChild(child);
        child_handle.TransferToDetachedOwned(child);
    });
    return py::cast(PyNodeHandle(detached_state, ExpectedTypeForNode(child)));
}

void NodeReparent(PyNodeHandle& handle, PyNodeHandle& parent_handle) {
    Node* node = handle.Resolve();
    Node* parent = parent_handle.Resolve();
    ExecuteSceneMutation("reparent", [&]() {
        node->Reparent(parent);
        handle.TransferToTree();
        handle.RefreshSnapshot(node);
    });
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
    ExecuteSceneMutation("rename_node", [&]() {
        node->SetName(name);
        handle.RefreshSnapshot(node);
    });
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
                handle.state->scene_epoch == GetActiveAppContext().GetSceneEpoch());
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

py::dict StepInfoFromResult(const py::dict& result) {
    py::dict info;
    info["frame_count"] = result["frame_count"];
    info["simulation_time"] = result["simulation_time"];
    info["error"] = result["error"];
    return info;
}

} // namespace

void SetActiveAppContext(EngineContext* context) {
    s_active_app_context = context;
}

EngineContext& GetActiveAppContext() {
    if (s_active_app_context != nullptr) {
        return *s_active_app_context;
    }
    return Runtime().GetAppContext();
}

EngineContext* GetActiveAppContextOrNull() {
    return s_active_app_context;
}

namespace {

EngineContext& EnsureRuntimeContext() {
    if (s_active_app_context == nullptr) {
        Runtime();
    }
    return GetActiveAppContext();
}

Node* PyNodeHandle::Resolve() const {
    if (!state || state->id.IsNull()) {
        ThrowReferenceError("Gobot node handle is invalid");
    }

    EngineContext& context = GetActiveAppContext();
    if (state->ownership != PyNodeOwnership::DetachedOwned &&
        state->scene_epoch != 0 &&
        state->scene_epoch != context.GetSceneEpoch()) {
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
        state = MakeState(node, GetActiveAppContext().GetSceneEpoch(), PyNodeOwnership::DetachedOwned);
        return;
    }
    state->id = node != nullptr ? node->GetInstanceId() : ObjectID();
    state->scene_epoch = GetActiveAppContext().GetSceneEpoch();
    state->ownership = PyNodeOwnership::DetachedOwned;
    RefreshSnapshot(node);
}

std::string ExpectedTypeForNode(Node* node) {
    return node == nullptr ? "Node" : std::string(node->GetClassStringName());
}

PyNodeHandle MakeNodeHandle(Node* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyNodeHandle(node, ExpectedTypeForNode(node), GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyNode3DHandle MakeNode3DHandle(Node3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyNode3DHandle(node, ExpectedTypeForNode(node), GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyRobot3DHandle MakeRobot3DHandle(Robot3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyRobot3DHandle(node, "Robot3D", GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyLink3DHandle MakeLink3DHandle(Link3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyLink3DHandle(node, "Link3D", GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyJoint3DHandle MakeJoint3DHandle(Joint3D* node, PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyJoint3DHandle(node, "Joint3D", GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyCollisionShape3DHandle MakeCollisionShape3DHandle(CollisionShape3D* node,
                                                    PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyCollisionShape3DHandle(node, "CollisionShape3D", GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyMeshInstance3DHandle MakeMeshInstance3DHandle(MeshInstance3D* node,
                                                PyNodeOwnership ownership = PyNodeOwnership::Borrowed) {
    return PyMeshInstance3DHandle(node, "MeshInstance3D", GetActiveAppContext().GetSceneEpoch(), ownership);
}

PyNodeHandle MakeTypedNodeHandle(Node* node, PyNodeOwnership ownership) {
    if (auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node)) {
        return MakeMeshInstance3DHandle(mesh_instance, ownership);
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

template <typename Func>
void ExecuteSceneMutation(const std::string&, Func&& func) {
    std::forward<Func>(func)();
    GetActiveAppContext().NotifySceneMutated();
}

} // namespace

void RegisterRuntime(py::module_& module) {
    module.doc() = "Gobot robotics scene, simulation, and rendering engine bindings.";
}

void RegisterManualApis(py::module_& module) {
    py::enum_<JointType>(module, "JointType")
            .value("Fixed", JointType::Fixed)
            .value("Revolute", JointType::Revolute)
            .value("Continuous", JointType::Continuous)
            .value("Prismatic", JointType::Prismatic)
            .value("Floating", JointType::Floating)
            .value("Planar", JointType::Planar)
            .export_values();

    py::enum_<RobotMode>(module, "RobotMode")
            .value("Assembly", RobotMode::Assembly)
            .value("Motion", RobotMode::Motion)
            .export_values();

    py::enum_<LinkRole>(module, "LinkRole")
            .value("Physical", LinkRole::Physical)
            .value("VirtualRoot", LinkRole::VirtualRoot)
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

    py::implicitly_convertible<PyRobot3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyLink3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyJoint3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyCollisionShape3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyMeshInstance3DHandle, PyNodeHandle>();

    py::class_<EngineContext>(module, "AppContext")
            .def_property_readonly("project_path", &EngineContext::GetProjectPath)
            .def_property_readonly("scene_path", &EngineContext::GetScenePath)
            .def_property_readonly("scene_epoch", &EngineContext::GetSceneEpoch)
            .def_property_readonly("root", [](EngineContext& context) -> py::object {
                Node* root = context.GetSceneRoot();
                if (root == nullptr) {
                    return py::none();
                }
                return MakeTypedNodeObject(root);
            })
            .def_property("backend_type",
                          &EngineContext::GetBackendType,
                          &EngineContext::SetBackendType)
            .def_property_readonly("has_scene", &EngineContext::HasScene)
            .def_property_readonly("has_world", &EngineContext::HasWorld)
            .def_property_readonly("simulation_time", &EngineContext::GetSimulationTime)
            .def_property_readonly("frame_count", &EngineContext::GetFrameCount)
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
            .def("get_last_error", &EngineContext::GetLastError);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return MakeTypedNodeObject(scene.root);
            })
            .def_property_readonly("scene_epoch", [](const PyScene& scene) {
                return scene.scene_epoch;
            });

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
                        ExecuteSceneMutation("remove_root_delete", [&]() {
                            Object::Delete(node);
                            handle.Invalidate();
                        });
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
                              ExecuteSceneMutation("set_position", [&]() {
                                  node->SetPosition(PythonToVector3(value));
                              });
                          })
            .def_property("rotation_degrees",
                          [](const PyNode3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Node3D>()->GetEulerDegree());
                          },
                          [](PyNode3DHandle& handle, const py::handle& value) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSceneMutation("set_rotation_degrees", [&]() {
                                  node->SetEulerDegree(PythonToVector3(value));
                              });
                          })
            .def_property("scale",
                          [](const PyNode3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Node3D>()->GetScale());
                          },
                          [](PyNode3DHandle& handle, const py::handle& value) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSceneMutation("set_scale", [&]() {
                                  node->SetScale(PythonToVector3(value));
                              });
                          })
            .def_property("visible",
                          [](const PyNode3DHandle& handle) {
                              return handle.ResolveAs<Node3D>()->IsVisible();
                          },
                          [](PyNode3DHandle& handle, bool visible) {
                              Node3D* node = handle.ResolveAs<Node3D>();
                              ExecuteSceneMutation("set_visible", [&]() {
                                  node->SetVisible(visible);
                              });
                          });

    robot3d_class
            .def_property("source_path",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetSourcePath();
                          },
                          [](PyRobot3DHandle& handle, const std::string& source_path) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSceneMutation("set_robot_source_path", [&]() {
                                  robot->SetSourcePath(source_path);
                              });
                          })
            .def_property("mode",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetMode();
                          },
                          [](PyRobot3DHandle& handle, RobotMode mode) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSceneMutation("set_robot_mode", [&]() {
                                  robot->SetMode(mode);
                              });
                          });

    link3d_class
            .def_property("has_inertial",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->HasInertial();
                          },
                          [](PyLink3DHandle& handle, bool has_inertial) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSceneMutation("set_link_has_inertial", [&]() {
                                  link->SetHasInertial(has_inertial);
                              });
                          })
            .def_property("mass",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetMass();
                          },
                          [](PyLink3DHandle& handle, RealType mass) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSceneMutation("set_link_mass", [&]() {
                                  link->SetMass(mass);
                              });
                          })
            .def_property("center_of_mass",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetCenterOfMass());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSceneMutation("set_link_center_of_mass", [&]() {
                                  link->SetCenterOfMass(PythonToVector3(value));
                              });
                          })
            .def_property("inertia_diagonal",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetInertiaDiagonal());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSceneMutation("set_link_inertia_diagonal", [&]() {
                                  link->SetInertiaDiagonal(PythonToVector3(value));
                              });
                          })
            .def_property("role",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetRole();
                          },
                          [](PyLink3DHandle& handle, LinkRole role) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSceneMutation("set_link_role", [&]() {
                                  link->SetRole(role);
                              });
                          });

    joint3d_class
            .def_property("joint_type",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointType();
                          },
                          [](PyJoint3DHandle& handle, JointType joint_type) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_type", [&]() {
                                  joint->SetJointType(joint_type);
                              });
                          })
            .def_property("parent_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetParentLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& parent_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_parent_link", [&]() {
                                  joint->SetParentLink(parent_link);
                              });
                          })
            .def_property("child_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetChildLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& child_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_child_link", [&]() {
                                  joint->SetChildLink(child_link);
                              });
                          })
            .def_property("axis",
                          [](const PyJoint3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Joint3D>()->GetAxis());
                          },
                          [](PyJoint3DHandle& handle, const py::handle& value) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_axis", [&]() {
                                  joint->SetAxis(PythonToVector3(value));
                              });
                          })
            .def_property("lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_lower_limit", [&]() {
                                  joint->SetLowerLimit(lower_limit);
                              });
                          })
            .def_property("upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_upper_limit", [&]() {
                                  joint->SetUpperLimit(upper_limit);
                              });
                          })
            .def_property("effort_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetEffortLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType effort_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_effort_limit", [&]() {
                                  joint->SetEffortLimit(effort_limit);
                              });
                          })
            .def_property("velocity_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetVelocityLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType velocity_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_velocity_limit", [&]() {
                                  joint->SetVelocityLimit(velocity_limit);
                              });
                          })
            .def_property("joint_position",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointPosition();
                          },
                          [](PyJoint3DHandle& handle, RealType joint_position) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSceneMutation("set_joint_position", [&]() {
                                  joint->SetJointPosition(joint_position);
                              });
                          });

    collision_shape_class
            .def_property("disabled",
                          [](const PyCollisionShape3DHandle& handle) {
                              return handle.ResolveAs<CollisionShape3D>()->IsDisabled();
                          },
                          [](PyCollisionShape3DHandle& handle, bool disabled) {
                              CollisionShape3D* collision_shape = handle.ResolveAs<CollisionShape3D>();
                              ExecuteSceneMutation("set_collision_disabled", [&]() {
                                  collision_shape->SetDisabled(disabled);
                              });
                          });

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
                              ExecuteSceneMutation("set_mesh_surface_color", [&]() {
                                  mesh_instance->SetSurfaceColor({
                                          py::cast<float>(sequence[0]),
                                          py::cast<float>(sequence[1]),
                                          py::cast<float>(sequence[2]),
                                          py::cast<float>(sequence[3])
                                  });
                              });
                          });

    py::class_<PyRLControllerConfig>(module, "RLControllerConfig")
            .def(py::init<>())
            .def_readwrite("controlled_joints", &PyRLControllerConfig::controlled_joints)
            .def_readwrite("default_action", &PyRLControllerConfig::default_action)
            .def_property("joint_gains",
                          [](const PyRLControllerConfig& config) {
                              return ReflectedToPythonDict(config.joint_gains);
                          },
                          [](PyRLControllerConfig& config, py::dict gains) {
                              config.joint_gains = DictToReflected<JointControllerGains>(gains);
                          })
            .def("to_dict", &ControllerConfigToDict)
            .def_static("from_dict", &ControllerConfigFromDict, py::arg("config"));

    py::class_<PyRLEnvironment>(module, "RLEnvironment")
            .def(py::init<const std::string&, const std::string&, const std::string&>(),
                 py::arg("scene_path") = "",
                 py::arg("robot") = "robot",
                 py::arg("backend") = "null")
            .def("reset", [](PyRLEnvironment& env, std::uint32_t seed) {
                py::dict result = ReflectedToPythonDict(env.Reset(seed));
                py::dict info = StepInfoFromResult(result);
                info["ok"] = result["ok"];
                info["seed"] = result["seed"];
                return py::make_tuple(result["observation"], info);
            }, py::arg("seed") = 0)
            .def("step", [](PyRLEnvironment& env, const std::vector<RealType>& action) {
                py::dict result = ReflectedToPythonDict(env.Step(action));
                return py::make_tuple(result["observation"],
                                      result["reward"],
                                      result["terminated"],
                                      result["truncated"],
                                      StepInfoFromResult(result));
            }, py::arg("action"))
            .def("reset_result", [](PyRLEnvironment& env, std::uint32_t seed) {
                return ReflectedToPythonDict(env.Reset(seed));
            }, py::arg("seed") = 0)
            .def("step_result", [](PyRLEnvironment& env, const std::vector<RealType>& action) {
                return ReflectedToPythonDict(env.Step(action));
            }, py::arg("action"))
            .def("get_observation", [](const PyRLEnvironment& env) {
                return env.environment->GetObservation();
            })
            .def("get_action_size", [](const PyRLEnvironment& env) {
                return env.environment->GetActionSize();
            })
            .def("get_observation_size", [](const PyRLEnvironment& env) {
                return env.environment->GetObservationSize();
            })
            .def("get_controlled_joint_names", [](const PyRLEnvironment& env) {
                return env.environment->GetControlledJointNames();
            })
            .def("get_contact_link_names", [](const PyRLEnvironment& env) {
                return env.environment->GetContactLinkNames();
            })
            .def("get_action_spec", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetActionSpec());
            })
            .def("get_observation_spec", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetObservationSpec());
            })
            .def("set_reward_settings", [](PyRLEnvironment& env, py::dict settings) {
                env.environment->SetRewardSettings(DictToReflected<RLEnvironmentRewardSettings>(settings));
            })
            .def("get_reward_settings", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetRewardSettings());
            })
            .def("set_default_joint_gains", [](PyRLEnvironment& env, py::dict gains) {
                env.SetDefaultJointGains(DictToReflected<JointControllerGains>(gains));
            })
            .def("get_default_joint_gains", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.GetDefaultJointGains());
            })
            .def("get_controller_config", &MakeControllerConfigFromEnv)
            .def("apply_controller_config", &ApplyControllerConfigToEnv, py::arg("config"))
            .def("get_last_error", [](const PyRLEnvironment& env) {
                return env.environment->GetLastError();
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

    module.def("load_resource", [](const std::string& path, const std::string& type_hint) {
        EnsureRuntimeContext();
        return ResourceToPythonDict(ResourceLoader::Load(path, type_hint));
    }, py::arg("path"), py::arg("type_hint") = "");

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
