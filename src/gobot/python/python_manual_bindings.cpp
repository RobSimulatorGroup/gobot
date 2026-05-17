#include "gobot/python/python_binding_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/python/native_vector_env.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/python/python_script_runner.hpp"
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
#include "gobot/scene/scene_command.hpp"
#include "gobot/scene/scene_initializer.hpp"
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
    result["initial_position"] = joint.joint_position;
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
        robots.append(robot_dict);
    }

    result["robots"] = robots;
    result["total_link_count"] = snapshot.total_link_count;
    result["total_joint_count"] = snapshot.total_joint_count;
    result["total_collision_shape_count"] = snapshot.total_collision_shape_count;
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
            .def("set_robot_action", [](EngineContext& context,
                                        const std::string& robot,
                                        const std::vector<RealType>& action) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->SetRobotJointPositionTargetsFromNormalizedAction(robot, action)) {
                    throw std::runtime_error(simulation->GetLastError());
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
                if (!simulation->SetRobotJointPositionTargetsFromNormalizedAction(robot, joint_names, action)) {
                    throw std::runtime_error(simulation->GetLastError());
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
                if (!simulation->SetJointPositionTarget(robot, joint, target_position)) {
                    throw std::runtime_error(simulation->GetLastError());
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
                if (!simulation->SetJointVelocityTarget(robot, joint, target_velocity)) {
                    throw std::runtime_error(simulation->GetLastError());
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
                if (!simulation->SetJointEffortTarget(robot, joint, target_effort)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("target_effort"))
            .def("set_joint_passive", [](EngineContext& context,
                                         const std::string& robot,
                                         const std::string& joint) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->SetJointPassive(robot, joint)) {
                    throw std::runtime_error(simulation->GetLastError());
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
                if (!simulation->ResetJointState(robot, joint, position, velocity)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint"), py::arg("position"), py::arg("velocity") = 0.0)
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

    RegisterNativeVectorEnv(module);

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
