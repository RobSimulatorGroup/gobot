#include "gobot/python/python_binding_registry.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
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
void ExecuteSceneMutation(const std::string&, Func&& func);

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

struct NativeVectorEnvConfig {
    std::string scene;
    std::string robot;
    PhysicsBackendType backend{PhysicsBackendType::MuJoCoCpu};
    int num_envs{1};
    int batch_size{0};
    int num_workers{0};
    RealType physics_dt{1.0 / 240.0};
    int decimation{1};
    int max_episode_steps{500};
    bool auto_reset{true};
    std::vector<std::string> controlled_joints;
    std::uint64_t seed{0};
};

enum class NativeVectorActionMode {
    NormalizedPosition,
    Position,
    Velocity,
    Effort
};

struct NativeVectorActionConfig {
    std::string name;
    std::string joint;
    NativeVectorActionMode mode{NativeVectorActionMode::NormalizedPosition};
    RealType scale{1.0};
    RealType offset{0.0};
    RealType lower{-1.0};
    RealType upper{1.0};
    std::string unit{"normalized"};
    std::vector<std::string> passive_joints;
};

NativeVectorActionMode ParseNativeVectorActionMode(const std::string& value) {
    if (value == "normalized_position" ||
        value == "normalized_joint_position" ||
        value == "normalized" ||
        value == "joint_normalized_position") {
        return NativeVectorActionMode::NormalizedPosition;
    }
    if (value == "position" || value == "target_position" || value == "joint_position") {
        return NativeVectorActionMode::Position;
    }
    if (value == "velocity" || value == "target_velocity" || value == "joint_velocity") {
        return NativeVectorActionMode::Velocity;
    }
    if (value == "effort" ||
        value == "force" ||
        value == "torque" ||
        value == "target_effort" ||
        value == "joint_effort") {
        return NativeVectorActionMode::Effort;
    }
    throw std::invalid_argument("unsupported NativeVectorEnv action mode '" + value + "'");
}

std::string NativeVectorActionModeName(NativeVectorActionMode mode) {
    switch (mode) {
        case NativeVectorActionMode::NormalizedPosition:
            return "normalized_position";
        case NativeVectorActionMode::Position:
            return "position";
        case NativeVectorActionMode::Velocity:
            return "velocity";
        case NativeVectorActionMode::Effort:
            return "effort";
    }
    return "unknown";
}

struct NativeVectorWorld {
    Ref<PhysicsWorld> world;
    std::vector<RealType> previous_action;
    std::int64_t episode_length{0};
    RealType episode_return{0.0};
    std::mt19937_64 rng;
};

struct NativeVectorStepResult {
    std::vector<int> env_ids;
    std::vector<RealType> observation;
    std::vector<RealType> reward;
    std::vector<std::uint8_t> terminated;
    std::vector<std::uint8_t> truncated;
    std::vector<RealType> terminal_observation;
    std::vector<std::uint8_t> has_terminal_observation;
    std::vector<std::int64_t> episode_length;
    std::vector<RealType> episode_return;
};

class NativeVectorEnv {
public:
    explicit NativeVectorEnv(NativeVectorEnvConfig config)
        : NativeVectorEnv(std::move(config), {}) {
    }

    NativeVectorEnv(NativeVectorEnvConfig config, std::vector<NativeVectorActionConfig> action_configs)
        : cfg_(std::move(config)),
          action_configs_(std::move(action_configs)) {
        try {
            if (cfg_.num_envs < 1) {
                throw std::invalid_argument("NativeVectorEnv num_envs must be at least 1");
            }
            if (cfg_.decimation < 1) {
                throw std::invalid_argument("NativeVectorEnv decimation must be at least 1");
            }
            if (cfg_.physics_dt <= 0.0) {
                throw std::invalid_argument("NativeVectorEnv physics_dt must be greater than zero");
            }
            if (cfg_.max_episode_steps < 1) {
                throw std::invalid_argument("NativeVectorEnv max_episode_steps must be at least 1");
            }
            if (cfg_.batch_size <= 0) {
                cfg_.batch_size = cfg_.num_envs;
            }
            if (cfg_.batch_size < 1 || cfg_.batch_size > cfg_.num_envs) {
                throw std::invalid_argument("NativeVectorEnv batch_size must be in [1, num_envs]");
            }
            if (cfg_.num_workers <= 0) {
                const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
                cfg_.num_workers = std::min<int>(cfg_.num_envs, static_cast<int>(hardware_threads));
            }
            if (cfg_.num_workers < 1) {
                throw std::invalid_argument("NativeVectorEnv num_workers must be at least 1");
            }
            cfg_.num_workers = std::min(cfg_.num_workers, cfg_.num_envs);
            if (cfg_.scene.empty()) {
                throw std::invalid_argument("NativeVectorEnv requires a scene path");
            }

            LoadScene();
            BuildWorlds();
            ConfigureRobotAndSpecs();
            StartWorkers();
            ResetAll(std::nullopt);
        } catch (...) {
            ShutdownWorkers();
            ClearWorldsAndRoots();
            throw;
        }
    }

    ~NativeVectorEnv() {
        ShutdownWorkers();
        ClearWorldsAndRoots();
    }

    int GetNumEnvs() const {
        return cfg_.num_envs;
    }

    int GetBatchSize() const {
        return cfg_.batch_size;
    }

    int GetNumWorkers() const {
        return cfg_.num_workers;
    }

    int GetObservationSize() const {
        return observation_size_;
    }

    int GetActionSize() const {
        return action_size_;
    }

    RealType GetEnvDt() const {
        return cfg_.physics_dt * cfg_.decimation;
    }

    py::dict GetObservationSpec() const {
        py::dict spec;
        py::list names;
        py::list lower;
        py::list upper;
        py::list units;
        for (const NativeObservationTerm& term : observation_terms_) {
            names.append(term.name);
            lower.append(term.lower);
            upper.append(term.upper);
            units.append(term.unit);
        }
        spec["version"] = "gobot.native_vector.observation.v1";
        spec["names"] = names;
        spec["lower_bounds"] = lower;
        spec["upper_bounds"] = upper;
        spec["units"] = units;
        return spec;
    }

    py::dict GetActionSpec() const {
        py::dict spec;
        py::list names;
        py::list lower;
        py::list upper;
        py::list units;
        for (const NativeActionTerm& term : action_terms_) {
            names.append(term.name);
            lower.append(term.lower);
            upper.append(term.upper);
            units.append(term.unit);
        }
        spec["version"] = "gobot.native_vector.action.v1";
        spec["names"] = names;
        spec["lower_bounds"] = lower;
        spec["upper_bounds"] = upper;
        spec["units"] = units;
        return spec;
    }

    py::tuple Reset(py::object seed = py::none(), py::object env_ids = py::none()) {
        const std::optional<std::uint64_t> parsed_seed = ParseOptionalSeed(seed);
        const std::vector<int> ids = ParseEnvIds(env_ids);
        std::vector<int> output_ids;
        std::vector<RealType> observations;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            if (ids.empty()) {
                output_ids = AllEnvIds();
                ResetAll(parsed_seed);
                observations = Observe(output_ids);
            } else {
                output_ids = ids;
                ResetIds(output_ids, parsed_seed);
                observations = Observe(output_ids);
            }
        }
        return MakeResetReturn(output_ids, observations, parsed_seed);
    }

    py::tuple Step(py::object action, py::object env_ids = py::none()) {
        const std::vector<int> ids = ParseEnvIds(env_ids);
        const std::vector<int> active_ids = ids.empty() ? AllEnvIds() : ids;
        const std::vector<RealType> actions = ParseActionArray(action, static_cast<int>(active_ids.size()));
        NativeVectorStepResult result;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            result = StepIds(active_ids, actions);
        }
        return MakeStepReturn(result);
    }

    void AsyncReset() {
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            async_queue_.clear();
            pending_actions_.clear();
            std::vector<int> all_ids = AllEnvIds();
            NativeVectorStepResult result;
            result.env_ids = std::move(all_ids);
            result.reward.assign(result.env_ids.size(), 0.0);
            result.terminated.assign(result.env_ids.size(), 0);
            result.truncated.assign(result.env_ids.size(), 0);
            result.terminal_observation.assign(result.env_ids.size() * observation_size_, 0.0);
            result.has_terminal_observation.assign(result.env_ids.size(), 0);
            ResetAll(std::nullopt);
            result.observation = Observe(result.env_ids);
            result.episode_length.reserve(result.env_ids.size());
            result.episode_return.reserve(result.env_ids.size());
            for (int env_id : result.env_ids) {
                result.episode_length.push_back(worlds_[env_id].episode_length);
                result.episode_return.push_back(worlds_[env_id].episode_return);
            }
            EnqueueBatches(result);
        }
    }

    void Send(py::object action, py::object env_ids = py::none()) {
        const std::vector<int> ids = ParseEnvIds(env_ids);
        const std::vector<int> active_ids = ids.empty() ? AllEnvIds() : ids;
        const std::vector<RealType> actions = ParseActionArray(action, static_cast<int>(active_ids.size()));
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            NativeVectorStepResult result = StepIds(active_ids, actions);
            EnqueueBatches(result);
        }
    }

    py::tuple Recv() {
        NativeVectorStepResult result;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> api_lock(api_mutex_);
            if (async_queue_.empty()) {
                throw std::runtime_error("NativeVectorEnv.recv() has no ready batch; call async_reset() or send() first");
            }
            result = std::move(async_queue_.front());
            async_queue_.pop_front();
        }
        return MakeStepReturn(result);
    }

private:
    struct NativeActionTerm {
        std::string name;
        std::string joint;
        NativeVectorActionMode mode{NativeVectorActionMode::NormalizedPosition};
        RealType scale{1.0};
        RealType offset{0.0};
        RealType lower{-1.0};
        RealType upper{1.0};
        std::string unit{"normalized"};
        std::vector<std::string> passive_joints;
    };

    enum class ObservationTermType {
        Time,
        EpisodeProgress,
        JointPosition,
        JointVelocity,
        PreviousAction
    };

    struct NativeObservationTerm {
        std::string name;
        ObservationTermType type{ObservationTermType::JointPosition};
        std::string joint;
        int action_index{-1};
        RealType lower{-std::numeric_limits<RealType>::infinity()};
        RealType upper{std::numeric_limits<RealType>::infinity()};
        std::string unit;
    };

    void LoadScene() {
        Ref<Resource> resource =
                ResourceLoader::Load(cfg_.scene, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
        packed_scene_ = dynamic_pointer_cast<PackedScene>(resource);
        if (!packed_scene_.IsValid()) {
            throw std::runtime_error("NativeVectorEnv failed to load PackedScene from '" + cfg_.scene + "'");
        }
    }

    void BuildWorlds() {
        worlds_.clear();
        worlds_.resize(static_cast<std::size_t>(cfg_.num_envs));
        roots_.clear();
        roots_.resize(static_cast<std::size_t>(cfg_.num_envs), nullptr);

        PhysicsWorldSettings settings;
        settings.fixed_time_step = cfg_.physics_dt;
        for (int env_id = 0; env_id < cfg_.num_envs; ++env_id) {
            Node* root = packed_scene_->Instantiate();
            if (root == nullptr) {
                throw std::runtime_error("NativeVectorEnv failed to instantiate PackedScene for env " +
                                         std::to_string(env_id));
            }
            roots_[env_id] = root;

            Ref<PhysicsWorld> world = PhysicsServer::CreateWorldForBackend(cfg_.backend, settings);
            if (!world.IsValid()) {
                throw std::runtime_error("NativeVectorEnv failed to create physics world");
            }
            if (world->GetBackendType() != cfg_.backend) {
                throw std::runtime_error("NativeVectorEnv requested physics backend is unavailable");
            }
            if (!world->BuildFromScene(root)) {
                const std::string error = world->GetLastError();
                throw std::runtime_error("NativeVectorEnv failed to build physics world: " + error);
            }
            world->Reset();
            worlds_[env_id].world = world;
            worlds_[env_id].rng.seed(cfg_.seed + static_cast<std::uint64_t>(env_id) * 9973ULL);
        }
    }

    void ClearWorldsAndRoots() {
        for (NativeVectorWorld& world : worlds_) {
            world.world.Reset();
        }
        worlds_.clear();
        for (Node*& root : roots_) {
            if (root != nullptr) {
                Object::Delete(root);
                root = nullptr;
            }
        }
        roots_.clear();
        packed_scene_.Reset();
    }

    void ConfigureRobotAndSpecs() {
        const PhysicsSceneSnapshot& snapshot = worlds_.front().world->GetSceneSnapshot();
        const PhysicsRobotSnapshot* robot = nullptr;
        if (!cfg_.robot.empty()) {
            for (const PhysicsRobotSnapshot& candidate : snapshot.robots) {
                if (candidate.name == cfg_.robot) {
                    robot = &candidate;
                    break;
                }
            }
        } else if (!snapshot.robots.empty()) {
            robot = &snapshot.robots.front();
            cfg_.robot = robot->name;
        }
        if (robot == nullptr) {
            throw std::runtime_error("NativeVectorEnv scene has no robot named '" + cfg_.robot + "'");
        }

        if (action_configs_.empty() && cfg_.controlled_joints.empty()) {
            for (const PhysicsJointSnapshot& joint : robot->joints) {
                const auto joint_type = static_cast<JointType>(joint.joint_type);
                if (joint_type == JointType::Revolute ||
                    joint_type == JointType::Continuous ||
                    joint_type == JointType::Prismatic) {
                    cfg_.controlled_joints.push_back(joint.name);
                }
            }
        }
        if (action_configs_.empty() && cfg_.controlled_joints.empty()) {
            throw std::runtime_error("NativeVectorEnv robot '" + cfg_.robot + "' has no controllable joints");
        }

        std::unordered_set<std::string> robot_joint_names;
        for (const PhysicsJointSnapshot& joint : robot->joints) {
            robot_joint_names.insert(joint.name);
        }
        if (action_configs_.empty()) {
            for (const std::string& joint : cfg_.controlled_joints) {
                action_configs_.push_back({
                        joint + "/target_position_normalized",
                        joint,
                        NativeVectorActionMode::NormalizedPosition,
                        1.0,
                        0.0,
                        -1.0,
                        1.0,
                        "normalized",
                        {}
                });
            }
        }

        for (const NativeVectorActionConfig& config : action_configs_) {
            if (!robot_joint_names.contains(config.joint)) {
                throw std::runtime_error("NativeVectorEnv robot '" + cfg_.robot + "' has no joint named '" + config.joint + "'");
            }
            action_terms_.push_back({
                    config.name,
                    config.joint,
                    config.mode,
                    config.scale,
                    config.offset,
                    config.lower,
                    config.upper,
                    config.unit,
                    config.passive_joints
            });
        }
        action_size_ = static_cast<int>(action_terms_.size());
        for (NativeVectorWorld& world : worlds_) {
            world.previous_action.assign(static_cast<std::size_t>(action_size_), 0.0);
        }

        observation_terms_.push_back({"time", ObservationTermType::Time, "", -1, 0.0,
                                      std::numeric_limits<RealType>::infinity(), "s"});
        observation_terms_.push_back({"episode_progress", ObservationTermType::EpisodeProgress, "", -1, 0.0,
                                      1.0, "ratio"});
        for (const PhysicsJointSnapshot& joint : robot->joints) {
            const auto joint_type = static_cast<JointType>(joint.joint_type);
            if (joint_type == JointType::Fixed) {
                continue;
            }
            RealType lower = -std::numeric_limits<RealType>::infinity();
            RealType upper = std::numeric_limits<RealType>::infinity();
            if (joint.upper_limit > joint.lower_limit && joint_type != JointType::Continuous) {
                lower = joint.lower_limit;
                upper = joint.upper_limit;
            }
            observation_terms_.push_back({joint.name + "/position",
                                          ObservationTermType::JointPosition,
                                          joint.name,
                                          -1,
                                          lower,
                                          upper,
                                          "rad_or_m"});
            observation_terms_.push_back({joint.name + "/velocity",
                                          ObservationTermType::JointVelocity,
                                          joint.name,
                                          -1,
                                          -std::numeric_limits<RealType>::infinity(),
                                          std::numeric_limits<RealType>::infinity(),
                                          "rad_or_m/s"});
        }
        for (int action_index = 0; action_index < action_size_; ++action_index) {
            observation_terms_.push_back({action_terms_[action_index].name + "/previous_action",
                                          ObservationTermType::PreviousAction,
                                          "",
                                          action_index,
                                          -1.0,
                                          1.0,
                                          "normalized"});
        }
        observation_size_ = static_cast<int>(observation_terms_.size());
    }

    std::optional<std::uint64_t> ParseOptionalSeed(const py::object& seed) const {
        if (seed.is_none()) {
            return std::nullopt;
        }
        return static_cast<std::uint64_t>(py::cast<std::uint64_t>(seed));
    }

    std::vector<int> ParseEnvIds(const py::object& env_ids) const {
        std::vector<int> ids;
        if (env_ids.is_none()) {
            return ids;
        }
        if (py::isinstance<py::int_>(env_ids)) {
            ids.push_back(py::cast<int>(env_ids));
        } else {
            for (const py::handle item : env_ids) {
                ids.push_back(py::cast<int>(item));
            }
        }
        for (int id : ids) {
            if (id < 0 || id >= cfg_.num_envs) {
                throw std::out_of_range("NativeVectorEnv env_id out of range: " + std::to_string(id));
            }
        }
        return ids;
    }

    std::vector<int> AllEnvIds() const {
        std::vector<int> ids(static_cast<std::size_t>(cfg_.num_envs));
        for (int index = 0; index < cfg_.num_envs; ++index) {
            ids[index] = index;
        }
        return ids;
    }

    std::vector<RealType> ParseActionArray(const py::object& action, int rows) const {
        py::array_t<RealType, py::array::c_style | py::array::forcecast> array(action);
        const py::buffer_info info = array.request();
        if (info.ndim == 1) {
            if (action_size_ != 1 && info.shape[0] != action_size_) {
                throw std::invalid_argument("NativeVectorEnv action has wrong shape");
            }
            if (rows == 1 && info.shape[0] == action_size_) {
                const auto* data = static_cast<const RealType*>(info.ptr);
                return std::vector<RealType>(data, data + action_size_);
            }
            if (action_size_ == 1 && info.shape[0] == rows) {
                const auto* data = static_cast<const RealType*>(info.ptr);
                return std::vector<RealType>(data, data + rows);
            }
            throw std::invalid_argument("NativeVectorEnv action first dimension must match env_ids");
        }
        if (info.ndim != 2 ||
            info.shape[0] != rows ||
            info.shape[1] != action_size_) {
            throw std::invalid_argument("NativeVectorEnv action must have shape (batch, action_dim)");
        }
        const auto* data = static_cast<const RealType*>(info.ptr);
        return std::vector<RealType>(data, data + rows * action_size_);
    }

    void ResetAll(std::optional<std::uint64_t> seed) {
        std::vector<int> ids = AllEnvIds();
        ResetIds(ids, seed);
    }

    void ResetOne(int env_id, std::optional<std::uint64_t> seed) {
        NativeVectorWorld& slot = worlds_[env_id];
        if (seed.has_value()) {
            slot.rng.seed(*seed + static_cast<std::uint64_t>(env_id) * 9973ULL);
        }
        slot.world->Reset();
        slot.previous_action.assign(static_cast<std::size_t>(action_size_), 0.0);
        slot.episode_length = 0;
        slot.episode_return = 0.0;
    }

    void ResetIds(const std::vector<int>& ids, std::optional<std::uint64_t> seed) {
        ParallelForRows(ids.size(), [&](std::size_t row) {
            ResetOne(ids[row], seed);
        });
    }

    std::vector<RealType> Observe(const std::vector<int>& ids) const {
        std::vector<RealType> observations(ids.size() * static_cast<std::size_t>(observation_size_), 0.0);
        ParallelForRows(ids.size(), [&](std::size_t row) {
            const std::vector<RealType> row_observation = ObserveOne(ids[row]);
            std::copy(row_observation.begin(),
                      row_observation.end(),
                      observations.begin() + static_cast<std::ptrdiff_t>(row * observation_size_));
        });
        return observations;
    }

    std::vector<RealType> ObserveOne(int env_id) const {
        std::vector<RealType> observation(static_cast<std::size_t>(observation_size_), 0.0);
        const NativeVectorWorld& slot = worlds_[env_id];
        const PhysicsRobotState* robot = FindRobotState(slot.world->GetSceneState(), cfg_.robot);
        if (robot == nullptr) {
            throw std::runtime_error("NativeVectorEnv runtime state has no robot named '" + cfg_.robot + "'");
        }
        for (int col = 0; col < observation_size_; ++col) {
            observation[col] = ComputeObservationTerm(slot, *robot, observation_terms_[col]);
        }
        return observation;
    }

    RealType ComputeObservationTerm(const NativeVectorWorld& slot,
                                    const PhysicsRobotState& robot,
                                    const NativeObservationTerm& term) const {
        switch (term.type) {
            case ObservationTermType::Time:
                return static_cast<RealType>(slot.episode_length) * GetEnvDt();
            case ObservationTermType::EpisodeProgress:
                return std::min<RealType>(static_cast<RealType>(slot.episode_length) /
                                                  static_cast<RealType>(cfg_.max_episode_steps),
                                          1.0);
            case ObservationTermType::JointPosition: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                return joint == nullptr ? 0.0 : joint->position;
            }
            case ObservationTermType::JointVelocity: {
                const PhysicsJointState* joint = FindJointState(robot, term.joint);
                return joint == nullptr ? 0.0 : joint->velocity;
            }
            case ObservationTermType::PreviousAction:
                if (term.action_index >= 0 &&
                    term.action_index < static_cast<int>(slot.previous_action.size())) {
                    return slot.previous_action[term.action_index];
                }
                return 0.0;
        }
        return 0.0;
    }

    const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state, const std::string& robot_name) const {
        for (const PhysicsRobotState& robot : state.robots) {
            if (robot.name == robot_name) {
                return &robot;
            }
        }
        return nullptr;
    }

    const PhysicsJointState* FindJointState(const PhysicsRobotState& robot, const std::string& joint_name) const {
        for (const PhysicsJointState& joint : robot.joints) {
            if (joint.joint_name == joint_name) {
                return &joint;
            }
        }
        return nullptr;
    }

    const PhysicsJointSnapshot* FindJointSnapshot(const PhysicsRobotSnapshot& robot, const std::string& joint_name) const {
        for (const PhysicsJointSnapshot& joint : robot.joints) {
            if (joint.name == joint_name) {
                return &joint;
            }
        }
        return nullptr;
    }

    void StartWorkers() {
        const int helper_count = std::max(0, cfg_.num_workers - 1);
        if (helper_count == 0) {
            return;
        }
        workers_.reserve(static_cast<std::size_t>(helper_count));
        for (int worker_index = 0; worker_index < helper_count; ++worker_index) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    void ShutdownWorkers() {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            workers_stop_ = true;
            active_task_ = nullptr;
            active_row_count_ = 0;
            ++task_epoch_;
        }
        worker_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    void WorkerLoop() {
        std::size_t seen_epoch = 0;
        while (true) {
            std::function<void(std::size_t)> task;
            std::size_t row_count = 0;
            {
                std::unique_lock<std::mutex> lock(worker_mutex_);
                worker_cv_.wait(lock, [&]() { return workers_stop_ || task_epoch_ != seen_epoch; });
                if (workers_stop_) {
                    return;
                }
                seen_epoch = task_epoch_;
                task = active_task_;
                row_count = active_row_count_;
            }

            RunWorkerTaskRows(task, row_count);

            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                if (active_helper_count_ > 0) {
                    --active_helper_count_;
                }
                if (active_helper_count_ == 0) {
                    worker_done_cv_.notify_one();
                }
            }
        }
    }

    void RunWorkerTaskRows(const std::function<void(std::size_t)>& task, std::size_t row_count) const {
        while (true) {
            const std::size_t row = next_worker_row_.fetch_add(1, std::memory_order_relaxed);
            if (row >= row_count) {
                break;
            }
            try {
                task(row);
            } catch (...) {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                if (!worker_error_) {
                    worker_error_ = std::current_exception();
                }
                break;
            }
        }
    }

    template <typename Func>
    void ParallelForRows(std::size_t row_count, Func&& func) const {
        if (row_count == 0) {
            return;
        }
        if (row_count == 1 || workers_.empty()) {
            for (std::size_t row = 0; row < row_count; ++row) {
                func(row);
            }
            return;
        }

        std::function<void(std::size_t)> task(std::forward<Func>(func));
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            active_task_ = task;
            active_row_count_ = row_count;
            next_worker_row_.store(0, std::memory_order_relaxed);
            active_helper_count_ = static_cast<int>(workers_.size());
            worker_error_ = nullptr;
            ++task_epoch_;
        }
        worker_cv_.notify_all();
        RunWorkerTaskRows(task, row_count);

        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_done_cv_.wait(lock, [&]() { return active_helper_count_ == 0; });
            error = worker_error_;
            active_task_ = nullptr;
            active_row_count_ = 0;
            worker_error_ = nullptr;
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

    NativeVectorStepResult StepIds(const std::vector<int>& ids, const std::vector<RealType>& actions) {
        NativeVectorStepResult result;
        result.env_ids = ids;
        result.reward.assign(ids.size(), 0.0);
        result.terminated.assign(ids.size(), 0);
        result.truncated.assign(ids.size(), 0);
        result.has_terminal_observation.assign(ids.size(), 0);
        result.terminal_observation.assign(ids.size() * static_cast<std::size_t>(observation_size_), 0.0);
        result.episode_length.assign(ids.size(), 0);
        result.episode_return.assign(ids.size(), 0.0);

        ParallelForRows(ids.size(), [&](std::size_t row) {
            const int env_id = ids[row];
            StepOne(env_id,
                    std::vector<RealType>(actions.begin() + static_cast<std::ptrdiff_t>(row * action_size_),
                                          actions.begin() + static_cast<std::ptrdiff_t>((row + 1) * action_size_)),
                    result,
                    row);
        });
        result.observation = Observe(ids);
        return result;
    }

    void StepOne(int env_id,
                 const std::vector<RealType>& action,
                 NativeVectorStepResult& result,
                 std::size_t row) {
        NativeVectorWorld& slot = worlds_[env_id];
        const PhysicsSceneSnapshot& snapshot = slot.world->GetSceneSnapshot();
        const PhysicsRobotSnapshot* robot_snapshot = nullptr;
        for (const PhysicsRobotSnapshot& candidate : snapshot.robots) {
            if (candidate.name == cfg_.robot) {
                robot_snapshot = &candidate;
                break;
            }
        }
        if (robot_snapshot == nullptr) {
            throw std::runtime_error("NativeVectorEnv scene snapshot has no robot named '" + cfg_.robot + "'");
        }

        for (int action_index = 0; action_index < action_size_; ++action_index) {
            const NativeActionTerm& term = action_terms_[action_index];
            const PhysicsJointSnapshot* joint_snapshot = FindJointSnapshot(*robot_snapshot, term.joint);
            const PhysicsRobotState* robot_state = FindRobotState(slot.world->GetSceneState(), cfg_.robot);
            const PhysicsJointState* joint_state =
                    robot_state == nullptr ? nullptr : FindJointState(*robot_state, term.joint);
            if (joint_snapshot == nullptr || joint_state == nullptr) {
                throw std::runtime_error("NativeVectorEnv cannot find controlled joint '" + term.joint + "'");
            }
            const RealType clipped = std::clamp(action[action_index], term.lower, term.upper);
            for (const std::string& passive_joint : term.passive_joints) {
                if (!slot.world->SetJointControl(cfg_.robot,
                                                 passive_joint,
                                                 PhysicsJointControlMode::Passive,
                                                 0.0)) {
                    throw std::runtime_error(slot.world->GetLastError());
                }
            }

            PhysicsJointControlMode control_mode = PhysicsJointControlMode::Position;
            RealType target = term.offset + clipped * term.scale;
            switch (term.mode) {
                case NativeVectorActionMode::NormalizedPosition: {
                    JointControllerLimits limits = MakeJointControllerLimits(*joint_snapshot);
                    if (static_cast<JointType>(joint_snapshot->joint_type) == JointType::Continuous) {
                        limits.has_position_limits = false;
                    }
                    target = JointController::MapNormalizedActionToTargetPosition(clipped,
                                                                                  limits,
                                                                                  joint_state->position,
                                                                                  term.scale);
                    control_mode = PhysicsJointControlMode::Position;
                    break;
                }
                case NativeVectorActionMode::Position:
                    control_mode = PhysicsJointControlMode::Position;
                    break;
                case NativeVectorActionMode::Velocity:
                    control_mode = PhysicsJointControlMode::Velocity;
                    break;
                case NativeVectorActionMode::Effort:
                    control_mode = PhysicsJointControlMode::Effort;
                    break;
            }
            if (!slot.world->SetJointControl(cfg_.robot, term.joint, control_mode, target)) {
                throw std::runtime_error(slot.world->GetLastError());
            }
            slot.previous_action[action_index] = clipped;
        }

        for (int substep = 0; substep < cfg_.decimation; ++substep) {
            slot.world->Step(cfg_.physics_dt);
        }
        slot.episode_length += 1;
        result.reward[row] = GetEnvDt();
        slot.episode_return += result.reward[row];
        result.truncated[row] = slot.episode_length >= cfg_.max_episode_steps ? 1 : 0;
        result.episode_length[row] = slot.episode_length;
        result.episode_return[row] = slot.episode_return;

        if (result.terminated[row] != 0 || result.truncated[row] != 0) {
            std::vector<int> single_id{env_id};
            std::vector<RealType> terminal = ObserveOne(env_id);
            std::copy(terminal.begin(),
                      terminal.end(),
                      result.terminal_observation.begin() +
                              static_cast<std::ptrdiff_t>(row * observation_size_));
            result.has_terminal_observation[row] = 1;
            if (cfg_.auto_reset) {
                ResetOne(env_id, std::nullopt);
            }
        }
    }

    void EnqueueBatches(const NativeVectorStepResult& result) {
        for (std::size_t begin = 0; begin < result.env_ids.size(); begin += static_cast<std::size_t>(cfg_.batch_size)) {
            const std::size_t end = std::min(result.env_ids.size(), begin + static_cast<std::size_t>(cfg_.batch_size));
            NativeVectorStepResult batch;
            batch.env_ids.assign(result.env_ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                 result.env_ids.begin() + static_cast<std::ptrdiff_t>(end));
            batch.reward.assign(result.reward.begin() + static_cast<std::ptrdiff_t>(begin),
                                result.reward.begin() + static_cast<std::ptrdiff_t>(end));
            batch.terminated.assign(result.terminated.begin() + static_cast<std::ptrdiff_t>(begin),
                                    result.terminated.begin() + static_cast<std::ptrdiff_t>(end));
            batch.truncated.assign(result.truncated.begin() + static_cast<std::ptrdiff_t>(begin),
                                   result.truncated.begin() + static_cast<std::ptrdiff_t>(end));
            batch.has_terminal_observation.assign(result.has_terminal_observation.begin() + static_cast<std::ptrdiff_t>(begin),
                                                  result.has_terminal_observation.begin() + static_cast<std::ptrdiff_t>(end));
            batch.episode_length.assign(result.episode_length.begin() + static_cast<std::ptrdiff_t>(begin),
                                        result.episode_length.begin() + static_cast<std::ptrdiff_t>(end));
            batch.episode_return.assign(result.episode_return.begin() + static_cast<std::ptrdiff_t>(begin),
                                        result.episode_return.begin() + static_cast<std::ptrdiff_t>(end));
            batch.observation.resize((end - begin) * static_cast<std::size_t>(observation_size_));
            batch.terminal_observation.resize((end - begin) * static_cast<std::size_t>(observation_size_));
            for (std::size_t row = begin; row < end; ++row) {
                const std::size_t dst = row - begin;
                std::copy(result.observation.begin() + static_cast<std::ptrdiff_t>(row * observation_size_),
                          result.observation.begin() + static_cast<std::ptrdiff_t>((row + 1) * observation_size_),
                          batch.observation.begin() + static_cast<std::ptrdiff_t>(dst * observation_size_));
                std::copy(result.terminal_observation.begin() + static_cast<std::ptrdiff_t>(row * observation_size_),
                          result.terminal_observation.begin() + static_cast<std::ptrdiff_t>((row + 1) * observation_size_),
                          batch.terminal_observation.begin() + static_cast<std::ptrdiff_t>(dst * observation_size_));
            }
            async_queue_.push_back(std::move(batch));
        }
    }

    py::array_t<RealType> MakeMatrix(const std::vector<RealType>& values, int rows, int cols) const {
        py::array_t<RealType> array({rows, cols});
        std::copy(values.begin(), values.end(), static_cast<RealType*>(array.request().ptr));
        return array;
    }

    py::array_t<RealType> MakeVector(const std::vector<RealType>& values) const {
        py::array_t<RealType> array({static_cast<py::ssize_t>(values.size())});
        std::copy(values.begin(), values.end(), static_cast<RealType*>(array.request().ptr));
        return array;
    }

    py::array_t<bool> MakeBoolVector(const std::vector<std::uint8_t>& values) const {
        py::array_t<bool> array({static_cast<py::ssize_t>(values.size())});
        auto* data = static_cast<bool*>(array.request().ptr);
        for (std::size_t index = 0; index < values.size(); ++index) {
            data[index] = values[index] != 0;
        }
        return array;
    }

    py::array_t<std::int64_t> MakeIntVector(const std::vector<std::int64_t>& values) const {
        py::array_t<std::int64_t> array({static_cast<py::ssize_t>(values.size())});
        std::copy(values.begin(), values.end(), static_cast<std::int64_t*>(array.request().ptr));
        return array;
    }

    py::array_t<std::int64_t> MakeEnvIdVector(const std::vector<int>& values) const {
        py::array_t<std::int64_t> array({static_cast<py::ssize_t>(values.size())});
        auto* data = static_cast<std::int64_t*>(array.request().ptr);
        for (std::size_t index = 0; index < values.size(); ++index) {
            data[index] = values[index];
        }
        return array;
    }

    py::tuple MakeResetReturn(const std::vector<int>& ids,
                              const std::vector<RealType>& observations,
                              std::optional<std::uint64_t> seed) const {
        py::dict info;
        info["env_id"] = MakeEnvIdVector(ids);
        if (seed.has_value()) {
            info["seed"] = *seed;
        } else {
            info["seed"] = py::none();
        }
        return py::make_tuple(MakeMatrix(observations, static_cast<int>(ids.size()), observation_size_), info);
    }

    py::tuple MakeStepReturn(const NativeVectorStepResult& result) const {
        py::dict info;
        info["env_id"] = MakeEnvIdVector(result.env_ids);
        info["episode_length"] = MakeIntVector(result.episode_length);
        info["episode_return"] = MakeVector(result.episode_return);
        info["has_terminal_observation"] = MakeBoolVector(result.has_terminal_observation);
        if (std::any_of(result.has_terminal_observation.begin(),
                        result.has_terminal_observation.end(),
                        [](std::uint8_t value) { return value != 0; })) {
            info["terminal_observation"] = MakeMatrix(result.terminal_observation,
                                                       static_cast<int>(result.env_ids.size()),
                                                       observation_size_);
        }
        return py::make_tuple(MakeMatrix(result.observation, static_cast<int>(result.env_ids.size()), observation_size_),
                              MakeVector(result.reward),
                              MakeBoolVector(result.terminated),
                              MakeBoolVector(result.truncated),
                              info);
    }

    NativeVectorEnvConfig cfg_;
    Ref<PackedScene> packed_scene_;
    std::vector<Node*> roots_;
    std::vector<NativeVectorWorld> worlds_;
    std::deque<NativeVectorStepResult> async_queue_;
    std::unordered_map<int, std::vector<RealType>> pending_actions_;
    std::vector<NativeVectorActionConfig> action_configs_;
    std::vector<NativeActionTerm> action_terms_;
    std::vector<NativeObservationTerm> observation_terms_;
    int action_size_{0};
    int observation_size_{0};

    mutable std::mutex api_mutex_;
    mutable std::mutex worker_mutex_;
    mutable std::condition_variable worker_cv_;
    mutable std::condition_variable worker_done_cv_;
    mutable std::vector<std::thread> workers_;
    mutable std::function<void(std::size_t)> active_task_;
    mutable std::atomic_size_t next_worker_row_{0};
    mutable std::size_t active_row_count_{0};
    mutable int active_helper_count_{0};
    mutable std::size_t task_epoch_{0};
    mutable bool workers_stop_{false};
    mutable std::exception_ptr worker_error_;
};

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

template <typename Func>
void ExecuteSceneMutation(const std::string&, Func&& func) {
    std::unique_ptr<SceneCommand> command = std::forward<Func>(func)();
    if (!GetActiveAppContext().ExecuteSceneCommand(std::move(command))) {
        throw std::runtime_error("failed to execute Gobot scene command");
    }
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

    py::class_<NativeVectorEnvConfig>(module, "NativeVectorEnvConfig")
            .def(py::init<>())
            .def_readwrite("scene", &NativeVectorEnvConfig::scene)
            .def_readwrite("robot", &NativeVectorEnvConfig::robot)
            .def_readwrite("backend", &NativeVectorEnvConfig::backend)
            .def_readwrite("num_envs", &NativeVectorEnvConfig::num_envs)
            .def_readwrite("batch_size", &NativeVectorEnvConfig::batch_size)
            .def_readwrite("num_workers", &NativeVectorEnvConfig::num_workers)
            .def_readwrite("physics_dt", &NativeVectorEnvConfig::physics_dt)
            .def_readwrite("decimation", &NativeVectorEnvConfig::decimation)
            .def_readwrite("max_episode_steps", &NativeVectorEnvConfig::max_episode_steps)
            .def_readwrite("auto_reset", &NativeVectorEnvConfig::auto_reset)
            .def_readwrite("controlled_joints", &NativeVectorEnvConfig::controlled_joints)
            .def_readwrite("seed", &NativeVectorEnvConfig::seed);

    py::enum_<NativeVectorActionMode>(module, "NativeVectorActionMode")
            .value("NormalizedPosition", NativeVectorActionMode::NormalizedPosition)
            .value("Position", NativeVectorActionMode::Position)
            .value("Velocity", NativeVectorActionMode::Velocity)
            .value("Effort", NativeVectorActionMode::Effort)
            .export_values();

    py::class_<NativeVectorActionConfig>(module, "NativeVectorActionConfig")
            .def(py::init<>())
            .def_readwrite("name", &NativeVectorActionConfig::name)
            .def_readwrite("joint", &NativeVectorActionConfig::joint)
            .def_readwrite("mode", &NativeVectorActionConfig::mode)
            .def_readwrite("scale", &NativeVectorActionConfig::scale)
            .def_readwrite("offset", &NativeVectorActionConfig::offset)
            .def_readwrite("lower", &NativeVectorActionConfig::lower)
            .def_readwrite("upper", &NativeVectorActionConfig::upper)
            .def_readwrite("unit", &NativeVectorActionConfig::unit)
            .def_readwrite("passive_joints", &NativeVectorActionConfig::passive_joints);

    py::class_<NativeVectorEnv>(module, "NativeVectorEnv")
            .def(py::init<NativeVectorEnvConfig, std::vector<NativeVectorActionConfig>>(),
                 py::arg("config"),
                 py::arg("actions") = std::vector<NativeVectorActionConfig>{})
            .def_property_readonly("num_envs", &NativeVectorEnv::GetNumEnvs)
            .def_property_readonly("batch_size", &NativeVectorEnv::GetBatchSize)
            .def_property_readonly("num_workers", &NativeVectorEnv::GetNumWorkers)
            .def_property_readonly("observation_size", &NativeVectorEnv::GetObservationSize)
            .def_property_readonly("action_size", &NativeVectorEnv::GetActionSize)
            .def_property_readonly("env_dt", &NativeVectorEnv::GetEnvDt)
            .def_property_readonly("observation_spec", &NativeVectorEnv::GetObservationSpec)
            .def_property_readonly("action_spec", &NativeVectorEnv::GetActionSpec)
            .def("reset", &NativeVectorEnv::Reset,
                 py::arg("seed") = py::none(),
                 py::arg("env_ids") = py::none())
            .def("step", &NativeVectorEnv::Step,
                 py::arg("action"),
                 py::arg("env_ids") = py::none())
            .def("async_reset", &NativeVectorEnv::AsyncReset)
            .def("send", &NativeVectorEnv::Send,
                 py::arg("action"),
                 py::arg("env_ids") = py::none())
            .def("recv", &NativeVectorEnv::Recv);

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
