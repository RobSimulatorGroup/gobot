#pragma once

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
#include <pybind11/numpy.h>
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
#include "gobot/rendering/headless_render_context.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
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
#include "gobot/scene/velocity_command_debug_3d.hpp"
#include "gobot/simulation/simulation_scene.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {

enum class PyNodeOwnership {
    Borrowed,
    DetachedOwned,
    TreeOwned
};

struct PyNodeHandleState {
    ObjectID id;
    EngineContext* context{nullptr};
    std::uint64_t scene_epoch{0};
    std::string name_snapshot;
    PyNodeOwnership ownership{PyNodeOwnership::Borrowed};
};

struct PyNodeHandle {
    std::shared_ptr<PyNodeHandleState> state;
    std::string expected_type;

    static std::shared_ptr<PyNodeHandleState> MakeState(Node* node,
                                                        EngineContext* context,
                                                        std::uint64_t epoch,
                                                        PyNodeOwnership node_ownership) {
        auto state = std::make_shared<PyNodeHandleState>();
        state->id = node != nullptr ? node->GetInstanceId() : ObjectID();
        state->context = context;
        state->scene_epoch = epoch;
        state->name_snapshot = node != nullptr ? node->GetName() : std::string();
        state->ownership = node_ownership;
        return state;
    }

    PyNodeHandle() = default;

    PyNodeHandle(Node* node,
                 std::string expected_type_name,
                 EngineContext* context,
                 std::uint64_t epoch,
                 PyNodeOwnership node_ownership)
        : state(MakeState(node, context, epoch, node_ownership)),
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

struct PyRayCastSensor3DHandle : public PySensor3DHandle {
    using PySensor3DHandle::PySensor3DHandle;
};

struct PyTerrainHeightSensor3DHandle : public PyRayCastSensor3DHandle {
    using PyRayCastSensor3DHandle::PyRayCastSensor3DHandle;
};

struct PyHeightScanner3DHandle : public PyTerrainHeightSensor3DHandle {
    using PyTerrainHeightSensor3DHandle::PyTerrainHeightSensor3DHandle;
};

struct PyVelocityCommandDebug3DHandle : public PyNode3DHandle {
    using PyNode3DHandle::PyNode3DHandle;
};

std::uint64_t ActiveSceneEpoch();

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

using PyNodeClass = py::class_<PyNodeHandle>;
using PyNode3DClass = py::class_<PyNode3DHandle, PyNodeHandle>;
using PyRobot3DClass = py::class_<PyRobot3DHandle, PyNode3DHandle>;
using PyLink3DClass = py::class_<PyLink3DHandle, PyNode3DHandle>;
using PyJoint3DClass = py::class_<PyJoint3DHandle, PyNode3DHandle>;
using PyCollisionShape3DClass = py::class_<PyCollisionShape3DHandle, PyNode3DHandle>;
using PyMeshInstance3DClass = py::class_<PyMeshInstance3DHandle, PyNode3DHandle>;
using PyTerrain3DClass = py::class_<PyTerrain3DHandle, PyNode3DHandle>;
using PySensor3DClass = py::class_<PySensor3DHandle, PyNode3DHandle>;
using PyIMUSensor3DClass = py::class_<PyIMUSensor3DHandle, PySensor3DHandle>;
using PyAngularMomentumSensor3DClass = py::class_<PyAngularMomentumSensor3DHandle, PySensor3DHandle>;
using PyContactSensor3DClass = py::class_<PyContactSensor3DHandle, PySensor3DHandle>;
using PyRayCastSensor3DClass = py::class_<PyRayCastSensor3DHandle, PySensor3DHandle>;
using PyTerrainHeightSensor3DClass = py::class_<PyTerrainHeightSensor3DHandle, PyRayCastSensor3DHandle>;
using PyHeightScanner3DClass = py::class_<PyHeightScanner3DHandle, PyTerrainHeightSensor3DHandle>;
using PyVelocityCommandDebug3DClass = py::class_<PyVelocityCommandDebug3DHandle, PyNode3DHandle>;

EngineContext& EnsureRuntimeContext();
std::uint64_t ActiveSceneEpoch();
EngineContext* ActiveSceneContext();
Node* ActiveSceneRoot();
std::uint64_t SceneEpochForContext(EngineContext* context);
Node* SceneRootForContext(EngineContext& context);
bool IsSceneScriptRuntimeMutation();
EngineContext* ResolveHandleContext(EngineContext* context);
EngineContext* ContextForNode(const Node* node);
void ExecuteSetNodeProperty(Node* node, const std::string& property_name, Variant value);

std::string ExpectedTypeForNode(Node* node);
PyNodeHandle MakeTypedNodeHandle(Node* node,
                                 PyNodeOwnership ownership = PyNodeOwnership::Borrowed,
                                 EngineContext* context = nullptr,
                                 std::uint64_t epoch = 0);
py::object MakeTypedNodeObject(Node* node,
                               PyNodeOwnership ownership = PyNodeOwnership::Borrowed,
                               EngineContext* context = nullptr,
                               std::uint64_t epoch = 0);
PyCollisionShape3DHandle MakeCollisionShape3DHandle(CollisionShape3D* node,
                                                    PyNodeOwnership ownership = PyNodeOwnership::Borrowed,
                                                    EngineContext* context = nullptr,
                                                    std::uint64_t epoch = 0);
PyMeshInstance3DHandle MakeMeshInstance3DHandle(MeshInstance3D* node,
                                                PyNodeOwnership ownership = PyNodeOwnership::Borrowed,
                                                EngineContext* context = nullptr,
                                                std::uint64_t epoch = 0);

template <typename Func>
void ExecuteSceneMutation(const std::string&, Func&& func) {
    std::unique_ptr<SceneCommand> command = std::forward<Func>(func)();
    if (!GetActiveAppContext().ExecuteSceneCommand(std::move(command))) {
        throw std::runtime_error("failed to execute Gobot scene command");
    }
}

Robot3D* CreateTestRobotScene();
Node* CreateNode(const std::string& type_name, const std::string& name);
CollisionShape3D* CreateBoxCollision(const std::string& name,
                                      const Vector3& size,
                                      const Vector3& position = Vector3::Zero());
MeshInstance3D* CreateBoxVisual(const std::string& name,
                                const Vector3& size,
                                const Vector3& position = Vector3::Zero());
Node* LoadSceneRoot(const std::string& scene_path);
bool SaveSceneRoot(Node* root, const std::string& path);
void ImportMJCFScene(const std::string& xml_path,
                     const std::string& scene_path,
                     const std::optional<std::string>& name,
                     const std::optional<std::string>& script_path);
PhysicsBackendType ParseBackend(const std::string& backend);

std::string NodeTypeName(const PyNodeHandle& handle);
py::list GetNodeChildren(PyNodeHandle& handle);
py::object GetNodeChild(PyNodeHandle& handle, int index);
py::object FindNode(PyNodeHandle& handle, const std::string& path);
py::object NodeGetProperty(const PyNodeHandle& handle, const std::string& name);
void NodeSetProperty(PyNodeHandle& handle, const std::string& name, const py::handle& value);
py::object NodeAddChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool force_readable_name);
py::object NodeRemoveChild(PyNodeHandle& handle, PyNodeHandle& child_handle, bool delete_child);
void NodeReparent(PyNodeHandle& handle, PyNodeHandle& parent_handle);
py::object NodeGetParent(PyNodeHandle& handle);
py::list NodeGetPropertyNames(const PyNodeHandle& handle);
std::string NodeGetName(const PyNodeHandle& handle);
void NodeSetName(PyNodeHandle& handle, const std::string& name);
std::string NodeGetPath(const PyNodeHandle& handle);
std::uint64_t NodeGetId(const PyNodeHandle& handle);
bool NodeIsValid(const PyNodeHandle& handle);
py::dict NodeToDict(const PyNodeHandle& handle);
py::dict ResourceToPythonDict(const Ref<Resource>& resource);
py::dict TransformToPythonDict(const Affine3& transform);

Quaternion PythonToQuaternionWxyz(const py::handle& object);
Vector2 PythonToVector2(const py::handle& object);
py::tuple Vector2ToPython(const Vector2& value);
Color PythonToColor4(const py::handle& object);
py::tuple ColorToPython(const Color& color);
std::vector<Vector3> PythonToVector3List(const py::handle& object);
py::list Vector3ListToPython(const std::vector<Vector3>& values);
std::vector<DebugArrow> PythonToDebugArrows(const py::handle& object);
Affine3 PythonToTransformWxyz(const py::handle& position, const py::handle& orientation);
py::array_t<std::uint8_t> CaptureRgb(const py::handle& root_handle,
                                     int width,
                                     int height,
                                     const py::handle& eye,
                                     const py::handle& target,
                                     const py::handle& up,
                                     RealType fov_y,
                                     RealType z_near,
                                     RealType z_far,
                                     const py::handle& debug_arrows);

py::dict RuntimeStateToPythonDict(const PhysicsSceneState& state);
const PhysicsLinkState* FindLinkState(const PhysicsRobotState& robot, const std::string& link_name);
const PhysicsJointState* FindJointState(const PhysicsRobotState& robot, const std::string& joint_name);
const PhysicsSensorState* FindSensorState(const PhysicsRobotState& robot, const std::string& sensor_name);
SimulationScene* RuntimeSceneForRobotHandle(const PyRobot3DHandle& handle);
const PhysicsRobotSnapshot& RequiredRobotSnapshotForHandle(const PyRobot3DHandle& handle);
const PhysicsRobotState& RequiredRobotStateForHandle(const PyRobot3DHandle& handle);
const PhysicsSceneState& RequiredSceneStateForHandle(const PyRobot3DHandle& handle);
py::dict RobotSnapshotToPythonDict(const PhysicsRobotSnapshot& robot);
py::dict RobotStateToPythonDict(const PhysicsRobotState& robot, const PhysicsSceneState* scene_state = nullptr);
py::dict JointStateToPythonDict(const PhysicsJointState& joint);
py::dict LinkStateToPythonDict(const PhysicsLinkState& link);
py::dict SensorStateToPythonDict(const PhysicsSensorState& sensor);
py::dict BatchRobotStateToPythonDict(SimulationServer& simulation,
                                     const std::string& robot_name,
                                     const std::string& base_link,
                                     const std::vector<std::string>& joint_names,
                                     const std::vector<std::string>& link_names,
                                     const std::vector<std::string>& sensor_names);

void RegisterManualCommonBindings(py::module_& module);
void RegisterManualAppContextBindings(py::module_& module);
void RegisterManualNodeBindings(PyNodeClass& node_class, PyNode3DClass& node3d_class);
void RegisterManualRobotBindings(PyRobot3DClass& robot3d_class,
                                 PyLink3DClass& link3d_class,
                                 PyJoint3DClass& joint3d_class,
                                 PyCollisionShape3DClass& collision_shape_class);
void RegisterManualTerrainSensorBindings(PyTerrain3DClass& terrain3d_class,
                                         PySensor3DClass& sensor3d_class,
                                         PyIMUSensor3DClass& imu_sensor3d_class,
                                         PyAngularMomentumSensor3DClass& angular_momentum_sensor3d_class,
                                         PyContactSensor3DClass& contact_sensor3d_class,
                                         PyRayCastSensor3DClass& raycast_sensor3d_class,
                                         PyTerrainHeightSensor3DClass& terrain_height_sensor3d_class,
                                         PyHeightScanner3DClass& height_scanner3d_class,
                                         PyMeshInstance3DClass& mesh_instance_class);
void RegisterManualModuleFunctions(py::module_& module);

} // namespace gobot::python
