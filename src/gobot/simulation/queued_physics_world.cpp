#include "gobot/simulation/queued_physics_world.hpp"

#include <iterator>
#include <utility>

namespace gobot {

QueuedPhysicsWorld::QueuedPhysicsWorld(PhysicsBackendType backend, const PhysicsWorld& preview,
        std::shared_ptr<const SimulationStateFrame> initial_frame)
    : backend_(backend), frame_(std::move(initial_frame)) {
    settings_ = preview.GetSettings();
    scene_snapshot_ = preview.GetSceneSnapshot();
    scene_topology_ = preview.GetSceneTopology();
}

void QueuedPhysicsWorld::SetBackendReady(PhysicsBackendCapabilities capabilities,
        std::optional<PhysicsSceneArtifact> artifact) {
    capabilities_ = capabilities;
    artifact_ = std::move(artifact);
    backend_ready_ = true;
}

PhysicsBackendCapabilities QueuedPhysicsWorld::GetCapabilities() const {
    auto result = capabilities_;
    result.environment_batch = false;
    result.masked_reset = false;
    // Checkpoints are requested at the scheduler boundary, not synchronously through this endpoint.
    result.runtime_checkpoint = false;
    return result;
}

PhysicsSolverDiagnostics QueuedPhysicsWorld::GetSolverDiagnostics() const {
    return frame_->step.diagnostics;
}

const PhysicsSceneState* QueuedPhysicsWorld::GetEnvironmentState(std::size_t index) const {
    return index == 0 && active_ ? &frame_->state : nullptr;
}

const PhysicsSceneArtifact* QueuedPhysicsWorld::GetSceneArtifact() const {
    return artifact_ ? &*artifact_ : nullptr;
}

void QueuedPhysicsWorld::Publish(std::shared_ptr<const SimulationStateFrame> frame) {
    if (active_ && frame) frame_ = std::move(frame);
}

bool QueuedPhysicsWorld::QueueCommands(PhysicsCommands commands) {
    if (!active_) {
        last_error_ = "Simulation command endpoint belongs to a retired session.";
        return false;
    }
    if (!command_error_.empty()) {
        last_error_ = command_error_;
        return false;
    }
    const auto bytes = PhysicsCommandsBytes(commands);
    if (commands.size() > MaxCommands - commands_.size() || bytes > MaxCommandBytes - command_bytes_) {
        command_error_ = last_error_ = "Simulation command queue capacity exceeded; reset is required.";
        return false;
    }
    if (!ValidatePhysicsCommands(scene_snapshot_, commands, &last_error_)) return false;
    command_bytes_ += bytes;
    commands_.insert(commands_.end(), std::make_move_iterator(commands.begin()), std::make_move_iterator(commands.end()));
    last_error_.clear();
    return true;
}

bool QueuedPhysicsWorld::TakeCommands(PhysicsCommands* commands) {
    if (!command_error_.empty() || !active_) {
        last_error_ = command_error_.empty() ? "Simulation session is retired." : command_error_;
        return false;
    }
    *commands = std::move(commands_);
    commands_.clear();
    command_bytes_ = 0;
    return true;
}

void QueuedPhysicsWorld::DiscardCommands() {
    commands_.clear();
    command_bytes_ = 0;
    command_error_.clear();
    last_error_.clear();
}

void QueuedPhysicsWorld::Retire() {
    DiscardCommands();
    active_ = false;
}

bool QueuedPhysicsWorld::Resolve(const std::string& robot, const std::string& member, bool joint,
        std::size_t* robot_index, std::size_t* member_index) {
    for (std::size_t r = 0; r < scene_snapshot_.robots.size(); ++r) {
        const auto& candidate = scene_snapshot_.robots[r];
        if (candidate.name != robot) continue;
        const std::size_t count = joint ? candidate.joints.size() : candidate.links.size();
        for (std::size_t m = 0; m < count; ++m) {
            if ((joint ? candidate.joints[m].name : candidate.links[m].name) == member) {
                *robot_index = r;
                *member_index = m;
                return true;
            }
        }
    }
    last_error_ = "Simulation command refers to a missing robot member: " + robot + "::" + member;
    return false;
}

bool QueuedPhysicsWorld::SetEnvironmentJointControl(std::size_t environment, const std::string& robot,
        const std::string& joint, PhysicsJointControlMode mode, RealType target) {
    PhysicsJointCommand command;
    if (environment != 0) return RejectSynchronousOperation();
    if (!Resolve(robot, joint, true, &command.robot, &command.joint)) return false;
    command.mode = mode;
    command.target = target;
    return QueueCommands({command});
}

bool QueuedPhysicsWorld::ResetJointState(const std::string& robot, const std::string& joint,
        RealType position, RealType velocity) {
    PhysicsJointResetCommand command;
    if (!Resolve(robot, joint, true, &command.robot, &command.joint)) return false;
    command.position = position;
    command.velocity = velocity;
    return QueueCommands({command});
}

bool QueuedPhysicsWorld::ResetEnvironmentJointState(std::size_t environment, const std::string& robot,
        const std::string& joint, RealType position, RealType velocity) {
    return environment == 0 ? ResetJointState(robot, joint, position, velocity) : RejectSynchronousOperation();
}

bool QueuedPhysicsWorld::ResetLinkState(const std::string& robot, const std::string& link,
        const Vector3& position, const Quaternion& orientation,
        const Vector3& linear_velocity, const Vector3& angular_velocity) {
    PhysicsLinkResetCommand command;
    if (!Resolve(robot, link, false, &command.robot, &command.link)) return false;
    command.position = position;
    command.orientation = orientation;
    command.linear_velocity = linear_velocity;
    command.angular_velocity = angular_velocity;
    return QueueCommands({command});
}

bool QueuedPhysicsWorld::ResetEnvironmentLinkState(std::size_t environment, const std::string& robot,
        const std::string& link, const Vector3& position, const Quaternion& orientation,
        const Vector3& linear_velocity, const Vector3& angular_velocity) {
    return environment == 0 ? ResetLinkState(robot, link, position, orientation, linear_velocity, angular_velocity)
                            : RejectSynchronousOperation();
}

bool QueuedPhysicsWorld::SetLinkExternalForce(const std::string& robot, const std::string& link,
        const Vector3& point, const Vector3& force) {
    PhysicsLinkForceCommand command;
    if (!Resolve(robot, link, false, &command.robot, &command.link)) return false;
    command.point = point;
    command.force = force;
    return QueueCommands({command});
}

bool QueuedPhysicsWorld::SetLinkSpringForce(const std::string& robot, const std::string& link,
        const Vector3& local_point, const Vector3& target, const Vector3& force) {
    PhysicsLinkForceCommand command;
    if (!Resolve(robot, link, false, &command.robot, &command.link)) return false;
    command.point = local_point;
    command.force = force;
    command.spring = true;
    command.target = target;
    return QueueCommands({command});
}

bool QueuedPhysicsWorld::SetDeformableExternalForces(std::uint64_t stable_id, const std::vector<Vector3>& forces) {
    return QueueCommands({PhysicsDeformableForceCommand{stable_id, forces}});
}

void QueuedPhysicsWorld::ClearExternalForces() { QueueCommands({PhysicsClearForcesCommand{false}}); }
void QueuedPhysicsWorld::ClearDeformableExternalForces() { QueueCommands({PhysicsClearForcesCommand{true}}); }

bool QueuedPhysicsWorld::RejectSynchronousOperation() {
    last_error_ = "This operation requires the SimulationServer asynchronous request API or a synchronous headless world.";
    return false;
}

bool QueuedPhysicsWorld::Build(PhysicsSceneSnapshot) { return RejectSynchronousOperation(); }
bool QueuedPhysicsWorld::RestoreCompatibleState(const PhysicsSceneState&) { return RejectSynchronousOperation(); }
void QueuedPhysicsWorld::Reset() { RejectSynchronousOperation(); }
bool QueuedPhysicsWorld::ResetEnvironment(std::size_t) { return RejectSynchronousOperation(); }
bool QueuedPhysicsWorld::WriteEnvironmentLinkVelocity(std::size_t, const std::string&, const std::string&,
        const Vector3&, const Vector3&) { return RejectSynchronousOperation(); }
PhysicsStepResult QueuedPhysicsWorld::Step(RealType) {
    RejectSynchronousOperation();
    return {.error = last_error_};
}
Ref<PhysicsRuntimeCheckpoint> QueuedPhysicsWorld::CaptureCheckpoint() const { return {}; }
bool QueuedPhysicsWorld::RestoreCheckpoint(const Ref<PhysicsRuntimeCheckpoint>&, const std::vector<std::size_t>&) {
    return RejectSynchronousOperation();
}

} // namespace gobot
