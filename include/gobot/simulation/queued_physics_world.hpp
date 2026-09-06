#pragma once

#include <optional>
#include "gobot/physics/physics_commands.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/simulation/simulation_state_frame.hpp"

namespace gobot {

// Main-thread endpoint. No backend pointers or worker-owned mutable state escape here.
class GOBOT_EXPORT QueuedPhysicsWorld final : public PhysicsWorld {
    GOBCLASS(QueuedPhysicsWorld, PhysicsWorld)
public:
    explicit QueuedPhysicsWorld(PhysicsBackendType backend, const PhysicsWorld& preview,
            std::shared_ptr<const SimulationStateFrame> initial_frame);
    PhysicsBackendType GetBackendType() const override { return backend_; }
    bool IsAvailable() const override { return active_; }
    const std::string& GetLastError() const override { return last_error_; }
    PhysicsBackendCapabilities GetCapabilities() const override;
    PhysicsSolverDiagnostics GetSolverDiagnostics() const override;
    const PhysicsSceneState& GetSceneState() const override { return frame_->state; }
    const PhysicsSceneState* GetEnvironmentState(std::size_t index) const override;
    const PhysicsSceneArtifact* GetSceneArtifact() const override;

    void Publish(std::shared_ptr<const SimulationStateFrame> frame);
    void SetBackendReady(PhysicsBackendCapabilities capabilities, std::optional<PhysicsSceneArtifact> artifact);
    bool IsBackendReady() const { return active_ && backend_ready_; }
    std::shared_ptr<const SimulationStateFrame> GetFrame() const { return frame_; }
    bool QueueCommands(PhysicsCommands commands);
    bool TakeCommands(PhysicsCommands* commands);
    void DiscardCommands();
    void Retire();
    std::size_t GetQueuedCommandCount() const { return commands_.size(); }
    static constexpr std::size_t MaxCommands = 16384;
    static constexpr std::size_t MaxCommandBytes = 64 * 1024 * 1024;

    bool SetEnvironmentJointControl(std::size_t environment, const std::string& robot,
            const std::string& joint, PhysicsJointControlMode mode, RealType target) override;
    bool SetJointControl(const std::string& robot, const std::string& joint,
            PhysicsJointControlMode mode, RealType target) override {
        return SetEnvironmentJointControl(0, robot, joint, mode, target);
    }
    bool ResetJointState(const std::string& robot, const std::string& joint,
            RealType position, RealType velocity) override;
    bool ResetEnvironmentJointState(std::size_t environment, const std::string& robot,
            const std::string& joint, RealType position, RealType velocity) override;
    bool ResetLinkState(const std::string& robot, const std::string& link, const Vector3& position,
            const Quaternion& orientation, const Vector3& linear_velocity, const Vector3& angular_velocity) override;
    bool ResetEnvironmentLinkState(std::size_t environment, const std::string& robot,
            const std::string& link, const Vector3& position, const Quaternion& orientation,
            const Vector3& linear_velocity, const Vector3& angular_velocity) override;
    bool SetLinkExternalForce(const std::string& robot, const std::string& link,
            const Vector3& point, const Vector3& force) override;
    bool SetLinkSpringForce(const std::string& robot, const std::string& link,
            const Vector3& local_point, const Vector3& target, const Vector3& force) override;
    bool SetDeformableExternalForces(std::uint64_t stable_id, const std::vector<Vector3>& forces) override;
    void ClearExternalForces() override;
    void ClearDeformableExternalForces() override;

    bool Build(PhysicsSceneSnapshot) override;
    bool RestoreCompatibleState(const PhysicsSceneState&) override;
    void Reset() override;
    bool ResetEnvironment(std::size_t) override;
    bool WriteEnvironmentLinkVelocity(std::size_t, const std::string&, const std::string&,
            const Vector3&, const Vector3&) override;
    PhysicsStepResult Step(RealType) override;
    Ref<PhysicsRuntimeCheckpoint> CaptureCheckpoint() const override;
    bool RestoreCheckpoint(const Ref<PhysicsRuntimeCheckpoint>&, const std::vector<std::size_t>&) override;

private:
    bool Resolve(const std::string& robot, const std::string& member, bool joint,
            std::size_t* robot_index, std::size_t* member_index);
    bool RejectSynchronousOperation();
    PhysicsBackendType backend_;
    PhysicsBackendCapabilities capabilities_;
    std::optional<PhysicsSceneArtifact> artifact_;
    std::shared_ptr<const SimulationStateFrame> frame_;
    PhysicsCommands commands_;
    std::size_t command_bytes_ = 0;
    std::string command_error_;
    bool active_ = true;
    bool backend_ready_ = false;
};

} // namespace gobot
