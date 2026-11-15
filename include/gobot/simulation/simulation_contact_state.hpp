#pragma once

#include <memory>
#include <unordered_map>
#include "gobot/simulation/simulation_data_runtime.hpp"
#include "gobot/simulation/simulation_state_frame.hpp"

namespace gobot {
// Convert a leased backend snapshot into scene-space physics state. Display
// transforms never modify solver data; only proper rigid transforms are valid.
inline std::shared_ptr<const SimulationStateFrame> CaptureContactStateFrame(
        const SimulationSnapshot& snapshot,
        const std::unordered_map<std::size_t, Affine3>& display_transforms = {}) {
    for (const auto& [environment, transform] : display_transforms) {
        if (!transform.matrix().allFinite() ||
            !(transform.linear().transpose() * transform.linear()).isApprox(
                    Eigen::Matrix<RealType, 3, 3>::Identity(), RealType(1e-5)) ||
            std::abs(transform.linear().determinant() - RealType(1)) > RealType(1e-5))
            throw std::invalid_argument("Contact presentation root must use a rigid transform");
    }
    auto frame = std::make_shared<SimulationStateFrame>();
    frame->epoch = snapshot.epoch;
    if (!snapshot.clocks.empty()) {
        frame->tick = snapshot.clocks.front().tick;
        frame->simulation_time = snapshot.clocks.front().time;
    }
    const auto vector = [](const auto& value) { return Vector3(value[0], value[1], value[2]); };
    for (const auto& input : snapshot.contacts) {
        Affine3 transform = Affine3::Identity();
        if (!display_transforms.empty()) {
            const auto found = display_transforms.find(input.environment);
            if (found == display_transforms.end()) continue;
            transform = found->second;
        }
        PhysicsContactState contact;
        contact.robot_name = input.body.robot_name;
        contact.link_name = input.body.link_name;
        contact.shape_name = input.body.shape_name;
        contact.other_robot_name = input.other_body.robot_name;
        contact.other_link_name = input.other_body.link_name;
        contact.other_shape_name = input.other_body.shape_name;
        contact.position = transform * vector(input.position);
        contact.normal = transform.linear() * vector(input.normal);
        contact.force = transform.linear() * vector(input.force);
        contact.torque = transform.linear() * vector(input.torque);
        contact.normal_force = std::max(RealType(0), contact.force.dot(contact.normal));
        contact.tangent_force = contact.force - contact.normal * contact.normal_force;
        contact.normal_impulse = input.normal_impulse;
        contact.distance = input.distance;
        frame->state.contacts.push_back(std::move(contact));
    }
    return frame;
}
} // namespace gobot
