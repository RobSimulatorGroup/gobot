#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gobot {
// Backend-neutral, immutable presentation data. Positions are metres, forces
// newtons and torques N.m, in the physics world's frame, acting on `body`.
struct SimulationContactBody {
    std::string robot_name;
    std::string link_name;
    std::string shape_name;
};
struct SimulationContact {
    std::size_t environment{0};
    SimulationContactBody body, other_body;
    std::array<double, 3> position{}, normal{}, force{}, torque{};
    double normal_impulse{0};
    double distance{0};
};
struct SimulationContactPair {
    std::size_t environment{0};
    std::size_t geometry_a{0}, geometry_b{0};
    std::array<double, 3> position{}, normal{}, force_on_b{}, torque_on_b{};
    double distance{0};
};

inline bool IsFiniteContact(const SimulationContact& contact) {
    for (const auto* values : {&contact.position, &contact.normal, &contact.force, &contact.torque})
        for (double v : *values) if (!std::isfinite(v)) return false;
    double norm = 0;
    for (double v : contact.normal) norm += v * v;
    return !contact.body.robot_name.empty() && !contact.body.link_name.empty() &&
            std::abs(norm - 1.) < 1e-4 && std::isfinite(contact.normal_impulse) &&
            contact.normal_impulse >= 0 && std::isfinite(contact.distance);
}

// SDK adapters report pairs once, with normal A->B and force on B. The engine
// exposes one entry per articulated receiver, matching native PhysicsWorlds.
inline std::vector<SimulationContact> MakeSimulationContacts(
        const std::vector<SimulationContactBody>& geometries,
        const std::vector<SimulationContactPair>& pairs, double sample_dt) {
    if (!std::isfinite(sample_dt) || sample_dt <= 0)
        throw std::invalid_argument("Contact sample timestep must be finite and positive");
    std::vector<SimulationContact> result;
    for (const auto& pair : pairs) {
        if (pair.geometry_a >= geometries.size() || pair.geometry_b >= geometries.size())
            throw std::invalid_argument("Contact geometry index is out of range");
        const auto append = [&](const auto& body, const auto& other, double sign) {
            if (body.robot_name.empty()) return;
            SimulationContact contact;
            contact.environment = pair.environment;
            contact.body = body;
            contact.other_body = other;
            contact.position = pair.position;
            contact.distance = pair.distance;
            double normal_force = 0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                contact.normal[axis] = pair.normal[axis] * sign;
                contact.force[axis] = pair.force_on_b[axis] * sign;
                contact.torque[axis] = pair.torque_on_b[axis] * sign;
                normal_force += contact.force[axis] * contact.normal[axis];
            }
            contact.normal_impulse = std::max(0., normal_force) * sample_dt;
            if (!IsFiniteContact(contact)) throw std::invalid_argument("Invalid contact state");
            result.push_back(std::move(contact));
        };
        append(geometries[pair.geometry_a], geometries[pair.geometry_b], -1.);
        append(geometries[pair.geometry_b], geometries[pair.geometry_a], 1.);
    }
    return result;
}
} // namespace gobot
