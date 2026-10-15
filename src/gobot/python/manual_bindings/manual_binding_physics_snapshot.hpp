#pragma once

// Owned, data-only copies of the engine compiler output. No nodes or backend handles.
#include "manual_bindings_internal.hpp"

namespace gobot::python {

py::dict JointSnapshotToPythonDict(const PhysicsJointSnapshot& joint);
py::dict LinkSnapshotToPythonDict(const PhysicsLinkSnapshot& link);

inline py::dict PhysicsSnapshotMaterial(const PhysicsMaterialSnapshot& material) {
    py::dict result;
    result["friction"] = material.sliding_friction;
    result["restitution"] = material.restitution;
    result["compliance"] = material.contact_compliance;
    result["damping"] = material.contact_damping;
    return result;
}

template <class T>
py::dict PhysicsSnapshotIdentity(const T& value) {
    py::dict result;
    result["name"] = value.name;
    result["path"] = value.scene_path;
    result["stable_id"] = value.stable_id;
    result["transform"] = TransformToPythonDict(value.global_transform);
    return result;
}

inline py::list PhysicsSnapshotVertices(const std::vector<Vector3>& vertices) {
    py::list result;
    for (const auto& vertex : vertices) result.append(Vector3ToPython(vertex));
    return result;
}

inline py::dict PhysicsSnapshotShape(const PhysicsShapeSnapshot& shape) {
    auto result = PhysicsSnapshotIdentity(shape);
    result["type"] = static_cast<int>(shape.type);
    result["size"] = Vector3ToPython(shape.box_size);
    result["radius"] = shape.radius;
    result["height"] = shape.height;
    result["vertices"] = PhysicsSnapshotVertices(shape.vertices);
    result["indices"] = shape.indices;
    result["material"] = PhysicsSnapshotMaterial(shape.material);
    result["collision_layer"] = shape.collision_layer;
    result["collision_mask"] = shape.collision_mask;
    result["contact_offset"] = shape.contact_offset;
    result["rest_offset"] = shape.rest_offset;
    result["disabled"] = shape.disabled;
    return result;
}

inline py::dict PhysicsSnapshotToPython(const PhysicsSceneSnapshot& snapshot) {
    py::dict result;
    result["schema_version"] = 1;
    result["scene_name"] = snapshot.scene_name;
    py::list robots, deformables, shapes, terrains;
    for (const auto& robot : snapshot.robots) {
        auto value = PhysicsSnapshotIdentity(robot);
        value["standalone_rigid_body"] = robot.standalone_rigid_body;
        py::list links, joints;
        for (const auto& link : robot.links) {
            auto item = LinkSnapshotToPythonDict(link);
            item["path"] = link.scene_path;
            item["stable_id"] = link.stable_id;
            item["has_inertial"] = link.has_inertial;
            py::list collisions;
            for (const auto& shape : link.collision_shapes) collisions.append(PhysicsSnapshotShape(shape));
            item["shapes"] = collisions;
            links.append(item);
        }
        for (const auto& joint : robot.joints) {
            auto item = JointSnapshotToPythonDict(joint);
            item["type"] = joint.joint_type;
            item["path"] = joint.scene_path;
            item["stable_id"] = joint.stable_id;
            joints.append(item);
        }
        value["links"] = links;
        value["joints"] = joints;
        value["sensor_count"] = robot.sensors.size();
        robots.append(value);
    }
    for (const auto& body : snapshot.deformables) {
        auto value = PhysicsSnapshotIdentity(body);
        value["vertices"] = PhysicsSnapshotVertices(body.vertices);
        value["tetrahedra"] = body.tetrahedra;
        value["triangles"] = body.surface_triangles;
        value["model"] = body.model;
        value["density"] = body.density;
        value["young_modulus"] = body.young_modulus;
        value["poisson_ratio"] = body.poisson_ratio;
        value["damping"] = body.damping;
        value["thickness"] = body.thickness;
        value["bending_stiffness"] = body.bending_stiffness;
        value["material"] = PhysicsSnapshotMaterial(body.material);
        value["kinematic"] = body.kinematic;
        value["collision_layer"] = body.collision_layer;
        value["collision_mask"] = body.collision_mask;
        value["self_collision_enabled"] = body.self_collision_enabled;
        deformables.append(value);
    }
    for (const auto& shape : snapshot.loose_collision_shapes) shapes.append(PhysicsSnapshotShape(shape));
    for (const auto& terrain : snapshot.terrains) {
        py::dict value;
        value["name"] = terrain.name;
        value["path"] = terrain.scene_path;
        value["stable_id"] = terrain.stable_id;
        py::list boxes, meshes, heights;
        for (const auto& box : terrain.boxes) {
            py::dict item;
            item["transform"] = TransformToPythonDict(box.global_transform);
            item["size"] = Vector3ToPython(box.size);
            boxes.append(item);
        }
        for (const auto& mesh : terrain.mesh_patches) {
            py::dict item;
            item["transform"] = TransformToPythonDict(mesh.global_transform);
            item["vertices"] = PhysicsSnapshotVertices(mesh.vertices);
            item["indices"] = mesh.indices;
            meshes.append(item);
        }
        for (const auto& height : terrain.heightfields) {
            py::dict item;
            item["transform"] = TransformToPythonDict(height.global_transform);
            item["rows"] = height.rows;
            item["cols"] = height.cols;
            item["heights"] = height.heights;
            item["size"] = py::make_tuple(height.size.x(), height.size.y());
            heights.append(item);
        }
        value["boxes"] = boxes;
        value["meshes"] = meshes;
        value["heightfields"] = heights;
        value["material"] = PhysicsSnapshotMaterial(terrain.material);
        value["collision_layer"] = terrain.collision_layer;
        value["collision_mask"] = terrain.collision_mask;
        value["contact_offset"] = terrain.contact_offset;
        value["rest_offset"] = terrain.rest_offset;
        terrains.append(value);
    }
    result["robots"] = robots;
    result["deformables"] = deformables;
    result["loose_shapes"] = shapes;
    result["terrains"] = terrains;
    // Consumers must reject unsupported features, rather than silently omit them.
    result["coupling_count"] = snapshot.couplings.size();
    result["attachment_count"] = snapshot.deformable_attachments.size();
    result["sensor_count"] = snapshot.total_sensor_count;
    return result;
}
} // namespace gobot::python
