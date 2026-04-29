/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/resource_format_urdf.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {
namespace {

struct VisualImportData {
    Vector3 origin_xyz{Vector3::Zero()};
    Vector3 origin_rpy{Vector3::Zero()};
    std::string mesh_path;
};

enum class CollisionGeometryType {
    None,
    Mesh,
    Box,
    Sphere,
    Cylinder,
};

struct CollisionImportData {
    Vector3 origin_xyz{Vector3::Zero()};
    Vector3 origin_rpy{Vector3::Zero()};
    CollisionGeometryType geometry_type{CollisionGeometryType::None};
    std::string mesh_path;
    Vector3 box_size{Vector3::Ones()};
    RealType radius{0.5};
    RealType height{1.0};
};

struct LinkImportData {
    std::string name;
    RealType mass{0.0};
    Vector3 center_of_mass{Vector3::Zero()};
    Vector3 inertia_diagonal{Vector3::Zero()};
    Vector3 inertia_off_diagonal{Vector3::Zero()};
    std::vector<VisualImportData> visuals;
    std::vector<CollisionImportData> collisions;
};

struct JointImportData {
    std::string name;
    JointType type{JointType::Fixed};
    std::string parent_link;
    std::string child_link;
    Vector3 origin_xyz{Vector3::Zero()};
    Vector3 origin_rpy{Vector3::Zero()};
    Vector3 axis{Vector3::UnitX()};
    RealType lower_limit{0.0};
    RealType upper_limit{0.0};
    RealType effort_limit{0.0};
    RealType velocity_limit{0.0};
};

std::string ResolveInputPath(const std::string& path) {
    if (path.starts_with("res://")) {
        ProjectSettings* settings = ProjectSettings::s_singleton;
        return settings != nullptr ? settings->GlobalizePath(path) : path.substr(6);
    }
    return path;
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open URDF file: {}.", path);
        return {};
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string GetAttribute(const std::string& attributes, const std::string& name) {
    const std::regex attr_regex(name + R"(\s*=\s*["']([^"']*)["'])", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(attributes, match, attr_regex)) {
        return {};
    }
    return match[1].str();
}

RealType ParseReal(const std::string& value, RealType fallback = 0.0) {
    if (value.empty()) {
        return fallback;
    }

    try {
        return static_cast<RealType>(std::stod(value));
    } catch (const std::exception&) {
        return fallback;
    }
}

Vector3 ParseVector3(const std::string& value, const Vector3& fallback = Vector3::Zero()) {
    if (value.empty()) {
        return fallback;
    }

    std::istringstream stream(value);
    RealType x{};
    RealType y{};
    RealType z{};
    if (!(stream >> x >> y >> z)) {
        return fallback;
    }
    return {x, y, z};
}

std::string FindTagAttributes(const std::string& body, const std::string& tag_name) {
    const std::regex tag_regex("<\\s*" + tag_name + R"(\b([^>]*)>)", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(body, match, tag_regex)) {
        return {};
    }
    return match[1].str();
}

std::string FindTagBody(const std::string& body, const std::string& tag_name) {
    const std::regex tag_regex("<\\s*" + tag_name + R"(\b[^>]*>([\s\S]*?)<\s*/\s*)" + tag_name + R"(\s*>)",
                               std::regex::icase);
    std::smatch match;
    if (!std::regex_search(body, match, tag_regex)) {
        return {};
    }
    return match[1].str();
}

std::vector<std::string> FindTagBodies(const std::string& body, const std::string& tag_name) {
    std::vector<std::string> bodies;
    const std::regex tag_regex("<\\s*" + tag_name + R"(\b[^>]*>([\s\S]*?)<\s*/\s*)" + tag_name + R"(\s*>)",
                               std::regex::icase);

    for (auto iter = std::sregex_iterator(body.begin(), body.end(), tag_regex);
         iter != std::sregex_iterator();
         ++iter) {
        bodies.push_back((*iter)[1].str());
    }

    return bodies;
}

std::string ReadPackageName(const std::filesystem::path& package_xml_path) {
    const std::string package_xml = ReadTextFile(package_xml_path.string());
    std::smatch match;
    if (!std::regex_search(package_xml, match, std::regex(R"(<\s*name\s*>\s*([^<]+?)\s*<\s*/\s*name\s*>)",
                                                          std::regex::icase))) {
        return {};
    }
    return match[1].str();
}

std::filesystem::path FindPackageRoot(const std::filesystem::path& source_file_path,
                                      const std::string& package_name) {
    std::filesystem::path current = source_file_path.parent_path();
    while (!current.empty() && current != current.root_path()) {
        const std::filesystem::path package_xml = current / "package.xml";
        if (std::filesystem::exists(package_xml) && ReadPackageName(package_xml) == package_name) {
            return current;
        }
        current = current.parent_path();
    }

    return {};
}

std::string LocalizeImportedAssetPath(const std::filesystem::path& asset_path) {
    const std::filesystem::path normalized_path = asset_path.lexically_normal();
    ProjectSettings* settings = ProjectSettings::s_singleton;
    return settings != nullptr ? settings->LocalizePath(normalized_path.string()) : normalized_path.string();
}

std::string ResolvePackageUri(const std::string& filename, const std::string& source_file_path) {
    constexpr std::string_view prefix = "package://";
    if (!filename.starts_with(prefix)) {
        return {};
    }

    const std::string package_relative = filename.substr(prefix.size());
    const std::size_t slash = package_relative.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= package_relative.size()) {
        return filename;
    }

    const std::string package_name = package_relative.substr(0, slash);
    const std::string relative_asset_path = package_relative.substr(slash + 1);
    const std::filesystem::path package_root = FindPackageRoot(source_file_path, package_name);
    if (package_root.empty()) {
        return filename;
    }

    return LocalizeImportedAssetPath(package_root / relative_asset_path);
}

std::string NormalizeImportedAssetPath(const std::string& filename, const std::string& source_file_path) {
    if (filename.empty() || filename.starts_with("model://") || filename.starts_with("res://")) {
        return filename;
    }

    if (filename.starts_with("package://")) {
        return ResolvePackageUri(filename, source_file_path);
    }

    std::filesystem::path mesh_path(filename);
    if (mesh_path.is_relative()) {
        mesh_path = std::filesystem::path(source_file_path).parent_path() / mesh_path;
    }

    return LocalizeImportedAssetPath(mesh_path);
}

JointType ParseJointType(const std::string& value) {
    if (value == "revolute") {
        return JointType::Revolute;
    }
    if (value == "continuous") {
        return JointType::Continuous;
    }
    if (value == "prismatic") {
        return JointType::Prismatic;
    }
    if (value == "floating") {
        return JointType::Floating;
    }
    if (value == "planar") {
        return JointType::Planar;
    }
    return JointType::Fixed;
}

void ParseOrigin(const std::string& body, Vector3& origin_xyz, Vector3& origin_rpy) {
    const std::string origin_attrs = FindTagAttributes(body, "origin");
    origin_xyz = ParseVector3(GetAttribute(origin_attrs, "xyz"));
    origin_rpy = ParseVector3(GetAttribute(origin_attrs, "rpy"));
}

std::vector<VisualImportData> ParseVisuals(const std::string& link_body, const std::string& source_file_path) {
    std::vector<VisualImportData> visuals;
    for (const std::string& visual_body : FindTagBodies(link_body, "visual")) {
        VisualImportData visual;
        ParseOrigin(visual_body, visual.origin_xyz, visual.origin_rpy);

        const std::string mesh_attrs = FindTagAttributes(visual_body, "mesh");
        visual.mesh_path = NormalizeImportedAssetPath(GetAttribute(mesh_attrs, "filename"), source_file_path);
        if (!visual.mesh_path.empty()) {
            visuals.push_back(std::move(visual));
        }
    }

    return visuals;
}

std::vector<CollisionImportData> ParseCollisions(const std::string& link_body, const std::string& source_file_path) {
    std::vector<CollisionImportData> collisions;
    for (const std::string& collision_body : FindTagBodies(link_body, "collision")) {
        CollisionImportData collision;
        ParseOrigin(collision_body, collision.origin_xyz, collision.origin_rpy);

        const std::string mesh_attrs = FindTagAttributes(collision_body, "mesh");
        const std::string box_attrs = FindTagAttributes(collision_body, "box");
        const std::string sphere_attrs = FindTagAttributes(collision_body, "sphere");
        const std::string cylinder_attrs = FindTagAttributes(collision_body, "cylinder");

        if (!mesh_attrs.empty()) {
            collision.geometry_type = CollisionGeometryType::Mesh;
            collision.mesh_path = NormalizeImportedAssetPath(GetAttribute(mesh_attrs, "filename"), source_file_path);
        } else if (!box_attrs.empty()) {
            collision.geometry_type = CollisionGeometryType::Box;
            collision.box_size = ParseVector3(GetAttribute(box_attrs, "size"), Vector3::Ones());
        } else if (!sphere_attrs.empty()) {
            collision.geometry_type = CollisionGeometryType::Sphere;
            collision.radius = ParseReal(GetAttribute(sphere_attrs, "radius"), 0.5);
        } else if (!cylinder_attrs.empty()) {
            collision.geometry_type = CollisionGeometryType::Cylinder;
            collision.radius = ParseReal(GetAttribute(cylinder_attrs, "radius"), 0.5);
            collision.height = ParseReal(GetAttribute(cylinder_attrs, "length"), 1.0);
        }

        if (collision.geometry_type != CollisionGeometryType::None) {
            collisions.push_back(std::move(collision));
        }
    }

    return collisions;
}

std::vector<LinkImportData> ParseLinks(const std::string& xml, const std::string& source_file_path) {
    std::vector<LinkImportData> links;
    const std::regex link_regex(R"(<\s*link\b([^>]*)>([\s\S]*?)<\s*/\s*link\s*>|<\s*link\b([^>]*)/\s*>)",
                                std::regex::icase);

    for (auto iter = std::sregex_iterator(xml.begin(), xml.end(), link_regex);
         iter != std::sregex_iterator();
         ++iter) {
        const std::smatch& match = *iter;
        const std::string attributes = match[1].matched ? match[1].str() : match[3].str();
        const std::string body = match[2].matched ? match[2].str() : std::string();

        LinkImportData link;
        link.name = GetAttribute(attributes, "name");
        if (link.name.empty()) {
            continue;
        }

        const std::string inertial_body = FindTagBody(body, "inertial");
        if (!inertial_body.empty()) {
            link.center_of_mass = ParseVector3(GetAttribute(FindTagAttributes(inertial_body, "origin"), "xyz"));
            link.mass = ParseReal(GetAttribute(FindTagAttributes(inertial_body, "mass"), "value"));

            const std::string inertia_attrs = FindTagAttributes(inertial_body, "inertia");
            link.inertia_diagonal = {
                    ParseReal(GetAttribute(inertia_attrs, "ixx")),
                    ParseReal(GetAttribute(inertia_attrs, "iyy")),
                    ParseReal(GetAttribute(inertia_attrs, "izz"))};
            link.inertia_off_diagonal = {
                    ParseReal(GetAttribute(inertia_attrs, "ixy")),
                    ParseReal(GetAttribute(inertia_attrs, "ixz")),
                    ParseReal(GetAttribute(inertia_attrs, "iyz"))};
        }

        link.visuals = ParseVisuals(body, source_file_path);
        link.collisions = ParseCollisions(body, source_file_path);

        links.push_back(link);
    }

    return links;
}

std::vector<JointImportData> ParseJoints(const std::string& xml) {
    std::vector<JointImportData> joints;
    const std::regex joint_regex(R"(<\s*joint\b([^>]*)>([\s\S]*?)<\s*/\s*joint\s*>)",
                                 std::regex::icase);

    for (auto iter = std::sregex_iterator(xml.begin(), xml.end(), joint_regex);
         iter != std::sregex_iterator();
         ++iter) {
        const std::smatch& match = *iter;
        const std::string attributes = match[1].str();
        const std::string body = match[2].str();

        JointImportData joint;
        joint.name = GetAttribute(attributes, "name");
        joint.type = ParseJointType(GetAttribute(attributes, "type"));
        joint.parent_link = GetAttribute(FindTagAttributes(body, "parent"), "link");
        joint.child_link = GetAttribute(FindTagAttributes(body, "child"), "link");
        joint.axis = ParseVector3(GetAttribute(FindTagAttributes(body, "axis"), "xyz"), Vector3::UnitX());
        if (joint.axis.squaredNorm() <= CMP_EPSILON2) {
            joint.axis = Vector3::UnitX();
        }

        const std::string origin_attrs = FindTagAttributes(body, "origin");
        joint.origin_xyz = ParseVector3(GetAttribute(origin_attrs, "xyz"));
        joint.origin_rpy = ParseVector3(GetAttribute(origin_attrs, "rpy"));

        const std::string limit_attrs = FindTagAttributes(body, "limit");
        joint.lower_limit = ParseReal(GetAttribute(limit_attrs, "lower"));
        joint.upper_limit = ParseReal(GetAttribute(limit_attrs, "upper"));
        joint.effort_limit = ParseReal(GetAttribute(limit_attrs, "effort"));
        joint.velocity_limit = ParseReal(GetAttribute(limit_attrs, "velocity"));

        if (!joint.name.empty() && !joint.parent_link.empty() && !joint.child_link.empty()) {
            joints.push_back(joint);
        }
    }

    return joints;
}

template <typename T>
void AddProperty(SceneState::NodeData& node_data, std::string name, T value) {
    node_data.properties.push_back({std::move(name), Variant(std::move(value))});
}

void AddTransformProperties(SceneState::NodeData& node_data, const Vector3& origin_xyz, const Vector3& origin_rpy) {
    AddProperty(node_data, "position", origin_xyz);
    AddProperty(node_data, "rotation_degrees", Vector3{
            RAD_TO_DEG(origin_rpy.x()),
            RAD_TO_DEG(origin_rpy.y()),
            RAD_TO_DEG(origin_rpy.z())});
}

Ref<Mesh> MakeMeshReference(const std::string& mesh_path) {
    if (mesh_path.empty()) {
        return {};
    }

    if (ResourceCache::Has(mesh_path)) {
        Ref<Mesh> cached_mesh = dynamic_pointer_cast<Mesh>(ResourceCache::GetRef(mesh_path));
        if (cached_mesh.IsValid()) {
            return cached_mesh;
        }
    }

    if (RenderServer::HasInstance() && ResourceLoader::Exists(mesh_path, "Mesh")) {
        Ref<Resource> loaded_resource = ResourceLoader::Load(mesh_path, "Mesh", ResourceFormatLoader::CacheMode::Reuse);
        Ref<Mesh> loaded_mesh = dynamic_pointer_cast<Mesh>(loaded_resource);
        if (loaded_mesh.IsValid()) {
            return loaded_mesh;
        }
    }

    Ref<Mesh> mesh = MakeRef<Mesh>();
    mesh->SetPath(mesh_path);
    return mesh;
}

SceneState::NodeData MakeRobotNode(const std::string& name, const std::string& source_path) {
    SceneState::NodeData node_data;
    node_data.type = "Robot3D";
    node_data.name = name.empty() ? "Robot" : name;
    AddProperty(node_data, "source_path", source_path);
    return node_data;
}

SceneState::NodeData MakeLinkNode(const LinkImportData& link, int parent) {
    SceneState::NodeData node_data;
    node_data.type = "Link3D";
    node_data.name = link.name;
    node_data.parent = parent;
    AddProperty(node_data, "mass", link.mass);
    AddProperty(node_data, "center_of_mass", link.center_of_mass);
    AddProperty(node_data, "inertia_diagonal", link.inertia_diagonal);
    AddProperty(node_data, "inertia_off_diagonal", link.inertia_off_diagonal);
    return node_data;
}

SceneState::NodeData MakeVisualNode(const LinkImportData& link,
                                    const VisualImportData& visual,
                                    int visual_index,
                                    int parent) {
    SceneState::NodeData node_data;
    node_data.type = "MeshInstance3D";
    node_data.name = visual_index == 0
                     ? link.name + "_visual"
                     : link.name + "_visual_" + std::to_string(visual_index + 1);
    node_data.parent = parent;
    AddTransformProperties(node_data, visual.origin_xyz, visual.origin_rpy);
    AddProperty(node_data, "mesh", MakeMeshReference(visual.mesh_path));
    return node_data;
}

SceneState::NodeData MakeCollisionNode(const LinkImportData& link,
                                       const CollisionImportData& collision,
                                       int collision_index,
                                       int parent) {
    SceneState::NodeData node_data;
    node_data.type = "CollisionShape3D";
    node_data.name = collision_index == 0
                     ? link.name + "_collision"
                     : link.name + "_collision_" + std::to_string(collision_index + 1);
    node_data.parent = parent;
    AddTransformProperties(node_data, collision.origin_xyz, collision.origin_rpy);

    if (collision.geometry_type == CollisionGeometryType::Box) {
        Ref<BoxShape3D> shape = MakeRef<BoxShape3D>();
        shape->SetSize(collision.box_size);
        AddProperty(node_data, "shape", dynamic_pointer_cast<Shape3D>(shape));
    } else if (collision.geometry_type == CollisionGeometryType::Sphere) {
        Ref<SphereShape3D> shape = MakeRef<SphereShape3D>();
        shape->SetRadius(static_cast<float>(collision.radius));
        AddProperty(node_data, "shape", dynamic_pointer_cast<Shape3D>(shape));
    } else if (collision.geometry_type == CollisionGeometryType::Cylinder) {
        Ref<CylinderShape3D> shape = MakeRef<CylinderShape3D>();
        shape->SetRadius(static_cast<float>(collision.radius));
        shape->SetHeight(static_cast<float>(collision.height));
        AddProperty(node_data, "shape", dynamic_pointer_cast<Shape3D>(shape));
    }

    return node_data;
}

SceneState::NodeData MakeJointNode(const JointImportData& joint, int parent) {
    SceneState::NodeData node_data;
    node_data.type = "Joint3D";
    node_data.name = joint.name;
    node_data.parent = parent;
    AddTransformProperties(node_data, joint.origin_xyz, joint.origin_rpy);
    AddProperty(node_data, "joint_type", joint.type);
    AddProperty(node_data, "parent_link", joint.parent_link);
    AddProperty(node_data, "child_link", joint.child_link);
    AddProperty(node_data, "axis", joint.axis);
    AddProperty(node_data, "lower_limit", joint.lower_limit);
    AddProperty(node_data, "upper_limit", joint.upper_limit);
    AddProperty(node_data, "effort_limit", joint.effort_limit);
    AddProperty(node_data, "velocity_limit", joint.velocity_limit);
    return node_data;
}

} // namespace

Ref<Resource> ResourceFormatLoaderURDF::Load(const std::string& path,
                                             const std::string& original_path,
                                             CacheMode cache_mode) {
    (void)cache_mode;

    const std::string input_path = ResolveInputPath(path);
    const std::string xml = ReadTextFile(input_path);
    if (xml.empty()) {
        return {};
    }

    std::smatch robot_match;
    std::string robot_name = "Robot";
    if (std::regex_search(xml, robot_match, std::regex(R"(<\s*robot\b([^>]*)>)", std::regex::icase))) {
        const std::string parsed_name = GetAttribute(robot_match[1].str(), "name");
        if (!parsed_name.empty()) {
            robot_name = parsed_name;
        }
    }

    const std::vector<LinkImportData> links = ParseLinks(xml, input_path);
    const std::vector<JointImportData> joints = ParseJoints(xml);
    if (links.empty()) {
        LOG_ERROR("URDF file {} contains no links.", path);
        return {};
    }

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();
    const int robot_index = state->AddNode(MakeRobotNode(robot_name, original_path.empty() ? path : original_path));

    std::unordered_map<std::string, LinkImportData> link_by_name;
    for (const LinkImportData& link : links) {
        link_by_name.emplace(link.name, link);
    }

    std::unordered_multimap<std::string, JointImportData> joints_by_parent;
    std::unordered_set<std::string> child_links;
    for (const JointImportData& joint : joints) {
        joints_by_parent.emplace(joint.parent_link, joint);
        child_links.insert(joint.child_link);
    }

    std::unordered_map<std::string, int> emitted_links;
    auto emit_link = [&](const std::string& link_name, int parent, auto&& emit_link_ref) -> int {
        auto emitted = emitted_links.find(link_name);
        if (emitted != emitted_links.end()) {
            return emitted->second;
        }

        auto link_iter = link_by_name.find(link_name);
        if (link_iter == link_by_name.end()) {
            return -1;
        }

        const LinkImportData& link = link_iter->second;
        const int link_index = state->AddNode(MakeLinkNode(link, parent));
        emitted_links[link_name] = link_index;

        for (std::size_t i = 0; i < link.visuals.size(); ++i) {
            state->AddNode(MakeVisualNode(link, link.visuals[i], static_cast<int>(i), link_index));
        }
        for (std::size_t i = 0; i < link.collisions.size(); ++i) {
            state->AddNode(MakeCollisionNode(link, link.collisions[i], static_cast<int>(i), link_index));
        }

        auto range = joints_by_parent.equal_range(link_name);
        for (auto joint_iter = range.first; joint_iter != range.second; ++joint_iter) {
            const JointImportData& joint = joint_iter->second;
            const int joint_index = state->AddNode(MakeJointNode(joint, link_index));
            emit_link_ref(joint.child_link, joint_index, emit_link_ref);
        }

        return link_index;
    };

    for (const LinkImportData& link : links) {
        if (!child_links.contains(link.name)) {
            emit_link(link.name, robot_index, emit_link);
        }
    }

    for (const LinkImportData& link : links) {
        emit_link(link.name, robot_index, emit_link);
    }

    return packed_scene;
}

void ResourceFormatLoaderURDF::GetRecognizedExtensionsForType(const std::string& type,
                                                              std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

bool ResourceFormatLoaderURDF::RecognizePath(const std::string& path, const std::string& type_hint) const {
    if (!type_hint.empty() && type_hint != "PackedScene") {
        return false;
    }

    const std::string normalized_path = ToLower(path);
    return normalized_path.ends_with(".urdf") || normalized_path.ends_with(".xml");
}

void ResourceFormatLoaderURDF::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("urdf");
    extensions->push_back("xml");
}

bool ResourceFormatLoaderURDF::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<ResourceFormatLoaderURDF>("ResourceFormatLoaderURDF")
            .constructor()(CtorAsRawPtr);

};
