/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/resource_format_urdf.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {
namespace {

struct LinkImportData {
    std::string name;
    RealType mass{0.0};
    Vector3 center_of_mass{Vector3::Zero()};
    Vector3 inertia_diagonal{Vector3::Zero()};
    Vector3 inertia_off_diagonal{Vector3::Zero()};
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
        ProjectSettings* settings = ProjectSettings::GetInstance();
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

std::vector<LinkImportData> ParseLinks(const std::string& xml) {
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

SceneState::NodeData MakeJointNode(const JointImportData& joint, int parent) {
    SceneState::NodeData node_data;
    node_data.type = "Joint3D";
    node_data.name = joint.name;
    node_data.parent = parent;
    AddProperty(node_data, "position", joint.origin_xyz);
    AddProperty(node_data, "rotation_degrees", Vector3{
            RAD_TO_DEG(joint.origin_rpy.x()),
            RAD_TO_DEG(joint.origin_rpy.y()),
            RAD_TO_DEG(joint.origin_rpy.z())});
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

    const std::vector<LinkImportData> links = ParseLinks(xml);
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

        const int link_index = state->AddNode(MakeLinkNode(link_iter->second, parent));
        emitted_links[link_name] = link_index;

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

void ResourceFormatLoaderURDF::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("urdf");
}

bool ResourceFormatLoaderURDF::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<ResourceFormatLoaderURDF>("ResourceFormatLoaderURDF")
            .constructor()(CtorAsRawPtr);

};
