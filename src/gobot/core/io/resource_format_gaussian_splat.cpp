/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_gaussian_splat.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/object.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/gaussian_splat_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/gaussian_splat.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gobot {
namespace {

template <typename T>
void AddProperty(SceneState::NodeData& node, const std::string& name, T&& value) {
    node.properties.push_back({name, Variant(std::forward<T>(value))});
}

std::optional<Json> ReadManifest(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        LOG_ERROR("Cannot open Gaussian manifest '{}'.", path);
        return std::nullopt;
    }
    try {
        Json manifest;
        stream >> manifest;
        return manifest;
    } catch (const std::exception& exception) {
        LOG_ERROR("Cannot parse Gaussian manifest '{}': {}", path, exception.what());
        return std::nullopt;
    }
}

std::string ResolveReference(const std::string& manifest_path, const std::string& reference) {
    if (reference.starts_with("res://")) {
        return ProjectSettings::HasInstance()
                       ? ProjectSettings::GetInstance()->GlobalizePath(reference)
                       : reference.substr(6);
    }
    const std::filesystem::path reference_path(reference);
    if (reference_path.is_absolute()) return reference_path.lexically_normal().string();
    return (std::filesystem::path(manifest_path).parent_path() / reference_path)
            .lexically_normal().string();
}

std::string LocalizeReference(const std::string& path) {
    return ProjectSettings::HasInstance()
                   ? ProjectSettings::GetInstance()->LocalizePath(path)
                   : path;
}

bool IsManifestVersionOne(const Json& value) {
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>() == 1u;
    }
    if (value.is_number_integer()) {
        return value.get<std::int64_t>() == 1;
    }
    return false;
}

bool ReadFiniteNumber(const Json& value, double* result) {
    if (!value.is_number() || result == nullptr) return false;
    try {
        *result = value.get<double>();
    } catch (const std::exception&) {
        return false;
    }
    return std::isfinite(*result);
}

bool ParseSourceTransform(const Json& manifest,
                          Vector3* position,
                          Vector3* rotation_degrees,
                          Vector3* scale,
                          std::string* error) {
    double meters_per_unit = 1.0;
    if (manifest.contains("meters_per_unit")) {
        if (!ReadFiniteNumber(manifest["meters_per_unit"], &meters_per_unit)) {
            *error = "meters_per_unit must be a finite positive number";
            return false;
        }
    }
    if (meters_per_unit <= 0.0) {
        *error = "meters_per_unit must be a finite positive number";
        return false;
    }

    Matrix4 matrix = Matrix4::Identity();
    if (manifest.contains("source_to_gobot")) {
        const Json& values = manifest["source_to_gobot"];
        if (!values.is_array() || values.size() != 16u) {
            *error = "source_to_gobot must contain 16 row-major numbers";
            return false;
        }
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                const Json& value = values[static_cast<std::size_t>(row * 4 + column)];
                double component = 0.0;
                if (!ReadFiniteNumber(value, &component)) {
                    *error = "source_to_gobot must contain only finite numbers";
                    return false;
                }
                matrix(row, column) = static_cast<RealType>(component);
            }
        }
    }
    if (!matrix.allFinite() ||
        std::abs(matrix(3, 0)) > CMP_EPSILON || std::abs(matrix(3, 1)) > CMP_EPSILON ||
        std::abs(matrix(3, 2)) > CMP_EPSILON || std::abs(matrix(3, 3) - 1.0) > CMP_EPSILON) {
        *error = "source_to_gobot must be a finite affine matrix";
        return false;
    }

    const Matrix3 linear = matrix.template block<3, 3>(0, 0);
    const Vector3 column_scales{
            linear.col(0).norm(), linear.col(1).norm(), linear.col(2).norm()};
    const RealType uniform_scale = column_scales.mean();
    if (!(uniform_scale > CMP_EPSILON) ||
        (column_scales.array() - uniform_scale).abs().maxCoeff() > 1e-5 * uniform_scale) {
        *error = "source_to_gobot must use a positive uniform scale";
        return false;
    }
    const Matrix3 rotation = linear / uniform_scale;
    if (!(rotation.transpose() * rotation).isApprox(Matrix3::Identity(), 1e-5) ||
        std::abs(rotation.determinant() - 1.0) > 1e-5) {
        *error = "source_to_gobot must contain a proper rigid rotation";
        return false;
    }

    *position = matrix.template block<3, 1>(0, 3) * meters_per_unit;
    *scale = Vector3::Constant(uniform_scale * meters_per_unit);
    if (!position->allFinite() || !scale->allFinite()) {
        *error = "source_to_gobot and meters_per_unit produce a non-finite transform";
        return false;
    }
    Affine3 rotation_transform = Affine3::Identity();
    rotation_transform.linear() = rotation;
    const Vector3 euler = rotation_transform.GetEulerAngle(EulerOrder::SXYZ);
    *rotation_degrees = Vector3{
            RAD_TO_DEG(euler.x()), RAD_TO_DEG(euler.y()), RAD_TO_DEG(euler.z())};
    return true;
}

void ConfigureProxy(Node* node) {
    if (node == nullptr) return;
    node->SetSceneInstance({});
    if (auto* mesh = Object::PointerCastTo<MeshInstance3D>(node); mesh != nullptr) {
        mesh->SetVisibleInRgb(false);
        mesh->SetCastShadow(false);
    }
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        ConfigureProxy(node->GetChild(static_cast<int>(i)));
    }
}

bool AppendPackedScene(const Ref<PackedScene>& packed, const Ref<SceneState>& destination, int parent) {
    if (!packed.IsValid() || !destination.IsValid()) return false;
    const Ref<SceneState> source = packed->GetState();
    std::vector<int> remap(source->GetNodeCount(), -1);
    for (std::size_t index = 0; index < source->GetNodeCount(); ++index) {
        const SceneState::NodeData* source_node = source->GetNodeData(index);
        if (source_node == nullptr || source_node->instance.IsValid()) return false;
        SceneState::NodeData node = *source_node;
        node.parent = source_node->parent < 0
                              ? parent
                              : remap[static_cast<std::size_t>(source_node->parent)];
        if (node.parent < 0) return false;
        remap[index] = destination->AddNode(node);
    }
    return true;
}

Ref<PackedScene> LoadProxyScene(const std::string& localized_path) {
    Ref<PackedScene> source = dynamic_pointer_cast<PackedScene>(
            ResourceLoader::Load(localized_path,
                                 "PackedScene",
                                 ResourceFormatLoader::CacheMode::Reuse));
    if (!source.IsValid()) return {};
    Node* root = source->Instantiate();
    if (root == nullptr) return {};
    ConfigureProxy(root);
    Ref<PackedScene> proxy = MakeRef<PackedScene>();
    const bool packed = proxy->Pack(root);
    Object::Delete(root);
    return packed ? proxy : Ref<PackedScene>{};
}

} // namespace

Ref<Resource> ResourceFormatLoaderGaussianSplat::Load(const std::string& path,
                                                      const std::string& original_path,
                                                      CacheMode cache_mode) {
    (void)cache_mode;
    const std::optional<Json> parsed = ReadManifest(path);
    if (!parsed || !parsed->is_object()) {
        LOG_ERROR("Gaussian manifest '{}' must contain a JSON object.", path);
        return {};
    }
    const Json& manifest = *parsed;
    if (!manifest.contains("__VERSION__") ||
        !IsManifestVersionOne(manifest["__VERSION__"]) ||
        !manifest.contains("__TYPE__") ||
        !manifest["__TYPE__"].is_string() ||
        manifest["__TYPE__"].get<std::string>() != "GaussianSplatScene") {
        LOG_ERROR("Gaussian manifest '{}' must use __VERSION__ 1 and __TYPE__ GaussianSplatScene.", path);
        return {};
    }
    if (!manifest.contains("ply") || !manifest["ply"].is_string() ||
        manifest["ply"].get<std::string>().empty()) {
        LOG_ERROR("Gaussian manifest '{}' requires a non-empty ply path.", path);
        return {};
    }

    Vector3 position = Vector3::Zero();
    Vector3 rotation_degrees = Vector3::Zero();
    Vector3 scale = Vector3::Ones();
    std::string transform_error;
    if (!ParseSourceTransform(manifest, &position, &rotation_degrees, &scale, &transform_error)) {
        LOG_ERROR("Gaussian manifest '{}' is invalid: {}.", path, transform_error);
        return {};
    }

    const std::string ply_path = ResolveReference(path, manifest["ply"].get<std::string>());
    Ref<GaussianSplatResource> splat = MakeRef<GaussianSplatResource>();
    std::string ply_error;
    if (!splat->LoadPly(ply_path, &ply_error)) {
        LOG_ERROR("Gaussian manifest '{}' failed to load '{}': {}", path, ply_path, ply_error);
        return {};
    }
    splat->SetSourcePath(LocalizeReference(ply_path));

    Ref<PackedScene> scene = MakeRef<PackedScene>();
    Ref<SceneState> state = scene->GetState();
    SceneState::NodeData root;
    root.type = "Node3D";
    root.name = std::filesystem::path(path).stem().string();
    AddProperty(root, "position", position);
    AddProperty(root, "rotation_degrees", rotation_degrees);
    AddProperty(root, "scale", scale);
    const int root_index = state->AddNode(root);

    SceneState::NodeData gaussian;
    gaussian.type = "GaussianSplat3D";
    gaussian.name = "GaussianEnvironment";
    gaussian.parent = root_index;
    AddProperty(gaussian, "splat", splat);
    AddProperty(gaussian, "enabled", true);
    state->AddNode(gaussian);

    if (manifest.contains("proxy_scene")) {
        if (!manifest["proxy_scene"].is_string()) {
            LOG_ERROR("Gaussian manifest '{}' proxy_scene must be a string.", path);
            return {};
        }
        const std::string proxy_reference = manifest["proxy_scene"].get<std::string>();
        if (!proxy_reference.empty()) {
            const std::string proxy_path = ResolveReference(path, proxy_reference);
            Ref<PackedScene> proxy = LoadProxyScene(LocalizeReference(proxy_path));
            if (!proxy.IsValid()) {
                LOG_ERROR("Gaussian manifest '{}' failed to load proxy scene '{}'.", path, proxy_path);
                return {};
            }
            if (!AppendPackedScene(proxy, state, root_index)) {
                LOG_ERROR("Gaussian manifest '{}' could not expand proxy scene '{}'.", path, proxy_path);
                return {};
            }
        }
    } else {
        LOG_WARN("Gaussian manifest '{}' has no proxy_scene; depth, AOV occlusion, and collision are unavailable.",
                 original_path.empty() ? path : original_path);
    }
    return scene;
}

bool ResourceFormatLoaderGaussianSplat::RecognizePath(const std::string& path,
                                                      const std::string& type_hint) const {
    return (type_hint.empty() || HandlesType(type_hint)) &&
           ToLower(GetFileExtension(path)) == "gsplat";
}

void ResourceFormatLoaderGaussianSplat::GetRecognizedExtensionsForType(
        const std::string& type, std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) GetRecognizedExtensions(extensions);
}

void ResourceFormatLoaderGaussianSplat::GetRecognizedExtensions(
        std::vector<std::string>* extensions) const {
    extensions->push_back("gsplat");
}

bool ResourceFormatLoaderGaussianSplat::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

}

GOBOT_REGISTRATION {
    Class_<ResourceFormatLoaderGaussianSplat>("ResourceFormatLoaderGaussianSplat")
            .constructor()(CtorAsRawPtr);
}
