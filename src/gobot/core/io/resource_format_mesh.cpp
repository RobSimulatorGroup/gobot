/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_format_mesh.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/image_loader_stb.hpp"
#include "gobot/core/math/math_defs.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/resources/array_mesh.hpp"

#ifdef GOBOT_HAS_ASSIMP
#include <assimp/matrix4x4.h>
#include <assimp/config.h>
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif

#include <cmath>
#include <filesystem>

namespace gobot {

namespace {

#ifdef GOBOT_HAS_ASSIMP
bool ReadMaterialColor(const aiMaterial* assimp_material, const char* key, unsigned int type, unsigned int index, Color& out) {
    aiColor4D color;
    if (assimp_material->Get(key, type, index, color) != AI_SUCCESS) {
        return false;
    }

    out = Color(color.r, color.g, color.b, color.a);
    return true;
}

bool ReadMaterialFloat(const aiMaterial* assimp_material, const char* key, unsigned int type, unsigned int index, RealType& out) {
    ai_real value = 0.0f;
    if (assimp_material->Get(key, type, index, value) != AI_SUCCESS) {
        return false;
    }

    out = static_cast<RealType>(value);
    return true;
}

TextureWrap ConvertWrapMode(aiTextureMapMode mode) {
    switch (mode) {
        case aiTextureMapMode_Clamp:
        case aiTextureMapMode_Decal:
            return TextureWrap::ClampToEdge;
        case aiTextureMapMode_Mirror:
            return TextureWrap::MirroredRepeat;
        case aiTextureMapMode_Wrap:
        default:
            return TextureWrap::Repeat;
    }
}

Ref<Image> LoadAssimpImage(const aiScene* scene,
                           const aiString& texture_path,
                           const std::string& base_directory) {
    if (scene == nullptr || texture_path.length == 0) {
        return {};
    }
    if (const aiTexture* embedded = scene->GetEmbeddedTexture(texture_path.C_Str()); embedded != nullptr) {
        if (embedded->mHeight == 0) {
            return ImageLoaderStb::LoadMemory(
                    reinterpret_cast<const std::uint8_t*>(embedded->pcData),
                    static_cast<int>(embedded->mWidth));
        }
        std::vector<std::uint8_t> rgba;
        rgba.reserve(static_cast<std::size_t>(embedded->mWidth) * embedded->mHeight * 4);
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(embedded->mWidth) * embedded->mHeight;
             ++i) {
            const aiTexel& texel = embedded->pcData[i];
            rgba.insert(rgba.end(), {texel.r, texel.g, texel.b, texel.a});
        }
        return MakeRef<Image>(static_cast<int>(embedded->mWidth),
                              static_cast<int>(embedded->mHeight),
                              false,
                              ImageFormat::RGBA8,
                              rgba);
    }

    std::filesystem::path path(texture_path.C_Str());
    if (path.is_relative()) {
        path = std::filesystem::path(base_directory) / path;
    }
    return Image::LoadFromFile(path.lexically_normal().string());
}

Ref<Texture2D> LoadAssimpTexture(const aiScene* scene,
                                 const aiMaterial* material,
                                 aiTextureType type,
                                 const std::string& base_directory) {
    if (scene == nullptr || material == nullptr || material->GetTextureCount(type) == 0) {
        return {};
    }
    aiString path;
    aiTextureMapMode map_mode[3] = {aiTextureMapMode_Wrap, aiTextureMapMode_Wrap, aiTextureMapMode_Wrap};
    if (material->GetTexture(type, 0, &path, nullptr, nullptr, nullptr, nullptr, map_mode) != AI_SUCCESS) {
        return {};
    }
    Ref<Image> image = LoadAssimpImage(scene, path, base_directory);
    if (!image.IsValid()) {
        LOG_WARN("Cannot load material texture '{}'.", path.C_Str());
        return {};
    }
    Ref<Texture2D> texture = MakeRef<Texture2D>(image);
    texture->SetWrapU(ConvertWrapMode(map_mode[0]));
    texture->SetWrapV(ConvertWrapMode(map_mode[1]));
    return texture;
}

Ref<PBRMaterial3D> ExtractPBRMaterial(const aiScene* scene,
                                      const aiMaterial* assimp_material,
                                      const std::string& base_directory) {
    if (assimp_material == nullptr) {
        return {};
    }

    bool has_material_property = false;
    Ref<PBRMaterial3D> material = MakeRef<PBRMaterial3D>();

    Color albedo;
    if (ReadMaterialColor(assimp_material, AI_MATKEY_BASE_COLOR, albedo) ||
        ReadMaterialColor(assimp_material, AI_MATKEY_COLOR_DIFFUSE, albedo)) {
        material->SetAlbedo(albedo);
        has_material_property = true;
    }

    RealType scalar = 0.0f;
    if (ReadMaterialFloat(assimp_material, AI_MATKEY_METALLIC_FACTOR, scalar)) {
        material->SetMetallic(scalar);
        has_material_property = true;
    }
    if (ReadMaterialFloat(assimp_material, AI_MATKEY_ROUGHNESS_FACTOR, scalar)) {
        material->SetRoughness(scalar);
        has_material_property = true;
    }
    if (ReadMaterialFloat(assimp_material, AI_MATKEY_SPECULAR_FACTOR, scalar)) {
        material->SetSpecular(scalar);
        has_material_property = true;
    }

    Color emissive;
    if (ReadMaterialColor(assimp_material, AI_MATKEY_COLOR_EMISSIVE, emissive)) {
        material->SetEmissive(emissive);
        has_material_property = true;
    }

    if (Ref<Texture2D> texture = LoadAssimpTexture(
                scene, assimp_material, aiTextureType_BASE_COLOR, base_directory);
        texture.IsValid()) {
        material->SetAlbedoTexture(texture);
        has_material_property = true;
    } else if (Ref<Texture2D> diffuse = LoadAssimpTexture(
                       scene, assimp_material, aiTextureType_DIFFUSE, base_directory);
               diffuse.IsValid()) {
        material->SetAlbedoTexture(diffuse);
        has_material_property = true;
    }

    Ref<Texture2D> metallic_roughness = LoadAssimpTexture(
            scene, assimp_material, aiTextureType_UNKNOWN, base_directory);
    if (!metallic_roughness.IsValid()) {
        metallic_roughness = LoadAssimpTexture(
                scene, assimp_material, aiTextureType_METALNESS, base_directory);
    }
    if (!metallic_roughness.IsValid()) {
        metallic_roughness = LoadAssimpTexture(
                scene, assimp_material, aiTextureType_DIFFUSE_ROUGHNESS, base_directory);
    }
    if (metallic_roughness.IsValid()) {
        material->SetMetallicRoughnessTexture(metallic_roughness);
        has_material_property = true;
    }

    Ref<Texture2D> normal = LoadAssimpTexture(
            scene, assimp_material, aiTextureType_NORMALS, base_directory);
    if (!normal.IsValid()) {
        normal = LoadAssimpTexture(scene, assimp_material, aiTextureType_HEIGHT, base_directory);
    }
    if (normal.IsValid()) {
        material->SetNormalTexture(normal);
        has_material_property = true;
    }
    if (ReadMaterialFloat(assimp_material,
                          AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_NORMALS, 0),
                          scalar)) {
        material->SetNormalScale(scalar);
        has_material_property = true;
    }

    Ref<Texture2D> occlusion = LoadAssimpTexture(
            scene, assimp_material, aiTextureType_AMBIENT_OCCLUSION, base_directory);
    if (!occlusion.IsValid()) {
        occlusion = LoadAssimpTexture(scene, assimp_material, aiTextureType_LIGHTMAP, base_directory);
    }
    if (occlusion.IsValid()) {
        material->SetOcclusionTexture(occlusion);
        has_material_property = true;
    }
    if (ReadMaterialFloat(assimp_material,
                          AI_MATKEY_GLTF_TEXTURE_STRENGTH(aiTextureType_AMBIENT_OCCLUSION, 0),
                          scalar)) {
        material->SetOcclusionStrength(scalar);
        has_material_property = true;
    }

    if (Ref<Texture2D> texture = LoadAssimpTexture(
                scene, assimp_material, aiTextureType_EMISSIVE, base_directory);
        texture.IsValid()) {
        material->SetEmissiveTexture(texture);
        has_material_property = true;
    }

    aiString alpha_mode;
    if (assimp_material->Get(AI_MATKEY_GLTF_ALPHAMODE, alpha_mode) == AI_SUCCESS) {
        const std::string mode = alpha_mode.C_Str();
        material->SetAlphaMode(mode == "MASK" ? AlphaMode::Mask
                                               : (mode == "BLEND" ? AlphaMode::Blend : AlphaMode::Opaque));
        has_material_property = true;
    }
    if (ReadMaterialFloat(assimp_material, AI_MATKEY_GLTF_ALPHACUTOFF, scalar)) {
        material->SetAlphaCutoff(scalar);
        has_material_property = true;
    }
    int two_sided = 0;
    if (assimp_material->Get(AI_MATKEY_TWOSIDED, two_sided) == AI_SUCCESS) {
        material->SetDoubleSided(two_sided != 0);
        has_material_property = true;
    }

    return has_material_property ? material : Ref<PBRMaterial3D>{};
}

Vector3 TransformDirection(const aiMatrix4x4& transform, const aiVector3D& direction) {
    const aiMatrix3x3 linear(transform);
    const aiVector3D transformed = linear * direction;
    Vector3 result(transformed.x, transformed.y, transformed.z);
    const RealType length = result.norm();
    if (length <= CMP_EPSILON || !result.allFinite()) {
        return Vector3::UnitX();
    }
    return result / length;
}

Vector3 TransformNormal(const aiMatrix4x4& transform, const aiVector3D& normal) {
    aiMatrix3x3 normal_transform(transform);
    normal_transform.Inverse().Transpose();

    const aiVector3D transformed = normal_transform * normal;
    Vector3 result(transformed.x, transformed.y, transformed.z);
    const RealType length = result.norm();
    if (length <= CMP_EPSILON ||
        !std::isfinite(result.x()) ||
        !std::isfinite(result.y()) ||
        !std::isfinite(result.z())) {
        return Vector3::UnitZ();
    }

    return result / length;
}

void AddMeshRecursive(const aiScene* scene,
                      const aiNode* node,
                      const aiMatrix4x4& parent_transform,
                      const std::string& base_directory,
                      MeshSurfaceList& surfaces) {
    if (scene == nullptr || node == nullptr) {
        return;
    }

    const aiMatrix4x4 node_transform = parent_transform * node->mTransformation;

    for (unsigned int mesh_index = 0; mesh_index < node->mNumMeshes; ++mesh_index) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[mesh_index]];
        MeshSurfaceData surface;
        if (mesh->mMaterialIndex < scene->mNumMaterials) {
            surface.material = ExtractPBRMaterial(
                    scene, scene->mMaterials[mesh->mMaterialIndex], base_directory);
        }
        const bool has_normals = mesh->HasNormals();
        const bool has_tangents = mesh->HasTangentsAndBitangents() && has_normals;
        const bool has_uv = mesh->HasTextureCoords(0);
        const bool has_colors = mesh->HasVertexColors(0);
        surface.vertices.reserve(mesh->mNumVertices);
        surface.normals.reserve(has_normals ? mesh->mNumVertices : 0);
        surface.tangents.reserve(has_tangents ? mesh->mNumVertices : 0);
        surface.uv0.reserve(has_uv ? mesh->mNumVertices : 0);
        surface.colors.reserve(has_colors ? mesh->mNumVertices : 0);
        for (unsigned int vertex_index = 0; vertex_index < mesh->mNumVertices; ++vertex_index) {
            const aiVector3D vertex = node_transform * mesh->mVertices[vertex_index];
            surface.vertices.emplace_back(vertex.x, vertex.y, vertex.z);
            if (has_normals) {
                surface.normals.push_back(TransformNormal(node_transform, mesh->mNormals[vertex_index]));
            }
            if (has_tangents) {
                const Vector3 normal = surface.normals.back();
                Vector3 tangent = TransformDirection(node_transform, mesh->mTangents[vertex_index]);
                tangent -= normal * normal.dot(tangent);
                if (tangent.norm() <= CMP_EPSILON || !tangent.allFinite()) {
                    const Vector3 helper = std::abs(normal.z()) < 0.999
                                                   ? Vector3::UnitZ()
                                                   : Vector3::UnitY();
                    tangent = helper.cross(normal).normalized();
                } else {
                    tangent.normalize();
                }
                const Vector3 bitangent = TransformDirection(node_transform, mesh->mBitangents[vertex_index]);
                const RealType handedness = normal.cross(tangent).dot(bitangent) < 0.0 ? -1.0 : 1.0;
                surface.tangents.emplace_back(tangent.x(), tangent.y(), tangent.z(), handedness);
            }
            if (has_uv) {
                const aiVector3D& uv = mesh->mTextureCoords[0][vertex_index];
                surface.uv0.emplace_back(uv.x, uv.y);
            }
            if (has_colors) {
                const aiColor4D& color = mesh->mColors[0][vertex_index];
                surface.colors.emplace_back(color.r, color.g, color.b, color.a);
            }
        }

        for (unsigned int face_index = 0; face_index < mesh->mNumFaces; ++face_index) {
            const aiFace& face = mesh->mFaces[face_index];
            if (face.mNumIndices != 3) {
                continue;
            }
            surface.indices.push_back(face.mIndices[0]);
            surface.indices.push_back(face.mIndices[1]);
            surface.indices.push_back(face.mIndices[2]);
        }
        if (!surface.vertices.empty() && !surface.indices.empty()) {
            surfaces.emplace_back(std::move(surface));
        }
    }

    for (unsigned int child_index = 0; child_index < node->mNumChildren; ++child_index) {
        AddMeshRecursive(scene,
                         node->mChildren[child_index],
                         node_transform,
                         base_directory,
                         surfaces);
    }
}
#endif

} // namespace

Ref<Resource> ResourceFormatLoaderMesh::Load(const std::string& path,
                                             const std::string& original_path,
                                             CacheMode cache_mode) {
    (void)original_path;
    (void)cache_mode;

#ifndef GOBOT_HAS_ASSIMP
    LOG_ERROR("Cannot load mesh '{}': Gobot was built without Assimp support.", path);
    return {};
#else
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    Assimp::Importer importer;
    // Robotics assets authored for URDF are already in the URDF/world up-axis convention.
    // Assimp's default Collada root-axis conversion would cancel Blender-exported node transforms.
    importer.SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, 1);
    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80.0f);
    const aiScene* scene = importer.ReadFile(global_path,
                                             aiProcess_Triangulate |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_DropNormals |
                                             aiProcess_GenSmoothNormals |
                                             aiProcess_CalcTangentSpace |
                                             aiProcess_ImproveCacheLocality);
    if (scene == nullptr || scene->mRootNode == nullptr) {
        LOG_ERROR("Assimp failed to load mesh '{}': {}", path, importer.GetErrorString());
        return {};
    }

    MeshSurfaceList surfaces;
    AddMeshRecursive(scene,
                     scene->mRootNode,
                     aiMatrix4x4(),
                     GetBaseDir(global_path),
                     surfaces);
    if (surfaces.empty()) {
        LOG_ERROR("Mesh '{}' did not contain renderable triangle geometry.", path);
        return {};
    }

    Ref<ArrayMesh> mesh = MakeRef<ArrayMesh>();
    mesh->SetSurfaces(std::move(surfaces));
    return mesh;
#endif
}

void ResourceFormatLoaderMesh::GetRecognizedExtensionsForType(const std::string& type,
                                                              std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

void ResourceFormatLoaderMesh::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("dae");
    extensions->push_back("obj");
    extensions->push_back("stl");
    extensions->push_back("fbx");
    extensions->push_back("ply");
    extensions->push_back("glb");
    extensions->push_back("gltf");
}

bool ResourceFormatLoaderMesh::HandlesType(const std::string& type) const {
    return type.empty() || type == "Mesh" || type == "ArrayMesh";
}

bool ResourceFormatLoaderMesh::Exists(const std::string& path) const {
#ifndef GOBOT_HAS_ASSIMP
    (void)path;
    return false;
#else
    return std::filesystem::exists(ProjectSettings::GetInstance()->GlobalizePath(path));
#endif
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::ResourceFormatLoaderMesh>("ResourceFormatLoaderMesh")
            .constructor()(CtorAsRawPtr);
};
