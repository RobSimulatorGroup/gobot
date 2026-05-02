/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/resource_format_mesh.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/math/math_defs.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/resources/array_mesh.hpp"

#ifdef GOBOT_HAS_ASSIMP
#include <assimp/matrix4x4.h>
#include <assimp/config.h>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif

#include <cmath>

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

Ref<PBRMaterial3D> ExtractPBRMaterial(const aiMaterial* assimp_material) {
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

    return has_material_property ? material : Ref<PBRMaterial3D>{};
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
                      std::vector<Vector3>& vertices,
                      std::vector<uint32_t>& indices,
                      std::vector<Vector3>& normals,
                      Ref<PBRMaterial3D>& imported_material) {
    if (scene == nullptr || node == nullptr) {
        return;
    }

    const aiMatrix4x4 node_transform = parent_transform * node->mTransformation;

    for (unsigned int mesh_index = 0; mesh_index < node->mNumMeshes; ++mesh_index) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[mesh_index]];
        if (!imported_material.IsValid() && mesh->mMaterialIndex < scene->mNumMaterials) {
            imported_material = ExtractPBRMaterial(scene->mMaterials[mesh->mMaterialIndex]);
        }

        const uint32_t base_vertex = static_cast<uint32_t>(vertices.size());
        const bool has_normals = mesh->HasNormals();
        for (unsigned int vertex_index = 0; vertex_index < mesh->mNumVertices; ++vertex_index) {
            const aiVector3D vertex = node_transform * mesh->mVertices[vertex_index];
            vertices.emplace_back(vertex.x, vertex.y, vertex.z);
            if (has_normals) {
                normals.push_back(TransformNormal(node_transform, mesh->mNormals[vertex_index]));
            }
        }

        for (unsigned int face_index = 0; face_index < mesh->mNumFaces; ++face_index) {
            const aiFace& face = mesh->mFaces[face_index];
            if (face.mNumIndices != 3) {
                continue;
            }
            indices.push_back(base_vertex + face.mIndices[0]);
            indices.push_back(base_vertex + face.mIndices[1]);
            indices.push_back(base_vertex + face.mIndices[2]);
        }
    }

    for (unsigned int child_index = 0; child_index < node->mNumChildren; ++child_index) {
        AddMeshRecursive(scene,
                         node->mChildren[child_index],
                         node_transform,
                         vertices,
                         indices,
                         normals,
                         imported_material);
    }
}
#endif

} // namespace

Ref<Resource> ResourceFormatLoaderMesh::Load(const std::string& path,
                                             const std::string& original_path,
                                             CacheMode cache_mode) {
    (void)original_path;
    (void)cache_mode;

    if (!RenderServer::HasInstance()) {
        LOG_ERROR("Cannot load mesh '{}' before RenderServer is initialized.", path);
        return {};
    }

#ifndef GOBOT_HAS_ASSIMP
    LOG_ERROR("Cannot load mesh '{}': Gobot was built without Assimp support.", path);
    return {};
#else
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    Assimp::Importer importer;
    // Robotics assets authored for URDF are already in the URDF/world up-axis convention.
    // Assimp's default Collada root-axis conversion would cancel Blender-exported node transforms.
    importer.SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, 1);
    const aiScene* scene = importer.ReadFile(global_path,
                                             aiProcess_Triangulate |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_GenNormals |
                                             aiProcess_ImproveCacheLocality);
    if (scene == nullptr || scene->mRootNode == nullptr) {
        LOG_ERROR("Assimp failed to load mesh '{}': {}", path, importer.GetErrorString());
        return {};
    }

    std::vector<Vector3> vertices;
    std::vector<uint32_t> indices;
    std::vector<Vector3> normals;
    Ref<PBRMaterial3D> material;
    AddMeshRecursive(scene, scene->mRootNode, aiMatrix4x4(), vertices, indices, normals, material);
    if (vertices.empty() || indices.empty()) {
        LOG_ERROR("Mesh '{}' did not contain renderable triangle geometry.", path);
        return {};
    }
    if (normals.size() != vertices.size()) {
        normals.clear();
    }

    Ref<ArrayMesh> mesh = MakeRef<ArrayMesh>();
    mesh->SetSurface(std::move(vertices), std::move(indices), std::move(normals));
    mesh->SetMaterial(material);
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
