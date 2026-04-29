/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/resource_format_mesh.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/resources/array_mesh.hpp"

#ifdef GOBOT_HAS_ASSIMP
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif

namespace gobot {

namespace {

#ifdef GOBOT_HAS_ASSIMP
void AddMeshRecursive(const aiScene* scene,
                      const aiNode* node,
                      std::vector<Vector3>& vertices,
                      std::vector<uint32_t>& indices) {
    if (scene == nullptr || node == nullptr) {
        return;
    }

    for (unsigned int mesh_index = 0; mesh_index < node->mNumMeshes; ++mesh_index) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[mesh_index]];
        const uint32_t base_vertex = static_cast<uint32_t>(vertices.size());
        for (unsigned int vertex_index = 0; vertex_index < mesh->mNumVertices; ++vertex_index) {
            const aiVector3D& vertex = mesh->mVertices[vertex_index];
            vertices.emplace_back(vertex.x, vertex.y, vertex.z);
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
        AddMeshRecursive(scene, node->mChildren[child_index], vertices, indices);
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
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
                                             aiProcess_Triangulate |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_GenSmoothNormals |
                                             aiProcess_ImproveCacheLocality);
    if (scene == nullptr || scene->mRootNode == nullptr) {
        LOG_ERROR("Assimp failed to load mesh '{}': {}", path, importer.GetErrorString());
        return {};
    }

    std::vector<Vector3> vertices;
    std::vector<uint32_t> indices;
    AddMeshRecursive(scene, scene->mRootNode, vertices, indices);
    if (vertices.empty() || indices.empty()) {
        LOG_ERROR("Mesh '{}' did not contain renderable triangle geometry.", path);
        return {};
    }

    Ref<ArrayMesh> mesh = MakeRef<ArrayMesh>();
    mesh->SetSurface(std::move(vertices), std::move(indices));
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
    return ResourceFormatLoader::Exists(path);
#endif
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::ResourceFormatLoaderMesh>("ResourceFormatLoaderMesh")
            .constructor()(CtorAsRawPtr);
};
