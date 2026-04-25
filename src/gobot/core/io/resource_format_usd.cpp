/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/resource_format_usd.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

#include <unordered_map>

#ifdef GOBOT_HAS_OPENUSD
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#endif

namespace gobot {

bool ResourceFormatLoaderUSD::IsOpenUSDAvailable() {
#ifdef GOBOT_HAS_OPENUSD
    return true;
#else
    return false;
#endif
}

Ref<Resource> ResourceFormatLoaderUSD::Load(const std::string& path,
                                            const std::string& original_path,
                                            CacheMode cache_mode) {
    (void)original_path;
    (void)cache_mode;

#ifndef GOBOT_HAS_OPENUSD
    LOG_ERROR("OpenUSD support is disabled. Reconfigure with -DGOB_BUILD_OPENUSD=ON to load {}.", path);
    return {};
#else
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(global_path);
    if (!stage) {
        LOG_ERROR("Cannot open USD stage: {}.", path);
        return {};
    }

    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    Ref<SceneState> state = packed_scene->GetState();
    std::unordered_map<std::string, int> prim_to_node;

    for (const pxr::UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsActive() || prim.IsPseudoRoot()) {
            continue;
        }

        SceneState::NodeData node_data;
        node_data.name = prim.GetName().GetString();
        node_data.type = pxr::UsdGeomXformable(prim) ? "Node3D" : "Node";

        pxr::UsdPrim parent = prim.GetParent();
        while (parent && !parent.IsPseudoRoot()) {
            auto it = prim_to_node.find(parent.GetPath().GetString());
            if (it != prim_to_node.end()) {
                node_data.parent = it->second;
                break;
            }
            parent = parent.GetParent();
        }

        const int node_index = state->AddNode(node_data);
        prim_to_node[prim.GetPath().GetString()] = node_index;
    }

    return packed_scene;
#endif
}

void ResourceFormatLoaderUSD::GetRecognizedExtensionsForType(const std::string& type,
                                                             std::vector<std::string>* extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

void ResourceFormatLoaderUSD::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("usd");
    extensions->push_back("usda");
    extensions->push_back("usdc");
}

bool ResourceFormatLoaderUSD::HandlesType(const std::string& type) const {
    return type.empty() || type == "PackedScene";
}

}

GOBOT_REGISTRATION {

    Class_<ResourceFormatLoaderUSD>("ResourceFormatLoaderUSD")
            .constructor()(CtorAsRawPtr)
            .method("is_openusd_available", &ResourceFormatLoaderUSD::IsOpenUSDAvailable);

};
