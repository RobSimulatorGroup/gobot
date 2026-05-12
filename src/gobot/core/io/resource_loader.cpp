/*
 * Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {


bool ResourceFormatLoader::Exists(const std::string &path) const {
    return std::filesystem::exists(path);
}

void ResourceFormatLoader::GetRecognizedExtensionsForType(const std::string &type, std::vector<std::string> *extensions) const {
    if (type.empty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

bool ResourceFormatLoader::RecognizePath(const std::string &path, const std::string &type_hint) const {

    std::string extension = GetFileExtension(path);

    std::vector<std::string> extensions;
    if (type_hint.empty()) {
        GetRecognizedExtensions(&extensions);
    } else {
        GetRecognizedExtensionsForType(type_hint, &extensions);
    }

    if (std::ranges::any_of(extensions, [&](auto const& ext){ return ToLower(ext) == ToLower(extension); })) {
        return true;
    }

    return false;
}


std::deque<Ref<ResourceFormatLoader>> ResourceLoader::s_loaders;

bool ResourceLoader::s_timestamp_on_load = false;

Ref<Resource> ResourceLoader::LoadImpl(const std::string &path,
                                       const std::string &type_hint,
                                       ResourceFormatLoader::CacheMode cache_mode) {
    // Try all loaders and pick the first match for the type hint
    bool found = false;
    for(const auto& loader: s_loaders) {
        if (!loader->RecognizePath(path, type_hint)) {
            continue;
        }
        found = true;
        Ref<Resource> res = loader->Load(path, path, cache_mode);
        if (!res) {
            continue;
        }
        return res;
    }

    ERR_FAIL_COND_V_MSG(!found, {},
                        fmt::format("Failed loading resource: {}. Make sure resources have been imported by opening the project in the editor at least once.", path));

    ERR_FAIL_COND_V_MSG(!std::filesystem::exists(path), {}, fmt::format("Resource file not found: {}.", path));
    LOG_ERROR("No loader found for resource: {}.", path);
    return {};
}

// TODO(wqq): Add multi-thread loader
Ref<Resource> ResourceLoader::Load(const std::string &path,
                                   const std::string &type_hint,
                                   ResourceFormatLoader::CacheMode cache_mode) {
    std::string local_path = ValidateLocalPath(path);
    std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);

    if (cache_mode == ResourceFormatLoader::CacheMode::Reuse) {
        Ref<Resource> cached_resource = ResourceCache::GetRef(local_path);
        if (cached_resource.IsValid()) {
            LOG_TRACE("ResourceLoader reused cached resource '{}' type '{}'.",
                      local_path,
                      cached_resource->GetClassStringName());
            return cached_resource;
        }
    }

    LOG_TRACE("ResourceLoader loading '{}' type_hint '{}' cache_mode {}.",
              local_path,
              type_hint,
              static_cast<int>(cache_mode));
    Ref<Resource> res = LoadImpl(global_path, type_hint, cache_mode);
    ERR_FAIL_COND_V_MSG(!res, {}, fmt::format("Loading resource: {} failed.", local_path));

    if (res.IsValid()) {
        if (cache_mode != ResourceFormatLoader::CacheMode::Ignore) {
            res->SetPath(local_path, cache_mode == ResourceFormatLoader::CacheMode::Replace);
        }
        LOG_TRACE("ResourceLoader loaded '{}' as type '{}' path '{}'.",
                  local_path,
                  res->GetClassStringName(),
                  res->GetPath());
    }
    return res;
}

bool ResourceLoader::Exists(const std::string &path, const std::string &type_hint) {
    std::string local_path = ValidateLocalPath(path);
    std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);

    if (ResourceCache::Has(local_path)) {
        return true; // If cached, it probably exists
    }

    // Try all loaders and pick the first match for the type hint
    for(const auto& loader: s_loaders) {
        if (!loader->RecognizePath(global_path, type_hint)) {
            continue;
        }

        if (loader->Exists(global_path)) {
            return true;
        }
    }

    return false;
}

void ResourceLoader::AddResourceFormatLoader(const Ref<ResourceFormatLoader>& format_loader, bool at_front) {
    ERR_FAIL_COND_MSG(!format_loader.IsValid(), "It's not a reference to a valid ResourceFormatLoader object.");

    if (at_front) {
        s_loaders.push_front(format_loader);
    } else {
        s_loaders.push_back(format_loader);
    }
}

void ResourceLoader::RemoveResourceFormatLoader(const Ref<ResourceFormatLoader>& format_loader) {
    auto it = std::find(s_loaders.begin(), s_loaders.end(), format_loader);

    ERR_FAIL_COND_MSG(it == s_loaders.end(), "The format saver is not find");

    s_loaders.erase(it);
}

}

GOBOT_REGISTRATION {
    Class_<ResourceLoader>("ResourceLoader");

};
