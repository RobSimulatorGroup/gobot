/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {


bool ResourceFormatLoader::Exists(const String &path) const {
    return QFile::exists(path);
}

void ResourceFormatLoader::GetRecognizedExtensionsForType(const String &type, std::vector<String> *extensions) const {
    if (type.isEmpty() || HandlesType(type)) {
        GetRecognizedExtensions(extensions);
    }
}

bool ResourceFormatLoader::RecognizePath(const String &path, const String &type_hint) const {

    String extension = GetFileExtension(path);

    std::vector<String> extensions;
    if (type_hint.isEmpty()) {
        GetRecognizedExtensions(&extensions);
    } else {
        GetRecognizedExtensionsForType(type_hint, &extensions);
    }

    if (std::ranges::any_of(extensions, [&](auto const& ext){ return ext.toLower() == extension.toLower(); })) {
        return true;
    }

    return false;
}


std::deque<Ref<ResourceFormatLoader>> ResourceLoader::s_loaders;

bool ResourceLoader::s_timestamp_on_load = false;

Ref<Resource> ResourceLoader::LoadImpl(const String &path,
                                       const String &type_hint,
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

    ERR_FAIL_COND_V_MSG(!QFile::exists(path), {}, fmt::format("Resource file not found: {}.", path));
    LOG_ERROR("No loader found for resource: {}.", path);
    return {};
}

// TODO(wqq): Add multi-thread loader
Ref<Resource> ResourceLoader::Load(const String &path,
                                   const String &type_hint,
                                   ResourceFormatLoader::CacheMode cache_mode) {
    String local_path = ValidateLocalPath(path);

    Ref<Resource> res = LoadImpl(path, type_hint, cache_mode);
    ERR_FAIL_COND_V_MSG(!res, {}, fmt::format("Loading resource: {} failed.", path));

    if (res.IsValid()) {
        if (cache_mode != ResourceFormatLoader::CacheMode::Ignore) {
            res->SetPath(local_path);
        }
    }
    return res;
}

bool ResourceLoader::Exists(const String &path, const String &type_hint) {
    String local_path = ValidateLocalPath(path);

    if (ResourceCache::Has(local_path)) {
        return true; // If cached, it probably exists
    }

    // Try all loaders and pick the first match for the type hint
    for(const auto& loader: s_loaders) {
        if (!loader->RecognizePath(path, type_hint)) {
            continue;
        }

        if (loader->Exists(path)) {
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